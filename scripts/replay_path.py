#!/usr/bin/env python3
"""Replay an OMPL/CBF path in PyBullet and check it against the real meshes.

This is the independent half of the loop. The planner never sees a mesh: it
reasons about 40 spheres and an interpolated distance grid, and drops the
collision checker entirely on the grounds that the barrier certifies every step.
That argument is only as good as the sphere model, the margin and the bake -- so
the path comes back here and is checked against the geometry the planner never
looked at.

Environment and self-collision are reported separately, because the barrier
claims the former and models nothing of the latter. Mixing them makes an
out-of-model gap look like the environment model being wrong.

Examples::

    python scripts/replay_path.py --env shelf --path out/shelf.path
    python scripts/replay_path.py --env shelf --path out/shelf.path --gui --hold
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5_nav import UR5, SimSession, make_env  # noqa: E402
from ur5_nav.envs import ENVIRONMENTS  # noqa: E402
from ur5_nav.sdf import (  # noqa: E402
    open_mount_hole,
    scene_distance,
    scene_primitives,
)
from ur5_nav.spheres import DEFAULT_MARGIN, SpherizedUR5  # noqa: E402


def load_path(path: str) -> np.ndarray:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(v) for v in line.split()])
    if not rows:
        raise ValueError(f"{path} holds no configurations")
    return np.asarray(rows, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env", required=True, help=f"one of {', '.join(sorted(ENVIRONMENTS))}")
    parser.add_argument("--path", required=True, help="path file written by demo_UR5PyBulletScene")
    parser.add_argument("--seed", type=int, default=0, help="must match the exported scene")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--mount-opening", type=float, default=0.30)
    parser.add_argument("--gui", action="store_true", help="animate the replay")
    parser.add_argument("--hold", action="store_true", help="keep the window open at the end")
    parser.add_argument("--save-frame", metavar="PNG", help="save a frame at the worst state")
    args = parser.parse_args()

    configurations = load_path(args.path)

    sim = SimSession(gui=args.gui)
    robot = UR5(sim.client)
    env = make_env(args.env, sim.client, seed=args.seed)
    robot.ignore_collisions_with(env.mount_body, [-1, "base_link", "shoulder_link"])
    arm = SpherizedUR5(sim.client)
    primitives = open_mount_hole(
        scene_primitives(sim.client, [sim.plane_id] + env.obstacles),
        env.mount_body,
        half_width=args.mount_opening,
    )
    sim.set_camera(**env.camera)
    for goal in env.goals:
        sim.draw_marker(goal.position, rgb=(0.15, 0.85, 0.25), radius=0.028)

    print(f"replaying {len(configurations)} configurations from {args.path} in '{args.env}'")

    env_hits = 0
    self_hits = 0
    worst_env = np.inf
    worst_self = np.inf
    worst_index = 0
    worst_contact = None
    worst_barrier = np.inf
    trace = []

    for i, q in enumerate(configurations):
        robot.set_config(q)
        # Environment and self-collision are judged apart: the barrier claims the
        # former and models nothing of the latter.
        clearance = robot.clearance(obstacles=env.obstacles, max_distance=0.5, include_self=False)
        if clearance < worst_env:
            worst_env, worst_index = clearance, i
            report = robot.collision_report(
                obstacles=env.obstacles, margin=max(clearance, 0.0) + 1e-4, include_self=False
            )
            worst_contact = report[0] if report else None
        if clearance <= 0.0:
            env_hits += 1

        own = robot.self_clearance()
        worst_self = min(worst_self, own)
        if own <= 0.0:
            self_hits += 1

        model = arm.at(q)
        worst_barrier = min(
            worst_barrier,
            float(model.barrier(scene_distance(primitives, model.centers), args.margin).min()),
        )

        trace.append(robot.ee_pose()[0])
        if args.gui:
            sim.step(1)
            import time

            time.sleep(0.01)

    if args.gui and len(trace) > 1:
        sim.draw_trace(trace, rgb=(0.2, 0.6, 1.0))

    total = len(configurations)
    print(f"  barrier, spheres vs grid (what the planner saw): min h = {worst_barrier:+.4f} m")
    print(f"  environment, real meshes (never seen):           min   = {worst_env:+.4f} m"
          f"   [{env_hits}/{total} states in collision]")
    print(f"  self-collision, real meshes (not modelled):      min   = {worst_self:+.4f} m"
          f"   [{self_hits}/{total} states in collision]")
    if worst_contact is not None:
        print(f"  closest environment pair at state {worst_index}: {worst_contact}")

    if args.save_frame:
        robot.set_config(configurations[worst_index])
        sim.save_frame(args.save_frame, **env.camera)
        print(f"  saved {args.save_frame} (worst state {worst_index})")

    sound = worst_barrier <= 0.0 or worst_env > 0.0
    if sound:
        print(
            f"  -> environment: barrier conservative, h > 0 kept the meshes "
            f"{worst_env * 1000:.1f} mm clear"
        )
    else:
        print(
            "  -> environment: UNSOUND. The barrier reported safe but the meshes collide, so "
            "the sphere model, the margin or the bake does not cover what it claims."
        )
    if self_hits:
        print(
            f"  -> self-collision: {self_hits} state(s) overlap by up to "
            f"{-worst_self * 1000:.1f} mm. ClearanceBarrier models robot-vs-environment "
            "only, so removing the state validity checker removed the only thing checking "
            "this. Needs a self-collision barrier term or a checker kept for these pairs."
        )

    if args.gui and args.hold:
        print("  (window open; Ctrl-C to exit)")
        try:
            while True:
                sim.step(1)
        except KeyboardInterrupt:
            pass
    sim.close()
    return 0 if sound and not self_hits else 1


if __name__ == "__main__":
    raise SystemExit(main())
