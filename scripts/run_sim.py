#!/usr/bin/env python3
"""Run a UR5 navigation scenario in PyBullet.

Examples::

    # interactive GUI, shelf environment
    python scripts/run_sim.py --env shelf

    # every environment, headless, save one screenshot each
    python scripts/run_sim.py --env all --headless --save-frames out/

    # physics-driven execution instead of kinematic replay
    python scripts/run_sim.py --env clutter --dynamic --seed 3
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5_nav import UR5, SimSession, make_env  # noqa: E402
from ur5_nav.envs import ENVIRONMENTS  # noqa: E402
from ur5_nav.motion import move_to_config, move_to_goal  # noqa: E402
from ur5_nav.robot import HOME_CONFIG  # noqa: E402


def run_scenario(args, env_name: str, sim: SimSession | None = None) -> dict:
    own_session = sim is None
    if own_session:
        sim = SimSession(gui=not args.headless)

    robot = UR5(sim.client)
    env = make_env(env_name, sim.client, seed=args.seed)
    obstacles = env.obstacles
    if env.mount_body is not None:
        # The base is bolted to the table; that contact is structural.
        robot.ignore_collisions_with(env.mount_body, [-1, "base_link", "shoulder_link"])

    print(f"\n=== {env.name} ===")
    print(f"{env.description}  ({len(obstacles)} obstacle bodies, {len(env.goals)} goals)")

    sim.set_camera(**env.camera)
    robot.set_config(HOME_CONFIG)
    robot.hold()
    if robot.in_collision(obstacles=obstacles):
        print("  ! home configuration is in collision with this scene")
    for goal in env.goals:
        sim.draw_marker(goal.position, rgb=(0.15, 0.85, 0.25), radius=0.028)

    stats = {"env": env.name, "goals": [], "reached": 0, "collided": 0}

    for index, goal in enumerate(env.goals):
        result, q_goal, ik = move_to_goal(
            sim,
            robot,
            goal,
            obstacles=obstacles,
            dynamic=args.dynamic,
            realtime=not args.headless,
            stop_on_collision=False,
        )
        ik_error = ik["position_error"]
        ik_collides = not ik["collision_free"]
        reached = bool(ik["feasible"])

        label = f" ({goal.label})" if goal.label else ""
        print(f"  goal {index} {np.round(goal.position, 3)}{label}: {result.summary()}")
        print(
            f"    IK: error {ik_error * 1000:6.1f} mm, "
            f"approach off by {np.degrees(ik['axis_error']):4.1f} deg, "
            f"goal config {'IN COLLISION' if ik_collides else 'collision-free'}"
            f"{'' if ik['on_target'] else ', POSE NOT REACHED'}"
        )
        for message in result.messages:
            print(f"    {message}")

        stats["goals"].append(
            {
                "index": index,
                "label": goal.label,
                "position": goal.position.tolist(),
                "ik_error_m": ik_error,
                "ik_axis_error_rad": ik["axis_error"],
                "ik_collision_free": not ik_collides,
                "path_collided": result.collided,
                "min_clearance_m": result.min_clearance,
            }
        )
        stats["reached"] += int(reached)
        stats["collided"] += int(result.collided)

        # Return home so each goal is attempted from the same start state.
        move_to_config(
            sim,
            robot,
            HOME_CONFIG,
            obstacles=obstacles,
            dynamic=args.dynamic,
            realtime=not args.headless,
            trace=False,
        )

    if args.save_frames:
        path = os.path.join(args.save_frames, f"{env.name}.png")
        sim.save_frame(path, **env.camera)
        print(f"  saved {path}")

    print(
        f"  -> {stats['reached']}/{len(env.goals)} goal configs reachable & collision-free; "
        f"{stats['collided']}/{len(env.goals)} straight-line paths hit something"
    )

    if own_session:
        if args.hold and sim.gui:
            print("  (window open; Ctrl-C to exit)")
            try:
                while True:
                    sim.step(1)
            except KeyboardInterrupt:
                pass
        sim.close()
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--env",
        default="shelf",
        help=f"environment name, or 'all'. choices: {', '.join(sorted(ENVIRONMENTS))}",
    )
    parser.add_argument("--headless", action="store_true", help="run without a GUI window")
    parser.add_argument("--dynamic", action="store_true", help="step physics instead of teleporting")
    parser.add_argument("--seed", type=int, default=0, help="seed for randomised environments")
    parser.add_argument("--tolerance", type=float, default=0.02, help="IK success threshold, metres")
    parser.add_argument("--save-frames", metavar="DIR", help="write one PNG per environment")
    parser.add_argument("--hold", action="store_true", help="keep the GUI open after the run")
    args = parser.parse_args()

    names = sorted(ENVIRONMENTS) if args.env == "all" else [args.env]
    for name in names:
        if name not in ENVIRONMENTS:
            parser.error(f"unknown env {name!r}; choices: {', '.join(sorted(ENVIRONMENTS))}")

    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)

    all_stats = [run_scenario(args, name) for name in names]

    if len(all_stats) > 1:
        print("\n=== summary ===")
        for s in all_stats:
            total = len(s["goals"])
            print(
                f"  {s['env']:<10} reachable {s['reached']}/{total}   "
                f"straight-line collisions {s['collided']}/{total}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
