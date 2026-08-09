#!/usr/bin/env python3
"""Replay OMPL-planned paths against the real UR5 meshes and count self-collisions.

This is the outermost check on `ompl::cbf::ClearanceBarrier`'s self-collision rows, and
the only one that does not share an assumption with them. The barrier reasons about
VAMP's 40-sphere model; `demos/UR5SelfCollisionAudit.h` audits that same sphere model
from the full pair list rather than the pruned one; this audits the *meshes*, in
PyBullet, with its own hand-built ignore set (`UR5._build_self_ignore_set`). If the
sphere model is too coarse to stand in for the geometry, only this can say so.

It reports environment collisions in the same pass, kept separate on purpose: mixing
them makes a self-collision look like the workspace model being wrong.

    python scripts/audit_self_collision.py --env shelf --path out/shelf.path
    python scripts/audit_self_collision.py --path out/mbm.cbf

`--env` is optional, and leaving it off is not a lesser audit of the self-collision
column: self-collision is a function of the configuration alone, so it says exactly the
same thing with an empty world. That is what makes MotionBenchMaker auditable here at all
-- its scenes are not built in `ur5_nav.envs`, and they do not need to be. Only the
environment column is lost.

The paths are what `demo_UR5PyBulletScene` writes with its third argument and what
`demo_UR5MBMBenchmark` writes with its thirteenth (`<prefix>.cbf`, `<prefix>.rrtc`), so
the usual flow is to plan with self rows on, plan again with them off (the PyBullet
demo's eighth argument, the MBM benchmark's twelfth, e.g. -100), and audit both.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5_nav import UR5, SimSession, load_runs, make_env  # noqa: E402


def densify(path: np.ndarray, resolution: float) -> np.ndarray:
    """Sample between waypoints so a collision cannot hide inside a segment.

    The stored path is already the executed motion -- `executedPath()` replays the
    rollout the planner recorded for each edge -- but its waypoints sit at certified
    step boundaries, which on this branch can be several centimetres apart. Checking
    only those would audit the barrier at exactly the states it chose.
    """
    if resolution <= 0.0:
        return path
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        steps = max(1, int(np.ceil(np.linalg.norm(b - a) / resolution)))
        for i in range(1, steps + 1):
            out.append(a + (b - a) * (i / steps))
    return np.asarray(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default=None,
                        help="environment name, e.g. shelf; omit to audit self-collision "
                             "alone, in an empty world")
    parser.add_argument("--path", required=True, help="path file written by the demo")
    parser.add_argument("--resolution", type=float, default=0.02,
                        help="joint-space spacing to densify to, in radians")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="report contacts closer than this; 0 means real overlap")
    # pillars and clutter are randomised, and the environment column is only
    # meaningful when this matches the seed the grid was baked with. The
    # self-collision column does not care: it is a function of the configuration
    # alone and says the same thing whatever is in the workspace.
    parser.add_argument("--seed", type=int, default=0, help="for randomised environments")
    args = parser.parse_args()

    runs = load_runs(args.path)
    waypoints = sum(len(run) for run in runs)
    states = np.concatenate([densify(run, args.resolution) for run in runs])

    with SimSession(gui=False) as session:
        env = make_env(args.env, session.client, seed=args.seed) if args.env else None
        robot = UR5(session.client, self_collision=True)
        if env is not None and env.mount_body is not None:
            # Same exclusion the exporter and smoke test use: the arm is bolted to the
            # table, so that contact is structural. Missing the -1 here makes every
            # configuration report the same phantom contact.
            robot.ignore_collisions_with(env.mount_body, [-1, "base_link", "shoulder_link"])

        obstacles = list(env.obstacles) if env is not None else []
        selfHits, envHits = 0, 0
        worstSelf, worstEnv = float("inf"), float("inf")
        examples: dict[str, float] = {}

        for q in states:
            hitSelf = hitEnv = False
            for contact in robot.collision_report(q, obstacles=obstacles,
                                                  margin=max(args.margin, 0.05),
                                                  include_self=True):
                # A self contact names both links and reports the robot as the other
                # body; an environment contact names one link and an obstacle body.
                isSelf = contact.other_body == robot.body_id
                distance = contact.distance
                bad = distance < args.margin or distance < 0.0
                if isSelf:
                    worstSelf = min(worstSelf, distance)
                    if bad:
                        hitSelf = True
                        # getClosestPoints reports a self pair once per direction, so
                        # normalise the name or every pair is listed twice.
                        key = "/".join(sorted(contact.link.split("/")))
                        examples[key] = min(examples.get(key, 1e9), distance)
                else:
                    worstEnv = min(worstEnv, distance)
                    hitEnv = hitEnv or bad
            selfHits += 1 if hitSelf else 0
            envHits += 1 if hitEnv else 0

    print(f"{args.env or '(no env)':12s} {args.path}")
    print(f"  {waypoints} waypoints in {len(runs)} motions, densified to {len(states)} "
          f"states at {args.resolution:.3f} rad")
    print(f"  self : {selfHits:6d} colliding states, closest {worstSelf * 1e3:8.2f} mm")
    if env is not None:
        print(f"  env  : {envHits:6d} colliding states, closest {worstEnv * 1e3:8.2f} mm")
    for name, distance in sorted(examples.items(), key=lambda kv: kv[1])[:6]:
        print(f"         {name:48s} {distance * 1e3:8.2f} mm")
    return 1 if selfHits or envHits else 0


if __name__ == "__main__":
    raise SystemExit(main())
