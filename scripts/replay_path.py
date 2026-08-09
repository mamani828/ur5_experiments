#!/usr/bin/env python3
"""Replay UR5 path with goal markers and visualization.

A path file holds one motion per goal, each re-prefixed with the scene's start
configuration: the arm is *reset* between goals, it does not fly back. Those seams are
not edges of any plan, so they are neither interpolated nor drawn -- the arm snaps back
and the next motion gets its own trace, in its goal's colour.

The trace turns red where the arm folds into itself. `ClearanceBarrier` models
robot-vs-environment only, so a path can be certified and still self-collide, and this
says where. The verdict is PyBullet's own -- `performCollisionDetection()` under the URDF's
self-collision flags, no pruned pair list of ours in the way. That makes it a different
question from `scripts/audit_self_collision.py`, which asks the same geometry through
`UR5`'s hand-built ignore set; expect the counts to differ.

Examples::

    python scripts/replay_path.py --env shelf --path out/shelf.path --gui
    python scripts/replay_path.py --env corridor --path out/corridor.path --gui --hold
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pybullet as p

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from ur5_nav import HOME_CONFIG, UR5, SimSession, load_runs, make_env  # noqa: E402
from ur5_nav.envs import ENVIRONMENTS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--env",
        required=True,
        help=f"one of {', '.join(sorted(ENVIRONMENTS))}",
    )

    parser.add_argument(
        "--path",
        required=True,
        help="path file from demo_UR5CBFPlanning",
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="animate in PyBullet GUI",
    )

    parser.add_argument(
        "--hold",
        action="store_true",
        help="keep GUI window open at end",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.6,
        help="playback speed multiplier (0.5 = slower, 2.0 = faster)",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="animation frame rate",
    )

    parser.add_argument(
        "--segment-time",
        type=float,
        default=0.20,
        help="seconds spent moving between path configurations",
    )

    parser.add_argument(
        "--self-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use PyBullet's native self-collision check and draw those stretches red",
    )

    parser.add_argument(
        "--self-check-raw",
        action="store_true",
        help="do not exclude the pairs that already overlap at rest; every state will "
             "report a self-collision, which is what the engine says unfiltered",
    )

    parser.add_argument(
        "--reset-pause",
        type=float,
        default=0.4,
        help="seconds to hold the start pose when resetting for the next goal",
    )

    args = parser.parse_args()

    if args.speed <= 0:
        parser.error("--speed must be greater than zero")

    if args.fps <= 0:
        parser.error("--fps must be greater than zero")

    if args.segment_time <= 0:
        parser.error("--segment-time must be greater than zero")

    if args.reset_pause < 0:
        parser.error("--reset-pause must not be negative")

    # ------------------------------------------------------------
    # Load path + simulation
    # ------------------------------------------------------------

    # One motion per goal, not one path: see `ur5_nav.paths.split_runs`.
    runs = load_runs(args.path)
    waypoints = sum(len(run) for run in runs)

    sim = SimSession(gui=args.gui)
    robot = UR5(sim.client)
    env = make_env(args.env, sim.client, seed=0)

    sim.set_camera(**env.camera)

    # ------------------------------------------------------------
    # What the engine considers a self-collision
    # ------------------------------------------------------------

    if args.self_check and not args.self_check_raw:
        # `URDF_USE_SELF_COLLISION_EXCLUDE_PARENT` excludes parent-child pairs and
        # nothing more, so links welded to a neighbour through a fixed joint still
        # report contact -- the force-torque sensor against wrist_3, each gripper
        # fingertip sunk into its finger. Those are the URDF's collision meshes
        # overlapping by construction, not the arm folding into itself: left in, they
        # paint every state red and bury the real thing.
        #
        # A pair earns exclusion by *not moving*: sampled across the configuration
        # space, links rigidly joined to each other hold the same penetration depth to
        # the micrometre, because no joint lies between them to change it. Anything
        # whose depth varies is configuration-dependent and stays in, however deep it
        # is at rest. One pose could not tell those apart; this can.
        #
        # The exclusion is applied through PyBullet's own filter, so the verdict during
        # replay is still entirely `performCollisionDetection`'s -- `UR5`'s hand-built
        # ignore set is never consulted.
        rng = np.random.default_rng(0)
        samples = [np.asarray(HOME_CONFIG, dtype=float)] + [
            robot.random_config(rng) for _ in range(8)
        ]

        depths: dict[tuple[int, int], list[float]] = {}
        for q in samples:
            robot.set_config(q)
            p.performCollisionDetection(physicsClientId=sim.client)

            for contact in p.getContactPoints(
                bodyA=robot.body_id, bodyB=robot.body_id, physicsClientId=sim.client
            ):
                if contact[8] < 0.0:
                    depths.setdefault(tuple(sorted((contact[3], contact[4]))), []).append(
                        contact[8]
                    )

        structural = {
            pair: seen[0]
            for pair, seen in depths.items()
            # Present in every sample, at a depth that never budges: welded.
            if len(seen) == len(samples) and max(seen) - min(seen) < 1e-6
        }

        for link_a, link_b in structural:
            p.setCollisionFilterPair(
                robot.body_id, robot.body_id, link_a, link_b, 0, physicsClientId=sim.client
            )

        if structural:
            print(
                f"Excluded {len(structural)} welded pairs -- fixed overlap in the URDF, "
                f"not self-collision:"
            )
            for (link_a, link_b), depth in sorted(structural.items(), key=lambda kv: kv[1]):
                name_a = robot.link_names.get(link_a, link_a)
                name_b = robot.link_names.get(link_b, link_b)
                print(f"  {name_a}/{name_b:38s} {depth * 1000:7.2f} mm in every sample")

    # ------------------------------------------------------------
    # Pretty goal markers
    # ------------------------------------------------------------

    goal_colors = [
        (0.20, 0.95, 0.35),  # green
        (1.00, 0.35, 0.20),  # red/orange
        (0.95, 0.65, 0.15),  # amber
        (0.70, 0.30, 1.00),  # purple
    ]

    if args.gui:
        for i, goal in enumerate(env.goals):
            pos = np.asarray(goal.position, dtype=float)
            color = goal_colors[i % len(goal_colors)]

            # Outer halo
            halo_color = tuple(0.50 * c for c in color)

            sim.draw_marker(
                pos,
                rgb=halo_color,
                radius=0.055,
            )

            # Bright inner goal marker
            sim.draw_marker(
                pos,
                rgb=color,
                radius=0.035,
            )

            # Small vertical goal pin
            pin_top = pos + np.array([0.0, 0.0, 0.09])

            sim.draw_trace(
                [pos, pin_top],
                rgb=color,
            )

            sim.draw_marker(
                pin_top,
                rgb=color,
                radius=0.018,
            )

    print(
        f"Replaying {waypoints} configurations in {len(runs)} motions "
        f"from {args.path} in '{args.env}'"
    )

    print(f"Goals: {len(env.goals)}")

    # ------------------------------------------------------------
    # Smooth replay + live end-effector trajectory
    # ------------------------------------------------------------

    frame_delay = 1.0 / args.fps

    # At speed=1.0, each original path segment takes segment_time.
    segment_time = args.segment_time / args.speed

    frames_per_segment = max(
        2,
        int(segment_time * args.fps),
    )

    # Draw one trajectory segment every N rendered frames.
    # Increase this if there are too many PyBullet debug lines.
# Draw every frame for a dense trajectory
    trace_stride = 1

    trajectory_color = (0.10, 0.65, 1.00)
    trajectory_width = 5.0

    frame_count = 0

    trajectory_color = (0.15, 0.70, 1.00)
    collision_color = (1.00, 0.10, 0.10)

    self_hits = 0
    checked = 0
    worst_self = float("inf")

    def ee_position() -> np.ndarray:
        return np.asarray(robot.ee_pose()[0], dtype=float)

    def self_colliding() -> bool:
        """Does the arm overlap itself at the current configuration?

        PyBullet's own self-collision and nothing else: the URDF is loaded with
        `URDF_USE_SELF_COLLISION | URDF_USE_SELF_COLLISION_EXCLUDE_PARENT`, and
        `performCollisionDetection()` honours exactly those flags. `UR5`'s hand-built
        ignore set is deliberately not consulted here, and neither is the sphere model
        the barrier reasons about -- this is the engine's verdict on its own terms.

        `getContactPoints` reports within the contact margin, so positive separations
        come back too; only real overlap counts as a hit.
        """
        nonlocal self_hits, checked, worst_self

        if not args.self_check:
            return False

        checked += 1

        p.performCollisionDetection(physicsClientId=sim.client)
        contacts = p.getContactPoints(
            bodyA=robot.body_id, bodyB=robot.body_id, physicsClientId=sim.client
        )

        overlap = min((c[8] for c in contacts), default=0.0)

        if overlap >= 0.0:
            return False

        self_hits += 1
        worst_self = min(worst_self, overlap)

        return True

    def motion_color(run: np.ndarray) -> tuple[float, float, float]:
        """Colour a motion like the goal it actually ends at.

        Not like `env.goals[run_index]`: the exporter drops goals it cannot certify, so
        the i-th motion in the file need not be the i-th goal of the scene. Asking where
        the motion ends keeps the trace and its marker the same colour whichever goals
        were dropped. Sets the robot's configuration -- callers set it back.
        """
        if not env.goals:
            return trajectory_color

        robot.set_config(run[-1])
        end = ee_position()

        distances = [
            float(np.linalg.norm(end - np.asarray(goal.position, dtype=float)))
            for goal in env.goals
        ]
        nearest = int(np.argmin(distances))

        # A motion that ends nowhere near a goal marker is not that goal's motion.
        if distances[nearest] > 0.25:
            return trajectory_color

        return goal_colors[nearest % len(goal_colors)]

    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []

    for run_index, run in enumerate(runs):
        color = motion_color(run)

        # The reset between goals is not a motion. Each run is re-prefixed with the
        # scene's start configuration, and the arm is *put* back there -- it never
        # travels. Interpolating across that seam would animate a straight line from
        # deep inside a shelf back to the home pose, sweeping through everything in
        # between: a segment no planner produced and nothing ever collision-checked.
        # So snap to the start, and begin a fresh trace so no line spans the cut.
        robot.set_config(run[0])
        trace = [ee_position()]
        starts.append(trace[0].copy())

        hits_before = self_hits
        was_colliding = self_colliding()

        if args.gui:
            sim.step(1)

            if run_index:
                # A visible beat, so the reset reads as a reset and not as motion.
                time.sleep(args.reset_pause)

        for segment_idx in range(len(run) - 1):
            q0 = run[segment_idx]
            q1 = run[segment_idx + 1]

            # Joint-space interpolation gives much smoother playback
            # than jumping directly between planner configurations.
            for alpha in np.linspace(
                0.0,
                1.0,
                frames_per_segment,
                endpoint=False,
            ):
                frame_started = time.time()

                q = (1.0 - alpha) * q0 + alpha * q1

                robot.set_config(q)

                ee_pos = ee_position()

                colliding = self_colliding()

                # ------------------------------------------------
                # Draw trajectory progressively
                # ------------------------------------------------

                if args.gui and frame_count % trace_stride == 0:
                    # A segment is red if either end of it is in collision, so the
                    # stretch that is drawn red covers every state that is.
                    bad = colliding or was_colliding

                    sim.draw_trace(
                        [
                            trace[-1],
                            ee_pos,
                        ],
                        rgb=collision_color if bad else color,
                        width=trajectory_width * (1.5 if bad else 1.0),
                    )

                trace.append(ee_pos.copy())
                was_colliding = colliding

                # ------------------------------------------------
                # Advance visualization
                # ------------------------------------------------

                if args.gui:
                    sim.step(1)
                    # Checking costs a few ms a frame; charge it against the frame
                    # rather than on top, so --fps still means what it says.
                    time.sleep(max(0.0, frame_delay - (time.time() - frame_started)))

                frame_count += 1

        # --------------------------------------------------------
        # Exact final configuration
        # --------------------------------------------------------

        robot.set_config(run[-1])

        final_pos = ee_position()

        colliding = self_colliding()

        if args.gui:
            bad = colliding or was_colliding

            sim.draw_trace(
                [
                    trace[-1],
                    final_pos,
                ],
                rgb=collision_color if bad else color,
                width=trajectory_width * (1.5 if bad else 1.0),
            )

            sim.step(1)

        ends.append(final_pos.copy())

        if args.self_check and self_hits > hits_before:
            print(
                f"  motion {run_index}: {self_hits - hits_before} self-colliding states"
            )

    # ------------------------------------------------------------
    # Highlight where every motion starts, and where each one ends
    # ------------------------------------------------------------

    if args.gui and starts:
        # Start marker -- every run begins at the same configuration, so one will do.
        sim.draw_marker(
            starts[0],
            rgb=(0.25, 0.85, 1.00),
            radius=0.025,
        )

        # End marker per motion, not just for the last one.
        for end in ends:
            sim.draw_marker(
                end,
                rgb=(1.00, 0.80, 0.15),
                radius=0.030,
            )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print(f"Replayed {waypoints} path states in {len(runs)} motions")

    if args.self_check:
        if self_hits:
            # Not a viewing defect: the barrier certifies robot-vs-environment, and
            # nothing in the planner was watching these pairs.
            print(
                f"Self-collision: {self_hits}/{checked} rendered states overlap, "
                f"worst {-worst_self * 1000:.1f} mm (drawn red)"
            )
        else:
            print(f"Self-collision: none in {checked} rendered states")

    print(
        f"Animation: {args.fps:.0f} FPS, "
        f"{segment_time:.3f}s per path segment"
    )

    # ------------------------------------------------------------
    # Hold GUI open
    # ------------------------------------------------------------

    if args.gui and args.hold:
        print("(GUI open; Ctrl-C to exit)")

        try:
            while True:
                sim.step(1)
                time.sleep(frame_delay)

        except KeyboardInterrupt:
            pass

    sim.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
