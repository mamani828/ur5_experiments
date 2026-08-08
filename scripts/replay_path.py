#!/usr/bin/env python3
"""Replay UR5 path with goal markers and visualization.

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

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from ur5_nav import UR5, SimSession, make_env  # noqa: E402
from ur5_nav.envs import ENVIRONMENTS  # noqa: E402


def load_path(path: str) -> np.ndarray:
    """Load configuration path from file."""
    rows = []

    with open(path) as f:
        for line in f:
            line = line.strip()

            if line and not line.startswith("#"):
                rows.append([float(v) for v in line.split()])

    if not rows:
        raise ValueError(f"{path} holds no configurations")

    return np.asarray(rows, dtype=float)


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

    args = parser.parse_args()

    if args.speed <= 0:
        parser.error("--speed must be greater than zero")

    if args.fps <= 0:
        parser.error("--fps must be greater than zero")

    if args.segment_time <= 0:
        parser.error("--segment-time must be greater than zero")

    # ------------------------------------------------------------
    # Load path + simulation
    # ------------------------------------------------------------

    configurations = load_path(args.path)

    sim = SimSession(gui=args.gui)
    robot = UR5(sim.client)
    env = make_env(args.env, sim.client, seed=0)

    sim.set_camera(**env.camera)

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
        f"Replaying {len(configurations)} configurations "
        f"from {args.path} in '{args.env}'"
    )

    print(f"Goals: {len(env.goals)}")

    # ------------------------------------------------------------
    # Smooth replay + live end-effector trajectory
    # ------------------------------------------------------------

    trace = []

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

    # ------------------------------------------------------------
    # Single-state path
    # ------------------------------------------------------------

    if len(configurations) == 1:
        robot.set_config(configurations[0])

        ee_pos = np.asarray(
            robot.ee_pose()[0],
            dtype=float,
        )

        trace.append(ee_pos.copy())

        if args.gui:
            sim.step(1)

    # ------------------------------------------------------------
    # Multi-state path
    # ------------------------------------------------------------

    else:
        for segment_idx in range(len(configurations) - 1):
            q0 = configurations[segment_idx]
            q1 = configurations[segment_idx + 1]

            # Joint-space interpolation gives much smoother playback
            # than jumping directly between planner configurations.
            for alpha in np.linspace(
                0.0,
                1.0,
                frames_per_segment,
                endpoint=False,
            ):
                q = (1.0 - alpha) * q0 + alpha * q1

                robot.set_config(q)

                ee_pos = np.asarray(
                    robot.ee_pose()[0],
                    dtype=float,
                )

                # ------------------------------------------------
                # Draw trajectory progressively
                # ------------------------------------------------

                if trace:
                    if (
                        args.gui
                        and frame_count % trace_stride == 0
                    ):
                        sim.draw_trace(
                            [
                                trace[-1],
                                ee_pos,
                            ],
                            rgb=trajectory_color,
                            width=trajectory_width,
                        )

                trace.append(ee_pos.copy())

                # ------------------------------------------------
                # Advance visualization
                # ------------------------------------------------

                if args.gui:
                    sim.step(1)
                    time.sleep(frame_delay)

                frame_count += 1

        # --------------------------------------------------------
        # Exact final configuration
        # --------------------------------------------------------

        robot.set_config(configurations[-1])

        final_pos = np.asarray(
            robot.ee_pose()[0],
            dtype=float,
        )

        if args.gui and trace:
            sim.draw_trace(
                [
                    trace[-1],
                    final_pos,
                ],
                rgb=trajectory_color,
                width=trajectory_width,
            )

        trace.append(final_pos.copy())

        if args.gui:
            sim.step(1)

    # ------------------------------------------------------------
    # Highlight trajectory start/end
    # ------------------------------------------------------------

    if args.gui and trace:
        # Start marker
        sim.draw_marker(
            trace[0],
            rgb=(0.25, 0.85, 1.00),
            radius=0.025,
        )

        # End marker
        sim.draw_marker(
            trace[-1],
            rgb=(1.00, 0.80, 0.15),
            radius=0.030,
        )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print(f"Replayed {len(configurations)} path states")

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
