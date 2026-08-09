#!/usr/bin/env python3
"""Bake a PyBullet environment into an SDF grid + planning problem for OMPL.

Writes two files per environment, both consumed by
``demos/UR5PyBulletSceneDemo.cpp`` in the OMPL fork:

    <env>.grid     ompl::sdf::GridSDF::load() -- value per node, no gradients
                   (GridSDF derives those from the interpolant)
    <env>.problem  start config, goal configs, joint limits, metadata

The grid is baked from the scene's *analytic* primitives, not sampled from
collision queries, so the field is a true signed distance and its gradient -- the
thing the CBF actually consumes -- is right.

Goal configurations are chosen to maximise the *barrier* value rather than merely
be collision-free: with a 0.06 m margin over a sphere model that under-covers the
links, a mesh-free configuration is routinely infeasible for the planner.

The bar is the margin the *filter* guards, not the one the path is audited
against. ``ClearanceBarrier::guarding`` adds ``interpolationBuffer`` -- one grid
spacing -- on top, because a filter enforcing h >= 0 against an interpolated
field only delivers h >= -O(voxel). A goal that clears the audited margin but not
the guarded one looks fine here and then blocks every step the planner tries: the
filter would have to break its own invariant to arrive.

Examples::

    python scripts/export_scene.py --env all --out out/
    python scripts/export_scene.py --env shelf --voxel 0.015 --slices
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5_nav.self_pairs import load_self_pairs, load_self_buffer  # noqa: E402
from ur5_nav import UR5, SimSession, make_env  # noqa: E402
from ur5_nav.envs import ENVIRONMENTS  # noqa: E402
from ur5_nav.robot import HOME_CONFIG  # noqa: E402
from ur5_nav.sdf import (  # noqa: E402
    REACHABLE_BOUNDS,
    bake,
    open_mount_hole,
    scene_distance,
    scene_primitives,
)
from ur5_nav.spheres import DEFAULT_MARGIN, SpherizedUR5  # noqa: E402


def export(args, env_name: str) -> dict:
    sim = SimSession(gui=False)
    robot = UR5(sim.client)
    env = make_env(env_name, sim.client, seed=args.seed)
    robot.ignore_collisions_with(env.mount_body, [-1, "base_link", "shoulder_link"])
    arm = SpherizedUR5(sim.client)

    # The robot itself is not an obstacle; goal markers have no collision shape
    # but are excluded explicitly so the field cannot depend on them.
    primitives = scene_primitives(sim.client, [sim.plane_id] + env.obstacles)
    primitives = open_mount_hole(primitives, env.mount_body, half_width=args.mount_opening)

    # The planner's barrier is not only the workspace one. `ClearanceBarrier` carries a
    # row per entry of `UR5::selfPairs()`, each with its own margin calibrated against
    # PyBullet's hulls, and its verdict is the minimum over *all* rows. Scoring goals on
    # the workspace rows alone certified goals the planner then refused: on `empty` the
    # exporter said h = +0.0768 where the planner saw +0.0129, and both planners spent
    # their whole time budget failing to reach them.
    pairs = load_self_pairs()
    pairs.check_alignment(arm.at(HOME_CONFIG).radii)
    self_buffer = load_self_buffer()

    def barrier_parts(q) -> tuple[float, float]:
        """(workspace, self) barrier minima at ``q``."""
        model = arm.at(q)
        world = float(model.barrier(scene_distance(primitives, model.centers), args.margin).min())
        return world, float(pairs.clearance(model.centers).min())

    def barrier_min(q) -> float:
        """The barrier the planner evaluates: the minimum over every row, unbuffered."""
        return min(barrier_parts(q))

    print(f"\n=== {env_name} ===")
    print(f"{len(primitives)} primitives (table opened to {args.mount_opening * 2:.2f} m square)")

    start_h = barrier_min(HOME_CONFIG)
    print(f"start config barrier: h = {start_h:+.4f} m", end="")
    if start_h <= 0.0:
        h, link, center, radius = arm.worst_sphere(HOME_CONFIG, primitives, args.margin)
        print(f"  ** INFEASIBLE ** worst sphere {link} r={radius:.3f} at {np.round(center, 3)}")
    else:
        print("  (feasible)")

    grid, _ = bake(sim.client, None, voxel=args.voxel, primitives=primitives)
    # ClearanceBarrier::interpolationBuffer(field) == field.spacing().maxCoeff()
    buffer = float(grid.spacing.max()) if args.guard_buffer is None else args.guard_buffer
    guarded = args.margin + buffer
    nodes = int(np.prod(grid.dims))
    print(
        f"grid {tuple(int(n) for n in grid.dims)} = {nodes} nodes, "
        f"spacing {np.round(grid.spacing, 4)}, "
        f"range [{grid.values.min():+.3f}, {grid.values.max():+.3f}] m"
    )

    # Interpolation error against the exact field: this is the "SDF
    # discretization" term the barrier margin has to cover, measured rather than
    # assumed. Sample off-node points so it is not trivially zero.
    rng = np.random.default_rng(1)
    probe = np.column_stack(
        [
            rng.uniform(-0.95, 0.95, 4000),
            rng.uniform(-0.95, 0.95, 4000),
            rng.uniform(0.90, 1.90, 4000),
        ]
    )
    exact = scene_distance(primitives, probe)
    interpolated, _ = grid.query(probe)
    error = interpolated - exact
    print(
        f"interpolation error vs exact: max |e| {np.abs(error).max() * 1000:.2f} mm, "
        f"most optimistic {error.max() * 1000:+.2f} mm "
        f"(margin covers {args.margin * 1000:.0f} mm)"
    )
    print(
        f"guarded margin {guarded:.4f} = margin {args.margin:.4f} + buffer {buffer:.4f}; "
        f"goals need h > {buffer:.4f}"
    )

    def headroom(q) -> float:
        """How much a configuration clears the thresholds the *filter* actually guards.

        The two kinds of row are held to different bars and it matters which one a goal
        is short of. `buffer` covers the SDF's interpolation error, a property of the
        grid; between two spheres there is no grid, so the self rows reserve only
        `ClearanceBarrier::defaultSelfBuffer` for step linearisation -- 1 mm against
        15 mm. Judging both by the larger one rejected every goal in `pillars` and
        `clutter` over 2 mm of self clearance the planner never asked for.
        """
        world, own = barrier_parts(q)
        return min(world - buffer, own - self_buffer)

    goals = []
    for index, goal in enumerate(env.goals):
        q, info = robot.ik_search(
            goal.position,
            approach=goal.approach,
            orientation=goal.orientation,
            obstacles=env.obstacles,
            prefer=headroom,
        )
        world, own = barrier_parts(q)
        h = min(world, own)
        room = headroom(q)
        goals.append((index, goal, q, info, room))
        if not info["feasible"]:
            status = "UNUSABLE (no collision-free IK)"
        elif room <= 0.0:
            short = "world" if world - buffer < own - self_buffer else "self"
            status = (f"UNUSABLE ({short} row short by {-room * 1e3:.1f} mm; "
                      f"world needs > {buffer:.3f}, self > {self_buffer:.3f})")
        else:
            status = "ok"
        print(
            f"  goal {index} ({goal.label}): mesh-free={info['collision_free']} "
            f"err={info['position_error'] * 1000:.1f} mm  barrier h={h:+.4f} "
            f"(world {world:+.4f}, self {own:+.4f})  {status}"
        )

    os.makedirs(args.out, exist_ok=True)
    grid_path = os.path.join(args.out, f"{env_name}.grid")
    grid.save(grid_path)
    problem_path = os.path.join(args.out, f"{env_name}.problem")

    lower, upper = REACHABLE_BOUNDS
    usable = [g for g in goals if g[3]["feasible"] and g[4] > 0.0]
    with open(problem_path, "w") as out:
        out.write("# ur5_nav scene export -- consumed by demo_UR5PyBulletScene\n")
        out.write(f"env {env_name}\n")
        out.write(f"seed {args.seed}\n")
        out.write(f"voxel {args.voxel}\n")
        out.write(f"margin {args.margin}\n")
        out.write(f"grid {os.path.basename(grid_path)}\n")
        out.write("bounds " + " ".join(f"{v:.6f}" for v in (*lower, *upper)) + "\n")
        out.write("dims " + " ".join(str(int(n)) for n in grid.dims) + "\n")
        out.write("limits_lower " + " ".join(f"{v:.6f}" for v in robot.lower) + "\n")
        out.write("limits_upper " + " ".join(f"{v:.6f}" for v in robot.upper) + "\n")
        out.write("start " + " ".join(f"{v:.6f}" for v in HOME_CONFIG) + "\n")
        for index, goal, q, _info, h in usable:
            label = (goal.label or f"goal{index}").replace(" ", "_").replace(",", "")
            out.write(
                f"goal {index} {label} "
                + " ".join(f"{v:.6f}" for v in q)
                + f" {h:.6f} "
                + " ".join(f"{v:.6f}" for v in goal.position)
                + "\n"
            )

    print(f"wrote {grid_path} ({os.path.getsize(grid_path) / 1e6:.1f} MB)")
    print(f"wrote {problem_path} ({len(usable)}/{len(goals)} goals usable)")

    if args.slices:
        write_slices(grid, args.out, env_name)

    sim.close()
    return {
        "env": env_name,
        "start_h": start_h,
        "goals": len(goals),
        "usable": len(usable),
        "nodes": nodes,
        "max_interp_error": float(np.abs(error).max()),
    }


def write_slices(grid, out_dir: str, env_name: str) -> None:
    """Save horizontal slices of the field, for eyeballing sign and shape."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (skipping slices: matplotlib not available)")
        return

    heights = [0.95, 1.10, 1.30, 1.50]
    fig, axes = plt.subplots(1, len(heights), figsize=(4 * len(heights), 4))
    for ax, z in zip(np.atleast_1d(axes), heights):
        k = int(round((z - grid.lower[2]) / grid.spacing[2]))
        k = int(np.clip(k, 0, grid.dims[2] - 1))
        field = grid.values[:, :, k].T
        span = float(np.abs(field).max())
        image = ax.imshow(
            field,
            origin="lower",
            cmap="RdBu",
            vmin=-span,
            vmax=span,
            extent=[grid.lower[0], grid.upper[0], grid.lower[1], grid.upper[1]],
        )
        ax.contour(
            np.linspace(grid.lower[0], grid.upper[0], grid.dims[0]),
            np.linspace(grid.lower[1], grid.upper[1], grid.dims[1]),
            field,
            levels=[0.0],
            colors="k",
            linewidths=1.2,
        )
        ax.set_title(f"{env_name}  z = {grid.lower[2] + k * grid.spacing[2]:.3f} m")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{env_name}_slices.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env", default="all", help="environment name, or 'all'")
    parser.add_argument("--out", default="out", help="output directory")
    parser.add_argument(
        "--voxel",
        type=float,
        default=0.015,
        help="grid voxel size, metres. Also sets the guard buffer, so a finer "
        "grid directly buys goal feasibility in tight scenes",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for randomised scenes")
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help="barrier margin used when judging feasibility (ClearanceBarrier::defaultMargin)",
    )
    parser.add_argument(
        "--mount-opening",
        type=float,
        default=0.30,
        help="half-width of the square opening cut in the table under the arm",
    )
    parser.add_argument(
        "--guard-buffer",
        type=float,
        default=None,
        help="override interpolationBuffer (default: the grid's max spacing)",
    )
    parser.add_argument("--slices", action="store_true", help="also save field slice images")
    args = parser.parse_args()

    names = sorted(ENVIRONMENTS) if args.env == "all" else [args.env]
    for name in names:
        if name not in ENVIRONMENTS:
            parser.error(f"unknown env {name!r}; choices: {', '.join(sorted(ENVIRONMENTS))}")

    results = [export(args, name) for name in names]

    print("\n=== export summary ===")
    for r in results:
        print(
            f"  {r['env']:<10} start h {r['start_h']:+.4f}  "
            f"goals {r['usable']}/{r['goals']} usable  "
            f"{r['nodes']} nodes  interp err <= {r['max_interp_error'] * 1000:.2f} mm"
        )
    return 0 if all(r["usable"] > 0 and r["start_h"] > 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
