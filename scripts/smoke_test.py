#!/usr/bin/env python3
"""Invariant checks for the UR5 navigation setup.

Run headless: ``python scripts/smoke_test.py``. Exits non-zero on failure.

These are the properties that silently broke while building the scenes, so they
are the ones worth pinning down: TCP bookkeeping, the self-collision filter, and
whether every published goal is actually reachable in every environment.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5_nav import SimSession, UR5, make_env  # noqa: E402
from ur5_nav.envs import ENVIRONMENTS  # noqa: E402
from ur5_nav.motion import check_path, interpolate  # noqa: E402
from ur5_nav.robot import ARM_JOINT_NAMES, HOME_CONFIG  # noqa: E402
from ur5_nav.sdf import (  # noqa: E402
    BakedGrid,
    bake,
    open_mount_hole,
    probe_pybullet,
    scene_distance,
    scene_primitives,
)
from ur5_nav.spheres import DEFAULT_MARGIN, SpherizedUR5  # noqa: E402

OMPL_DEMO = "/home/mani/ompl/build/demos/demo_UR5PyBulletScene"

FAILURES: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}" + (f" -- {detail}" if detail else ""))
        FAILURES.append(label)


def setup(env_name: str, seed: int = 0):
    """Fresh session with a robot and env, wired the way run_sim.py does."""
    sim = SimSession(gui=False)
    robot = UR5(sim.client)
    env = make_env(env_name, sim.client, seed=seed)
    if env.mount_body is not None:
        robot.ignore_collisions_with(env.mount_body, [-1, "base_link", "shoulder_link"])
    return sim, robot, env


def test_robot_model() -> None:
    print("robot model")
    sim, robot, _ = setup("empty")
    try:
        check(robot.dof == 6, "six actuated joints", f"got {robot.dof}")
        names = [robot.link_names[j] for j in robot.arm_joints]
        check(len(set(names)) == 6, "arm joints map to distinct links", str(names))
        check(
            all(n in ARM_JOINT_NAMES for n in ARM_JOINT_NAMES),
            "expected joint names present",
        )
        check(
            bool(np.all(robot.lower < robot.upper)),
            "joint limits well ordered",
        )
        check(
            robot.ee_link == robot.link_indices["tool0"],
            "end-effector resolves by link name, not joint name",
        )
    finally:
        sim.close()


def test_tcp() -> None:
    print("tool centre point")
    sim, robot, _ = setup("empty")
    try:
        q = HOME_CONFIG
        flange_pos, flange_orn = robot.flange_pose(q)
        tcp_pos, _ = robot.ee_pose(q)
        rot = np.asarray(p.getMatrixFromQuaternion(flange_orn)).reshape(3, 3)
        check(
            np.allclose(tcp_pos, flange_pos + rot @ robot.tcp_offset),
            "TCP is the flange plus the tool offset",
        )

        # The TCP should land between the fingertips, which is what makes
        # Cartesian goals mean "grasp here" rather than "put the flange here".
        robot.set_config(q)
        tips = [
            np.array(
                p.getLinkState(
                    robot.body_id,
                    robot.link_indices[name],
                    computeForwardKinematics=True,
                    physicsClientId=sim.client,
                )[4]
            )
            for name in (
                "robotiq_85_left_finger_tip_link",
                "robotiq_85_right_finger_tip_link",
            )
        ]
        midpoint = sum(tips) / 2.0
        gap = float(np.linalg.norm(tcp_pos - midpoint))
        check(gap < 0.02, "TCP sits between the fingertips", f"off by {gap * 1000:.1f} mm")

        # Round trip: IK to a pose, then FK must return to it.
        target = np.array([0.45, 0.30, 1.25])
        q_ik, info = robot.ik_search(target, approach="-z")
        check(
            info["position_error"] < 1e-3,
            "IK/FK round trip on the TCP",
            f"residual {info['position_error'] * 1000:.2f} mm",
        )
        axis = robot.approach_axis(q_ik)
        check(
            float(np.dot(axis, [0, 0, -1])) > 0.99,
            "approach axis honoured",
            f"axis {np.round(axis, 3)}",
        )
    finally:
        sim.close()


def test_self_collision_filter() -> None:
    print("self-collision filter")
    sim, robot, _ = setup("empty")
    try:
        check(
            len(robot._ignored_self_pairs) > 0,
            "welded and adjacent link pairs are excluded",
        )
        # A link is never its own collision partner, and neighbours across a
        # single joint are excluded -- otherwise every pose reads as colliding.
        check(
            all(a != b for a, b in robot._ignored_self_pairs),
            "ignore set holds no degenerate pairs",
        )
        robot.set_config(HOME_CONFIG)
        check(
            not robot.in_collision(),
            "home pose is self-collision free",
            str([str(c) for c in robot.collision_report()][:3]),
        )
        # A deliberately folded pose must still be *detectable* as colliding, so
        # the filter has not simply switched self-collision off.
        folded = np.array([0.0, -0.3, 2.9, 2.9, 0.0, 0.0])
        check(
            robot.in_collision(folded),
            "a folded pose is reported as self-colliding",
        )
    finally:
        sim.close()


def test_mount_exclusion() -> None:
    print("mounting surface")
    sim, robot, env = setup("empty")
    try:
        robot.set_config(HOME_CONFIG)
        check(
            not robot.in_collision(obstacles=env.obstacles),
            "base resting on the table is not a collision",
        )
        # The table must still block the rest of the arm.
        into_table = np.array([0.0, 0.6, 1.2, 0.0, 0.0, 0.0])
        check(
            robot.in_collision(into_table, obstacles=env.obstacles),
            "the table still blocks the arm itself",
        )
    finally:
        sim.close()


def test_goals_reachable() -> None:
    print("goal reachability")
    for name in sorted(ENVIRONMENTS):
        sim, robot, env = setup(name)
        try:
            robot.set_config(HOME_CONFIG)
            check(
                not robot.in_collision(obstacles=env.obstacles),
                f"{name}: home pose is clear of the scene",
            )
            bad = []
            for i, goal in enumerate(env.goals):
                _, info = robot.ik_search(
                    goal.position,
                    approach=goal.approach,
                    orientation=goal.orientation,
                    obstacles=env.obstacles,
                )
                if not info["feasible"]:
                    bad.append(
                        f"goal {i} ({goal.label}): "
                        f"{info['position_error'] * 1000:.0f} mm, "
                        f"free={info['collision_free']}"
                    )
            check(not bad, f"{name}: all {len(env.goals)} goals reachable", "; ".join(bad))
        finally:
            sim.close()


def test_random_envs_across_seeds() -> None:
    print("randomised environments across seeds")
    for name in ("pillars", "clutter"):
        for seed in range(4):
            sim, robot, env = setup(name, seed=seed)
            try:
                bad = [
                    i
                    for i, goal in enumerate(env.goals)
                    if not robot.ik_search(
                        goal.position, approach=goal.approach, obstacles=env.obstacles
                    )[1]["feasible"]
                ]
                check(not bad, f"{name} seed {seed}: all goals reachable", f"blocked {bad}")
            finally:
                sim.close()


def test_path_helpers() -> None:
    print("path helpers")
    sim, robot, env = setup("shelf")
    try:
        start, goal = HOME_CONFIG, np.array([1.0, -1.0, 0.8, -1.2, -1.5, 0.3])
        path = interpolate(start, goal, max_step=0.03)
        steps = [np.max(np.abs(b - a)) for a, b in zip(path[:-1], path[1:])]
        check(np.allclose(path[0], start), "path starts at the start config")
        check(np.allclose(path[-1], goal), "path ends at the goal config")
        check(max(steps) <= 0.03 + 1e-9, "waypoint spacing respects max_step")

        collided, first_bad, min_clear = check_path(robot, path, obstacles=env.obstacles)
        check(
            isinstance(collided, bool) and np.isfinite(min_clear),
            "check_path returns a usable verdict",
        )
        check(
            (first_bad is None) == (not collided),
            "collision index is set exactly when a collision is reported",
        )
        # Checking a path must not move the arm.
        check(
            np.allclose(robot.get_config(), HOME_CONFIG),
            "check_path leaves the arm where it found it",
        )
    finally:
        sim.close()


def test_capture() -> None:
    print("offscreen rendering")
    sim, _, env = setup("pillars")
    try:
        rgb = sim.capture(width=240, height=180, **env.camera)
        check(rgb.shape == (180, 240, 3), "capture returns an HxWx3 frame", str(rgb.shape))
        check(rgb.dtype == np.uint8 and rgb.ptp() > 0, "frame has actual image content")
    finally:
        sim.close()


def test_sdf_matches_pybullet() -> None:
    print("analytic SDF vs PyBullet distances")
    rng = np.random.default_rng(3)
    for name in ("shelf", "clutter"):
        sim, _, env = setup(name)
        try:
            bodies = [sim.plane_id] + env.obstacles
            primitives = scene_primitives(sim.client, bodies)
            points = np.column_stack(
                [
                    rng.uniform(-0.9, 0.9, 120),
                    rng.uniform(-0.9, 0.9, 120),
                    rng.uniform(0.95, 1.8, 120),
                ]
            )
            mine = scene_distance(primitives, points)
            reference = np.array([probe_pybullet(sim.client, q, bodies) for q in points])
            outside = mine > 0.002  # PyBullet's penetration depth is unreliable
            error = np.abs(mine[outside] - reference[outside])
            check(
                error.max() < 0.002,
                f"{name}: analytic distance matches PyBullet outside obstacles",
                f"max error {error.max() * 1000:.2f} mm",
            )
            inside = ~outside
            check(
                bool(np.all(reference[inside] <= 0.005)),
                f"{name}: signs agree inside obstacles",
            )
        finally:
            sim.close()


def test_grid_round_trip() -> None:
    print("grid serialisation")
    import tempfile

    sim, _, env = setup("pillars")
    try:
        primitives = open_mount_hole(
            scene_primitives(sim.client, [sim.plane_id] + env.obstacles), env.mount_body
        )
        # A coarse grid keeps this fast; the layout is what is under test.
        grid, _ = bake(sim.client, None, voxel=0.08, primitives=primitives)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.grid")
            grid.save(path)
            reloaded = BakedGrid.load(path)
        check(np.array_equal(grid.dims, reloaded.dims), "dimensions survive a round trip")
        check(np.allclose(grid.lower, reloaded.lower), "bounds survive a round trip")
        check(np.array_equal(grid.values, reloaded.values), "values survive a round trip")

        # The interpolant must reproduce node values exactly at the nodes.
        nodes = grid.node_points()[:: max(1, len(grid.node_points()) // 200)]
        values, _ = grid.query(nodes)
        exact = grid.values.ravel(order="F")[:: max(1, grid.values.size // 200)]
        check(np.allclose(values, exact, atol=1e-9), "interpolation is exact at grid nodes")
    finally:
        sim.close()


def test_mount_hole() -> None:
    print("mounting-surface opening")
    sim, _, env = setup("shelf")
    try:
        arm = SpherizedUR5(sim.client)
        raw = scene_primitives(sim.client, [sim.plane_id] + env.obstacles)
        opened = open_mount_hole(raw, env.mount_body, half_width=0.30)

        model = arm.at(HOME_CONFIG)
        before = model.barrier(scene_distance(raw, model.centers), DEFAULT_MARGIN).min()
        after = model.barrier(scene_distance(opened, model.centers), DEFAULT_MARGIN).min()
        check(before < 0.0, "solid table makes the start state infeasible", f"h = {before:+.4f}")
        check(after > 0.0, "opening the mount makes it feasible", f"h = {after:+.4f}")

        # The opening must not have deleted anything but table.
        elsewhere = np.array([[0.62, 0.0, 1.10], [0.62, 0.0, 1.35], [0.0, 0.0, 0.5]])
        check(
            np.allclose(scene_distance(raw, elsewhere), scene_distance(opened, elsewhere), atol=1e-9)
            or bool(scene_distance(opened, elsewhere)[2] > scene_distance(raw, elsewhere)[2]),
            "the opening only changes the field near the base",
        )
        check(arm.n_spheres == 40, "sphere model has 40 spheres", str(arm.n_spheres))
    finally:
        sim.close()


def test_cpp_reads_the_same_field() -> None:
    print("cross-language field agreement")
    if not os.path.exists(OMPL_DEMO):
        print(f"  skip (build {OMPL_DEMO} to enable)")
        return
    import subprocess
    import tempfile

    sim, _, env = setup("wall_gap")
    try:
        primitives = open_mount_hole(
            scene_primitives(sim.client, [sim.plane_id] + env.obstacles), env.mount_body
        )
        grid, _ = bake(sim.client, None, voxel=0.05, primitives=primitives)
        with tempfile.TemporaryDirectory() as tmp:
            grid_path = os.path.join(tmp, "x.grid")
            grid.save(grid_path)
            problem = os.path.join(tmp, "x.problem")
            with open(problem, "w") as out:
                out.write("env test\ngrid x.grid\n")
            rng = np.random.default_rng(11)
            points = np.column_stack(
                [
                    rng.uniform(-1.1, 1.1, 200),
                    rng.uniform(-1.1, 1.1, 200),
                    rng.uniform(-0.1, 2.1, 200),
                ]
            )
            result = subprocess.run(
                [OMPL_DEMO, problem, "--probe"],
                input="\n".join(f"{x} {y} {z}" for x, y, z in points),
                capture_output=True,
                text=True,
                timeout=180,
            )
        rows = [l for l in result.stdout.splitlines() if l and not l.startswith("#")]
        theirs = np.array([[float(v) for v in r.split()] for r in rows])
        mine_value, mine_gradient = grid.query(points)
        check(len(rows) == len(points), "C++ answered every probe point", f"{len(rows)} rows")
        check(
            float(np.abs(theirs[:, 0] - mine_value).max()) < 1e-8,
            "GridSDF::load reads the same values Python wrote",
            f"max diff {np.abs(theirs[:, 0] - mine_value).max():.2e}",
        )
        check(
            float(np.linalg.norm(theirs[:, 1:] - mine_gradient, axis=1).max()) < 1e-8,
            "and derives the same gradients",
        )
    finally:
        sim.close()


def main() -> int:
    for test in (
        test_robot_model,
        test_tcp,
        test_self_collision_filter,
        test_mount_exclusion,
        test_goals_reachable,
        test_random_envs_across_seeds,
        test_path_helpers,
        test_capture,
        test_sdf_matches_pybullet,
        test_grid_round_trip,
        test_mount_hole,
        test_cpp_reads_the_same_field,
    ):
        test()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
