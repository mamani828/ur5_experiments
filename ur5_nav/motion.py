"""Path execution utilities.

Deliberately planner-free: these helpers move the arm along a *given* sequence
of configurations and report what happens (clearance, collisions, tracking
error). Dropping a real planner in later means producing the waypoint list --
nothing here needs to change.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .robot import UR5
from .sim import SimSession


def interpolate(q_start, q_goal, max_step: float = 0.03) -> list[np.ndarray]:
    """Densely sample the straight line between two configurations.

    ``max_step`` is the largest per-joint change between consecutive waypoints,
    in radians -- small enough that collision checks cannot tunnel through thin
    obstacles.
    """
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    span = float(np.max(np.abs(q_goal - q_start)))
    steps = max(2, int(np.ceil(span / max_step)) + 1)
    return [q_start + (q_goal - q_start) * t for t in np.linspace(0.0, 1.0, steps)]


@dataclass
class ExecutionResult:
    """Outcome of running one path."""

    waypoints: int
    collided: bool
    first_collision_index: int | None
    min_clearance: float
    final_position_error: float | None
    ee_trace: list[np.ndarray] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "COLLISION" if self.collided else "clear"
        parts = [
            f"{self.waypoints} waypoints",
            status,
            f"min clearance {self.min_clearance:+.3f} m",
        ]
        if self.first_collision_index is not None:
            parts.append(f"first hit at waypoint {self.first_collision_index}")
        if self.final_position_error is not None:
            parts.append(f"goal error {self.final_position_error * 1000:.1f} mm")
        return " | ".join(parts)


def check_path(
    robot: UR5, path, obstacles=None, margin: float = 0.0
) -> tuple[bool, int | None, float]:
    """Collision-check a path without animating it.

    Returns ``(collided, first_bad_index, min_clearance)``.
    """
    saved = robot.get_config()
    collided = False
    first_bad = None
    min_clear = float("inf")
    try:
        for i, q in enumerate(path):
            clear = robot.clearance(q, obstacles=obstacles)
            min_clear = min(min_clear, clear)
            if clear <= margin:
                collided = True
                if first_bad is None:
                    first_bad = i
    finally:
        robot.set_config(saved)
    return collided, first_bad, min_clear if np.isfinite(min_clear) else 0.0


def execute_path(
    sim: SimSession,
    robot: UR5,
    path,
    obstacles=None,
    dynamic: bool = False,
    realtime: bool = True,
    steps_per_waypoint: int = 8,
    goal_position=None,
    trace: bool = True,
    stop_on_collision: bool = False,
) -> ExecutionResult:
    """Animate ``path`` and record clearance along the way.

    ``dynamic=False`` teleports the arm through each waypoint (pure kinematic
    replay -- what a planner's output looks like). ``dynamic=True`` drives the
    joint position controllers and steps physics, so the arm lags its target
    and can actually push obstacles around.
    """
    path = [np.asarray(q, dtype=float) for q in path]
    result = ExecutionResult(
        waypoints=len(path),
        collided=False,
        first_collision_index=None,
        min_clearance=float("inf"),
        final_position_error=None,
    )

    for i, q in enumerate(path):
        if dynamic:
            robot.control_to(q)
            sim.step(steps_per_waypoint)
        else:
            robot.set_config(q)
            sim.step(0)

        clearance = robot.clearance(obstacles=obstacles)
        result.min_clearance = min(result.min_clearance, clearance)
        if clearance <= 0.0:
            if not result.collided:
                result.first_collision_index = i
                hits = robot.collision_report(obstacles=obstacles)
                for hit in hits[:3]:
                    result.messages.append(f"waypoint {i}: {hit}")
            result.collided = True
            if stop_on_collision:
                break

        if trace:
            result.ee_trace.append(robot.ee_pose()[0])
        if realtime and sim.gui:
            time.sleep(sim.timestep * steps_per_waypoint if dynamic else 0.01)

    if not np.isfinite(result.min_clearance):
        result.min_clearance = 0.0
    if goal_position is not None:
        result.final_position_error = robot.ee_position_error(
            robot.get_config(), goal_position
        )
    if trace and len(result.ee_trace) > 1:
        sim.draw_trace(result.ee_trace, rgb=(1.0, 0.45, 0.0) if result.collided else (0.1, 0.9, 0.4))
    return result


def move_to_config(
    sim: SimSession,
    robot: UR5,
    q_goal,
    obstacles=None,
    max_step: float = 0.03,
    **kwargs,
) -> ExecutionResult:
    """Straight-line joint-space move from the current configuration."""
    path = interpolate(robot.get_config(), robot.clamp(q_goal), max_step=max_step)
    return execute_path(sim, robot, path, obstacles=obstacles, **kwargs)


def move_to_goal(
    sim: SimSession,
    robot: UR5,
    goal,
    obstacles=None,
    **kwargs,
) -> tuple[ExecutionResult, np.ndarray, dict]:
    """Solve a :class:`~ur5_nav.envs.Goal`, then move there in a straight line.

    Returns ``(result, q_goal, ik_info)``. A collision in the result is expected
    in the cluttered environments -- a straight line in joint space ignores
    obstacles entirely, and closing that gap is exactly what a planner is for.
    """
    q_goal, info = robot.ik_search(
        goal.position,
        approach=goal.approach,
        orientation=goal.orientation,
        obstacles=obstacles,
    )
    result = move_to_config(
        sim, robot, q_goal, obstacles=obstacles, goal_position=goal.position, **kwargs
    )
    return result, q_goal, info
