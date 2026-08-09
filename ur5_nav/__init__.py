"""PyBullet UR5 navigation experiments.

Layout:
    sim.py     -- simulator session (connect, ground plane, camera, capture)
    robot.py   -- UR5 wrapper: joint access, FK/IK, collision queries
    envs.py    -- obstacle environments with goal specs
    motion.py  -- joint-space execution and goal-reaching helpers
    paths.py   -- reading the multi-motion path files the OMPL demos write
"""

from .sim import SimSession
from .robot import UR5, UR5_URDF, HOME_CONFIG, APPROACH_AXES, tool_orientation
from .envs import ENVIRONMENTS, Env, Goal, make_env
from .motion import check_path, execute_path, interpolate, move_to_config, move_to_goal
from .paths import load_runs, split_runs

__all__ = [
    "SimSession",
    "UR5",
    "UR5_URDF",
    "HOME_CONFIG",
    "APPROACH_AXES",
    "tool_orientation",
    "ENVIRONMENTS",
    "Env",
    "Goal",
    "make_env",
    "check_path",
    "execute_path",
    "interpolate",
    "move_to_config",
    "move_to_goal",
    "load_runs",
    "split_runs",
]
