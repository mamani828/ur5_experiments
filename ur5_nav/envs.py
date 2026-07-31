"""Obstacle environments for UR5 navigation experiments.

Each builder returns an :class:`Env` holding the obstacle body ids plus a start
configuration and a list of Cartesian goals for the end-effector. Goals are
chosen so that reaching them requires routing the arm *around* geometry rather
than moving in a straight line.

Distances are metres in world frame. The robot's URDF lifts its own base to
z = 0.9144, so the table top sits at that height and every workspace goal is
specified in absolute world coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pybullet as p

from .robot import APPROACH_AXES

TABLE_HEIGHT = 0.9144


@dataclass
class Goal:
    """A Cartesian target for the TCP (the point between the fingers).

    ``approach`` names the world axis the tool must point along (``"-z"`` for a
    top-down grasp, ``"+x"`` to reach into a shelf bay). The roll about that
    axis is left free for :meth:`UR5.ik_search` to choose. ``orientation`` pins
    the full wrist pose instead, when that matters.
    """

    position: np.ndarray
    approach: str | None = None
    orientation: tuple | None = None
    label: str = ""

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)


@dataclass
class Env:
    """A populated scene."""

    name: str
    client: int
    obstacles: list[int] = field(default_factory=list)
    goals: list[Goal] = field(default_factory=list)
    start_config: np.ndarray | None = None
    description: str = ""
    # The surface the arm is bolted to. Still an obstacle for the rest of the
    # arm, but its permanent contact with the base links must be excluded.
    mount_body: int | None = None
    # Viewpoint that shows this scene's interesting geometry. PyBullet's yaw
    # convention here: 0 places the camera on the -y side, 90 on +x, 270 on -x.
    camera: dict = field(
        default_factory=lambda: dict(distance=2.4, yaw=315, pitch=-25, target=(0.0, 0.1, 1.15))
    )

    def __len__(self) -> int:
        return len(self.obstacles)


# -- primitives ------------------------------------------------------------


def add_box(
    client: int,
    half_extents,
    position,
    orientation=(0, 0, 0, 1),
    rgba=(0.55, 0.35, 0.2, 1.0),
    mass: float = 0.0,
) -> int:
    """Add a box. ``mass=0`` makes it static (the default for obstacles)."""
    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=list(half_extents), physicsClientId=client
    )
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=list(half_extents), rgbaColor=list(rgba), physicsClientId=client
    )
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=list(position),
        baseOrientation=list(orientation),
        physicsClientId=client,
    )


def add_cylinder(
    client: int,
    radius: float,
    height: float,
    position,
    rgba=(0.3, 0.4, 0.7, 1.0),
    mass: float = 0.0,
) -> int:
    col = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=client
    )
    vis = p.createVisualShape(
        p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=list(rgba), physicsClientId=client
    )
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=list(position),
        physicsClientId=client,
    )


def add_sphere(
    client: int, radius: float, position, rgba=(0.8, 0.3, 0.3, 1.0), mass: float = 0.0
) -> int:
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=client)
    vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=radius, rgbaColor=list(rgba), physicsClientId=client
    )
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=list(position),
        physicsClientId=client,
    )


def add_table(client: int, size=(0.8, 1.2), height: float = TABLE_HEIGHT) -> int:
    """The surface the arm is bolted to. Counts as an obstacle."""
    return add_box(
        client,
        half_extents=(size[0], size[1], height / 2.0),
        position=(0.0, 0.0, height / 2.0),
        rgba=(0.45, 0.42, 0.38, 1.0),
    )


# -- environments ----------------------------------------------------------


def env_empty(client: int) -> Env:
    """Table only. Baseline for checking kinematics and reachability."""
    obstacles = [add_table(client)]
    goals = [
        Goal([0.45, 0.35, 1.25], "-z", label="front-right, top-down"),
        Goal([-0.45, 0.35, 1.25], "-z", label="front-left, top-down"),
        Goal([0.0, 0.50, 1.40], "+y", label="high reach, forward"),
        Goal([0.35, -0.40, 1.15], "-z", label="behind, top-down"),
    ]
    return Env(
        name="empty",
        client=client,
        camera=dict(distance=2.3, yaw=315, pitch=-25, target=(0.0, 0.05, 1.15)),
        obstacles=obstacles,
        goals=goals,
        mount_body=obstacles[0],  # the table, always added first
        description="Bare table, no obstacles.",
    )


def env_shelf(client: int) -> Env:
    """A two-tier shelf in front of the arm; goals sit inside the bays.

    Reaching a bay means threading the wrist through a horizontal slot, so the
    arm has to approach from the front rather than dropping in from above.
    """
    obstacles = [add_table(client)]
    shelf_x, depth, width = 0.62, 0.14, 0.75
    # Bay pitch and depth are set by the *planner's* footprint, not the
    # gripper's. A spherized wrist needs r + margin + one voxel = 0.115 m of
    # clear space around its centres, and the surface that binds is the front
    # edge of the shelf board above it -- so shallower boards buy as much as
    # taller bays. At the 0.30 m pitch and 0.16 m depth that mesh checking is
    # perfectly happy with, a CBF has no feasible pose inside a bay at all.
    # See README_OMPL_CBF.md.
    bay_pitch, panel_half = 0.44, 0.46
    wood = (0.62, 0.44, 0.26, 1.0)

    # Back panel and two side panels.
    obstacles.append(
        add_box(
            client,
            (0.02, width / 2, panel_half),
            (shelf_x + depth, 0.0, TABLE_HEIGHT + panel_half),
            rgba=wood,
        )
    )
    for side in (-1, 1):
        obstacles.append(
            add_box(
                client,
                (depth, 0.02, panel_half),
                (shelf_x, side * width / 2, TABLE_HEIGHT + panel_half),
                rgba=wood,
            )
        )
    # Bottom, middle and top shelves -> two open bays.
    for z in (0.0, bay_pitch, 2 * bay_pitch):
        obstacles.append(
            add_box(client, (depth, width / 2, 0.015), (shelf_x, 0.0, TABLE_HEIGHT + z), rgba=wood)
        )

    # Reach in along +x; the search picks a roll whose finger profile fits.
    goals = [
        Goal([shelf_x, -0.20, TABLE_HEIGHT + bay_pitch / 2], "+x", label="lower bay, left"),
        Goal([shelf_x, 0.20, TABLE_HEIGHT + bay_pitch / 2], "+x", label="lower bay, right"),
        Goal([shelf_x, -0.10, TABLE_HEIGHT + 1.5 * bay_pitch], "+x", label="upper bay, left"),
        Goal([shelf_x, 0.18, TABLE_HEIGHT + 1.5 * bay_pitch], "+x", label="upper bay, right"),
    ]
    return Env(
        name="shelf",
        client=client,
        camera=dict(distance=2.1, yaw=300, pitch=-18, target=(0.30, 0.00, 1.25)),
        obstacles=obstacles,
        goals=goals,
        mount_body=obstacles[0],  # the table, always added first
        description="Two-bay shelf; goals require entering horizontal slots.",
    )


def env_wall_gap(client: int) -> Env:
    """A wall bisecting the workspace with a single window to pass through."""
    obstacles = [add_table(client)]
    wall_y, thickness = 0.34, 0.03
    # The window admits the wrist *and* the forearm behind it, so it is sized
    # off the gripper's 0.16 m open width with clearance to spare. Its height
    # matters as much as its size: set low (~0.35 m above the table) the elbow
    # cannot clear the slab underneath and no goal beyond the wall is reachable.
    gap_center_z, gap_h, gap_w = TABLE_HEIGHT + 0.50, 0.46, 0.50
    wall_half_w, wall_top = 0.70, TABLE_HEIGHT + 0.85
    grey = (0.7, 0.7, 0.72, 1.0)

    # Wall built as four slabs around the window opening.
    below_h = (gap_center_z - gap_h / 2) - TABLE_HEIGHT
    obstacles.append(
        add_box(
            client,
            (wall_half_w, thickness, below_h / 2),
            (0.0, wall_y, TABLE_HEIGHT + below_h / 2),
            rgba=grey,
        )
    )
    above_h = wall_top - (gap_center_z + gap_h / 2)
    obstacles.append(
        add_box(
            client,
            (wall_half_w, thickness, above_h / 2),
            (0.0, wall_y, gap_center_z + gap_h / 2 + above_h / 2),
            rgba=grey,
        )
    )
    for side in (-1, 1):
        side_w = (wall_half_w - gap_w / 2) / 2
        obstacles.append(
            add_box(
                client,
                (side_w, thickness, gap_h / 2),
                (side * (gap_w / 2 + side_w), wall_y, gap_center_z),
                rgba=grey,
            )
        )

    # The wrist has to pass through the window, so approach along +y.
    goals = [
        Goal([0.0, wall_y + 0.24, gap_center_z], "+y", label="straight through the window"),
        Goal([0.07, wall_y + 0.20, gap_center_z + 0.05], "+y", label="beyond, offset right"),
        Goal([-0.07, wall_y + 0.20, gap_center_z - 0.05], "+y", label="beyond, offset left"),
    ]
    return Env(
        name="wall_gap",
        client=client,
        camera=dict(distance=2.2, yaw=340, pitch=-14, target=(0.0, 0.35, 1.30)),
        obstacles=obstacles,
        goals=goals,
        mount_body=obstacles[0],  # the table, always added first
        description="Dividing wall with one narrow window; goals lie beyond it.",
    )


def env_pillars(client: int, seed: int = 0) -> Env:
    """A forest of vertical posts the arm must weave between.

    Posts are kept clear of the goals and of the column above each one, so the
    goals stay reachable for every seed -- otherwise a randomly placed post can
    simply sit on a goal and the scenario becomes unsolvable rather than hard.
    """
    rng = np.random.default_rng(seed)
    obstacles = [add_table(client)]

    goals = [
        Goal([0.46, 0.28, TABLE_HEIGHT + 0.15], "-z", label="between posts, right"),
        Goal([-0.40, 0.42, TABLE_HEIGHT + 0.18], "-z", label="between posts, left"),
        Goal([0.26, 0.50, TABLE_HEIGHT + 0.14], "-z", label="far side, low"),
        Goal([0.58, -0.18, TABLE_HEIGHT + 0.22], "-z", label="behind the field"),
    ]
    goal_xy = [g.position[:2] for g in goals]

    placed: list[np.ndarray] = []
    attempts = 0
    while len(placed) < 7 and attempts < 400:
        attempts += 1
        angle = rng.uniform(-np.pi / 2, np.pi / 2)
        radius = rng.uniform(0.34, 0.62)
        xy = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        if any(np.linalg.norm(xy - other) < 0.20 for other in placed):
            continue
        # Every goal here is a top-down approach, so keep the column clear.
        # 0.24 m, not just the gripper width: the forearm has to come down
        # into the goal too, and it is what a nearby post actually blocks.
        if any(np.linalg.norm(xy - g) < 0.28 for g in goal_xy):
            continue
        placed.append(xy)
        # Capped at 0.50 m: taller posts stand above the whole reachable
        # workspace and block the extended forearm outright rather than
        # forcing it to weave.
        height = rng.uniform(0.30, 0.50)
        obstacles.append(
            add_cylinder(
                client,
                radius=rng.uniform(0.035, 0.055),
                height=height,
                position=(xy[0], xy[1], TABLE_HEIGHT + height / 2),
                rgba=(0.30, 0.45, 0.72, 1.0),
            )
        )
    return Env(
        name="pillars",
        client=client,
        camera=dict(distance=2.0, yaw=315, pitch=-30, target=(0.10, 0.20, 1.05)),
        obstacles=obstacles,
        goals=goals,
        mount_body=obstacles[0],  # the table, always added first
        description="Randomised post field; low goals between the posts.",
    )


def env_clutter(client: int, seed: int = 0) -> Env:
    """Randomised mixed-primitive clutter, including overhead obstacles."""
    rng = np.random.default_rng(seed)
    obstacles = [add_table(client)]

    goals = [
        Goal([0.50, 0.32, TABLE_HEIGHT + 0.15], "-z", label="low right"),
        Goal([-0.42, 0.38, TABLE_HEIGHT + 0.30], "-z", label="mid left"),
        Goal([0.20, 0.55, TABLE_HEIGHT + 0.42], "+y", label="high forward"),
        Goal([0.55, -0.25, TABLE_HEIGHT + 0.20], "-z", label="behind"),
    ]

    def blocks_a_goal(center: np.ndarray) -> bool:
        """Reject clutter sitting on a goal, or on its approach corridor."""
        for goal in goals:
            if np.linalg.norm(center - goal.position) < 0.24:
                return True
            axis = APPROACH_AXES[goal.approach]
            # Distance from the obstacle to the 0.32 m approach segment leading
            # into the goal, measured along the tool's approach direction. The
            # radius covers the forearm, not just the gripper.
            along = float(np.dot(center - goal.position, -axis))
            if 0.0 <= along <= 0.32:
                lateral = np.linalg.norm((center - goal.position) + along * axis)
                if lateral < 0.20:
                    return True
        return False

    placed: list[np.ndarray] = []
    attempts = 0
    while len(placed) < 10 and attempts < 600:
        attempts += 1
        angle = rng.uniform(-2.6, 2.6)
        radius = rng.uniform(0.32, 0.65)
        z = TABLE_HEIGHT + rng.uniform(0.05, 0.70)
        center = np.array([radius * np.cos(angle), radius * np.sin(angle), z])
        if any(np.linalg.norm(center - other) < 0.19 for other in placed):
            continue
        if blocks_a_goal(center):
            continue
        placed.append(center)
        kind = rng.integers(0, 3)
        if kind == 0:
            half = rng.uniform(0.035, 0.075, size=3)
            yaw = rng.uniform(0, np.pi)
            obstacles.append(
                add_box(
                    client,
                    half,
                    center,
                    orientation=p.getQuaternionFromEuler([0, 0, yaw]),
                    rgba=(0.75, 0.55, 0.25, 1.0),
                )
            )
        elif kind == 1:
            obstacles.append(
                add_sphere(client, rng.uniform(0.04, 0.08), center, rgba=(0.78, 0.32, 0.32, 1.0))
            )
        else:
            h = rng.uniform(0.10, 0.30)
            obstacles.append(
                add_cylinder(
                    client, rng.uniform(0.03, 0.06), h, center, rgba=(0.35, 0.62, 0.45, 1.0)
                )
            )

    return Env(
        name="clutter",
        client=client,
        camera=dict(distance=2.2, yaw=315, pitch=-26, target=(0.10, 0.20, 1.15)),
        obstacles=obstacles,
        goals=goals,
        mount_body=obstacles[0],  # the table, always added first
        description="Random boxes, spheres and cylinders at mixed heights.",
    )


def env_corridor(client: int) -> Env:
    """Two parallel walls forming a corridor with a partial ceiling.

    The corridor starts well clear of the base -- run it flush to the origin and
    the walls enclose the arm itself, which makes every goal unreachable rather
    than merely hard.
    """
    obstacles = [add_table(client)]
    grey = (0.66, 0.68, 0.72, 1.0)
    half_gap, height = 0.28, 0.52
    y_near, y_far = 0.26, 0.88
    y_mid, half_len = (y_near + y_far) / 2, (y_far - y_near) / 2

    for side in (-1, 1):
        obstacles.append(
            add_box(
                client,
                (0.025, half_len, height / 2),
                (side * half_gap, y_mid, TABLE_HEIGHT + height / 2),
                rgba=grey,
            )
        )
    # Ceiling over the far third only, so the near end can be entered from above
    # while the far goals demand a horizontal traverse underneath.
    roof_near = 0.62
    obstacles.append(
        add_box(
            client,
            (half_gap, (y_far - roof_near) / 2, 0.02),
            (0.0, (y_far + roof_near) / 2, TABLE_HEIGHT + height),
            rgba=(0.6, 0.6, 0.65, 1.0),
        )
    )
    goals = [
        Goal([0.0, 0.40, TABLE_HEIGHT + 0.16], "-z", label="corridor entrance, from above"),
        Goal([0.0, 0.72, TABLE_HEIGHT + 0.16], "+y", label="under the ceiling"),
        Goal([0.0, 0.84, TABLE_HEIGHT + 0.22], "+y", label="far end, under the ceiling"),
    ]
    return Env(
        name="corridor",
        client=client,
        camera=dict(distance=2.0, yaw=330, pitch=-32, target=(0.0, 0.45, 1.05)),
        obstacles=obstacles,
        goals=goals,
        mount_body=obstacles[0],  # the table, always added first
        description="Narrow corridor with a partial ceiling; goals at the far end.",
    )


ENVIRONMENTS: dict[str, Callable[..., Env]] = {
    "empty": env_empty,
    "shelf": env_shelf,
    "wall_gap": env_wall_gap,
    "corridor": env_corridor,
    "pillars": env_pillars,
    "clutter": env_clutter,
}


def make_env(name: str, client: int, **kwargs) -> Env:
    """Build environment ``name`` in ``client``.

    Randomised environments (``pillars``, ``clutter``) accept ``seed``.
    """
    if name not in ENVIRONMENTS:
        raise KeyError(f"unknown env {name!r}; available: {sorted(ENVIRONMENTS)}")
    builder = ENVIRONMENTS[name]
    if name in ("pillars", "clutter"):
        return builder(client, **kwargs)
    kwargs.pop("seed", None)
    return builder(client, **kwargs)
