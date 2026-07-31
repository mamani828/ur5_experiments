"""UR5 wrapper: joint access, kinematics and collision queries."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pybullet as p

# Self-contained UR5 + Robotiq-85 description (meshes live alongside it).
UR5_URDF = "/home/mani/vamp/resources/ur5/ur5.urdf"

# The six actuated joints, in kinematic order.
ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# Arm folded straight up, so it occupies only a thin column above its own base.
# Every environment here is built to leave that column clear, which makes it a
# valid start state everywhere -- an elbow-out pose collides with the shelf,
# the wall and the post field.
HOME_CONFIG = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, -np.pi / 2, 0.0])

# World directions the tool's local +z can be aligned with.
APPROACH_AXES = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}


def tool_orientation(axis: str, roll: float = 0.0):
    """Quaternion aligning the tool's local +z with a world axis.

    ``roll`` spins the gripper about its own approach axis, which decides
    whether the fingers open vertically or horizontally.
    """
    base = {
        "+x": [0.0, np.pi / 2, 0.0],
        "-x": [0.0, -np.pi / 2, 0.0],
        "+y": [-np.pi / 2, 0.0, 0.0],
        "-y": [np.pi / 2, 0.0, 0.0],
        "+z": [0.0, 0.0, 0.0],
        "-z": [np.pi, 0.0, 0.0],
    }
    if axis not in base:
        raise KeyError(f"axis must be one of {sorted(base)}, got {axis!r}")
    q = p.getQuaternionFromEuler(base[axis])
    if roll:
        # Post-multiply so the roll is about the tool axis, not a world axis.
        q = p.multiplyTransforms(
            [0, 0, 0], q, [0, 0, 0], p.getQuaternionFromEuler([0, 0, roll])
        )[1]
    return q


@dataclass
class Contact:
    """One colliding or near-colliding link pair."""

    link: str
    other_body: int
    other_link: int
    distance: float

    def __str__(self) -> str:
        return f"{self.link} <-> body {self.other_body} link {self.other_link} @ {self.distance:+.4f} m"


class UR5:
    """A UR5 loaded into a PyBullet client, with collision-aware helpers.

    The URDF's fixed ``offset_joint`` already raises ``base_link`` by 0.9144 m,
    so loading at the world origin places the arm on top of a 0.9144 m table.
    """

    def __init__(
        self,
        client: int,
        urdf: str = UR5_URDF,
        base_position=(0.0, 0.0, 0.0),
        base_orientation=(0.0, 0.0, 0.0, 1.0),
        ee_link_name: str = "tool0",
        tcp_offset=(0.0, 0.0, 0.15),
        self_collision: bool = True,
    ) -> None:
        if not os.path.exists(urdf):
            raise FileNotFoundError(f"UR5 URDF not found: {urdf}")

        self.client = client
        # package:// mesh references resolve against the additional search path.
        p.setAdditionalSearchPath(os.path.dirname(os.path.abspath(urdf)), physicsClientId=client)

        flags = p.URDF_USE_INERTIA_FROM_FILE
        if self_collision:
            flags |= p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT

        self.body_id = p.loadURDF(
            urdf,
            basePosition=list(base_position),
            baseOrientation=list(base_orientation),
            useFixedBase=True,
            flags=flags,
            physicsClientId=client,
        )

        root_link = p.getBodyInfo(self.body_id, physicsClientId=client)[0].decode()
        self.link_names: dict[int, str] = {-1: root_link}
        name_to_joint: dict[str, int] = {}
        self.link_indices: dict[str, int] = {}
        for j in range(p.getNumJoints(self.body_id, physicsClientId=client)):
            info = p.getJointInfo(self.body_id, j, physicsClientId=client)
            name_to_joint[info[1].decode()] = j
            # A link is indexed by the joint that attaches it to its parent.
            link = info[12].decode()
            self.link_names[j] = link
            self.link_indices[link] = j
        self.link_indices.setdefault(root_link, -1)

        self.arm_joints = [name_to_joint[n] for n in ARM_JOINT_NAMES]
        self.all_links = list(self.link_names.keys())

        limits = [
            p.getJointInfo(self.body_id, j, physicsClientId=client) for j in self.arm_joints
        ]
        self.lower = np.array([i[8] for i in limits])
        self.upper = np.array([i[9] for i in limits])
        self.max_velocity = np.array([i[11] if i[11] > 0 else 1.0 for i in limits])
        self.max_force = np.array([i[10] if i[10] > 0 else 150.0 for i in limits])

        ee = self.link_indices.get(ee_link_name)
        if ee is None:
            raise KeyError(
                f"no link named {ee_link_name!r} in {urdf}; "
                f"available: {sorted(self.link_indices)}"
            )
        self.ee_link = ee
        self.ee_link_name = ee_link_name
        # Cartesian goals refer to the grasp point between the Robotiq fingers,
        # 0.15 m along tool0's local +z -- not the flange. Without this the
        # gripper overshoots every goal by its own length.
        self.tcp_offset = np.asarray(tcp_offset, dtype=float)

        # Link pairs that must never count as self-collisions: same link, links
        # welded into one rigid cluster, and clusters joined by a single joint.
        self.self_collision = self_collision
        self._ignored_self_pairs = self._build_self_ignore_set()
        # (body, robot_link) pairs excluded from obstacle checks -- used for the
        # surface the arm is bolted to, which always touches its base.
        self._ignored_env_pairs: set[tuple[int, int]] = set()

        self.set_config(HOME_CONFIG)

    # -- configuration -----------------------------------------------------

    @property
    def dof(self) -> int:
        return len(self.arm_joints)

    def set_config(self, q, zero_velocity: bool = True) -> None:
        """Teleport the arm to ``q`` (kinematic, ignores dynamics)."""
        q = np.asarray(q, dtype=float).reshape(-1)
        for joint, value in zip(self.arm_joints, q):
            if zero_velocity:
                p.resetJointState(
                    self.body_id, joint, float(value), 0.0, physicsClientId=self.client
                )
            else:
                p.resetJointState(
                    self.body_id, joint, float(value), physicsClientId=self.client
                )

    def get_config(self) -> np.ndarray:
        states = p.getJointStates(self.body_id, self.arm_joints, physicsClientId=self.client)
        return np.array([s[0] for s in states])

    def get_velocity(self) -> np.ndarray:
        states = p.getJointStates(self.body_id, self.arm_joints, physicsClientId=self.client)
        return np.array([s[1] for s in states])

    def clamp(self, q) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=float), self.lower, self.upper)

    def random_config(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.uniform(self.lower, self.upper)

    def random_free_config(
        self,
        obstacles=None,
        rng: np.random.Generator | None = None,
        margin: float = 0.0,
        max_tries: int = 500,
    ) -> np.ndarray | None:
        """Sample until a collision-free configuration is found."""
        rng = rng or np.random.default_rng()
        saved = self.get_config()
        try:
            for _ in range(max_tries):
                q = self.random_config(rng)
                if not self.in_collision(q, obstacles=obstacles, margin=margin):
                    return q
            return None
        finally:
            self.set_config(saved)

    # -- control -----------------------------------------------------------

    def control_to(self, q, position_gain: float = 0.3, velocity_scale: float = 1.0) -> None:
        """Command joint position targets (respects dynamics; needs stepping)."""
        q = self.clamp(q)
        p.setJointMotorControlArray(
            self.body_id,
            self.arm_joints,
            p.POSITION_CONTROL,
            targetPositions=[float(v) for v in q],
            forces=[float(f) for f in self.max_force],
            positionGains=[position_gain] * self.dof,
            physicsClientId=self.client,
        )

    def hold(self) -> None:
        """Command the arm to hold its current configuration."""
        self.control_to(self.get_config())

    # -- kinematics --------------------------------------------------------

    def flange_pose(self, q=None) -> tuple[np.ndarray, np.ndarray]:
        """World pose of the end-effector *link* frame (tool0 by default)."""
        if q is not None:
            saved = self.get_config()
            self.set_config(q)
        state = p.getLinkState(
            self.body_id, self.ee_link, computeForwardKinematics=True, physicsClientId=self.client
        )
        if q is not None:
            self.set_config(saved)
        return np.array(state[4]), np.array(state[5])

    def ee_pose(self, q=None) -> tuple[np.ndarray, np.ndarray]:
        """World pose of the TCP (grasp point), sharing the flange's rotation."""
        pos, orn = self.flange_pose(q)
        rot = np.asarray(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        return pos + rot @ self.tcp_offset, orn

    def link_positions(self, q=None) -> dict[str, np.ndarray]:
        """World positions of every link origin, keyed by link name."""
        if q is not None:
            saved = self.get_config()
            self.set_config(q)
        out = {}
        for link in self.all_links:
            if link == -1:
                pos, _ = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client)
            else:
                pos = p.getLinkState(
                    self.body_id, link, computeForwardKinematics=True, physicsClientId=self.client
                )[4]
            out[self.link_names[link]] = np.array(pos)
        if q is not None:
            self.set_config(saved)
        return out

    def ik(
        self,
        position,
        orientation=None,
        rest_config=None,
        iterations: int = 400,
        threshold: float = 1e-5,
        use_nullspace: bool = False,
    ) -> np.ndarray:
        """Numerical IK placing the **TCP** at ``position``, clamped to limits.

        PyBullet's IK targets a link frame, so the flange target depends on the
        achieved orientation. With ``orientation`` given that is known up front;
        otherwise the flange target is refined over a few passes until the TCP
        settles.

        By default this uses PyBullet's plain damped-least-squares solver seeded
        from the current joint state, then clamps to the joint limits. The
        null-space variant (``use_nullspace=True``) respects the limits during
        the solve but converges far less tightly -- millimetres to centimetres
        of residual on these targets, versus effectively zero -- so it is off by
        default. Always confirm the clamped result with
        :meth:`ee_position_error`, which :meth:`ik_search` does.

        The result is *not* checked for collision -- do that with
        :meth:`in_collision` before using it.
        """
        position = np.asarray(position, dtype=float)
        rest = self.clamp(HOME_CONFIG if rest_config is None else rest_config)

        def solve(flange_target) -> np.ndarray:
            kwargs = dict(
                bodyUniqueId=self.body_id,
                endEffectorLinkIndex=self.ee_link,
                targetPosition=list(flange_target),
                maxNumIterations=iterations,
                residualThreshold=threshold,
                physicsClientId=self.client,
            )
            if use_nullspace:
                kwargs.update(
                    lowerLimits=self.lower.tolist(),
                    upperLimits=self.upper.tolist(),
                    jointRanges=(self.upper - self.lower).tolist(),
                    restPoses=rest.tolist(),
                )
            if orientation is not None:
                kwargs["targetOrientation"] = list(orientation)
            return self.clamp(np.array(p.calculateInverseKinematics(**kwargs)[: self.dof]))

        if orientation is not None:
            rot = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
            return solve(position - rot @ self.tcp_offset)

        # Unconstrained orientation: alternate between solving and correcting
        # the flange target by the offset the current solution implies.
        flange_target = position - self.tcp_offset
        q = solve(flange_target)
        for _ in range(6):
            tcp, _ = self.ee_pose(q)
            error = position - tcp
            if np.linalg.norm(error) < threshold:
                break
            flange_target = flange_target + error
            q = solve(flange_target)
        return q

    def ee_position_error(self, q, target_position) -> float:
        pos, _ = self.ee_pose(q)
        return float(np.linalg.norm(pos - np.asarray(target_position, dtype=float)))

    def approach_axis(self, q=None) -> np.ndarray:
        """World direction the tool points along (its local +z)."""
        _, orn = self.flange_pose(q)
        return np.asarray(p.getMatrixFromQuaternion(orn)).reshape(3, 3)[:, 2]

    def ik_search(
        self,
        position,
        approach: str | None = None,
        orientation=None,
        obstacles=None,
        rolls: int = 16,
        seeds: int = 24,
        position_tol: float = 0.01,
        axis_tol: float = 0.30,
        rng: np.random.Generator | None = None,
        prefer=None,
    ) -> tuple[np.ndarray, dict]:
        """Search for a reachable, collision-free configuration at ``position``.

        A single IK call is unreliable here: PyBullet's solver is a local
        iterative method, the joints are limited to +-180 degrees, and a
        specific wrist roll may simply be infeasible or leave the gripper
        intersecting geometry. So this sweeps the roll about the approach axis
        and several seed configurations, and returns the best candidate --
        preferring collision-free solutions that hit the position and approach
        direction, and falling back to the closest near-miss.

        ``approach`` fixes only the tool's pointing direction and lets the roll
        float; pass ``orientation`` instead to pin the full wrist pose.

        ``prefer`` picks among *feasible* solutions instead of taking the first:
        given a configuration it returns a score to maximise. Use it when
        "collision-free" is not the real requirement — a CBF planner needs a goal
        with positive barrier clearance, which is stricter than mesh-free, so
        passing a barrier value here finds a goal the planner can actually accept.
        Every candidate is evaluated when it is supplied, so this costs
        ``rolls * seeds`` IK solves rather than stopping early.

        Returns ``(q, info)`` where ``info['feasible']`` says whether the
        solution satisfies every tolerance and is collision-free.
        """
        rng = rng or np.random.default_rng(0)
        position = np.asarray(position, dtype=float)

        if orientation is not None:
            orientations = [orientation]
        elif approach is not None:
            orientations = [
                tool_orientation(approach, roll)
                for roll in np.linspace(0.0, 2 * np.pi, rolls, endpoint=False)
            ]
        else:
            orientations = [None]

        want_axis = APPROACH_AXES[approach] if approach is not None else None

        seed_configs = [self.get_config(), HOME_CONFIG]
        seed_configs += [self.random_config(rng) for _ in range(max(0, seeds - 2))]

        saved = self.get_config()
        # Two independent bests: the best feasible solution, and the closest
        # near-miss to fall back on. Keeping them apart stops a near-miss from
        # ever outranking something that actually satisfies the tolerances.
        best_q, best_info, best_key = None, None, None
        chosen_q, chosen_info, chosen_score = None, None, -np.inf
        tried = 0
        try:
            for orn in orientations:
                for seed in seed_configs:
                    tried += 1
                    self.set_config(self.clamp(seed))
                    q = self.ik(position, orn, rest_config=seed)
                    pos_error = self.ee_position_error(q, position)
                    axis_error = 0.0
                    if want_axis is not None:
                        cos = float(np.clip(np.dot(self.approach_axis(q), want_axis), -1.0, 1.0))
                        axis_error = float(np.arccos(cos))
                    colliding = self.in_collision(q, obstacles=obstacles)
                    on_target = pos_error <= position_tol and axis_error <= axis_tol
                    feasible = on_target and not colliding

                    # Rank lexicographically: a solution that actually reaches
                    # the pose beats one that does not, and only then does
                    # collision-freedom decide. Scoring these into a single
                    # number lets a wildly-off-target pose win just for being
                    # collision-free, which is worse than useless.
                    key = (not on_target, colliding, pos_error + 0.05 * axis_error)
                    info = {
                        "feasible": feasible,
                        "on_target": on_target,
                        "position_error": pos_error,
                        "axis_error": axis_error,
                        "collision_free": not colliding,
                        "orientation": orn,
                        "attempts": tried,
                    }

                    if feasible:
                        if prefer is None:
                            return q, info
                        score = float(prefer(q))
                        if score > chosen_score:
                            chosen_q, chosen_score = q, score
                            chosen_info = {**info, "preference": score}
                        continue

                    if best_key is None or key < best_key:
                        best_key, best_q, best_info = key, q, info

            if chosen_info is not None:
                return chosen_q, chosen_info
            return best_q, best_info
        finally:
            self.set_config(saved)

    # -- collision ---------------------------------------------------------

    def _build_self_ignore_set(self) -> set[tuple[int, int]]:
        """Link pairs that can never meaningfully self-collide.

        Fixed joints weld links into rigid clusters (the whole Robotiq gripper
        is one), and neighbouring clusters share a joint axis, so their meshes
        touch by construction. Both cases are excluded; everything else is a
        genuine self-collision candidate.

        This filtering has to be done by hand because ``getClosestPoints``
        bypasses PyBullet's collision filter groups -- the URDF self-collision
        flags only affect the physics solver, not distance queries.
        """
        n = p.getNumJoints(self.body_id, physicsClientId=self.client)
        links = list(range(-1, n))
        parent: dict[int, int] = {}
        welded: dict[int, bool] = {}
        for j in range(n):
            info = p.getJointInfo(self.body_id, j, physicsClientId=self.client)
            parent[j] = info[16]
            welded[j] = info[2] == p.JOINT_FIXED

        root = {i: i for i in links}

        def find(x: int) -> int:
            while root[x] != x:
                x = root[x] = root[root[x]]
            return x

        for j in range(n):
            if welded[j]:
                a, b = find(parent[j]), find(j)
                if a != b:
                    root[b] = a

        cluster = {i: find(i) for i in links}
        adjacent = {
            frozenset((cluster[parent[j]], cluster[j]))
            for j in range(n)
            if cluster[parent[j]] != cluster[j]
        }

        ignored = set()
        for a in links:
            for b in links:
                if a >= b:
                    continue
                if cluster[a] == cluster[b] or frozenset((cluster[a], cluster[b])) in adjacent:
                    ignored.add((a, b))
        return ignored

    def ignore_collisions_with(self, body: int, links=None) -> None:
        """Stop reporting contacts between ``body`` and the given robot links.

        Used for the mounting surface: the arm's base is bolted to the table, so
        that contact is structural, not a collision. ``links=None`` ignores the
        body entirely.
        """
        targets = self.all_links if links is None else [
            self.link_indices[n] if isinstance(n, str) else n for n in links
        ]
        for link in targets:
            self._ignored_env_pairs.add((body, link))
        # Also stop the solver from fighting the contact during dynamic runs.
        for link in targets:
            p.setCollisionFilterPair(
                self.body_id, body, link, -1, 0, physicsClientId=self.client
            )

    def _self_contacts(self, margin: float) -> list[Contact]:
        found = []
        for c in p.getClosestPoints(
            bodyA=self.body_id,
            bodyB=self.body_id,
            distance=margin,
            physicsClientId=self.client,
        ):
            link_a, link_b = c[3], c[4]
            if link_a == link_b:
                continue
            if tuple(sorted((link_a, link_b))) in self._ignored_self_pairs:
                continue
            found.append(
                Contact(
                    f"{self.link_names.get(link_a, link_a)}/{self.link_names.get(link_b, link_b)}",
                    self.body_id,
                    link_b,
                    c[8],
                )
            )
        return found

    def collision_report(
        self, q=None, obstacles=None, margin: float = 0.0, include_self: bool = True
    ) -> list[Contact]:
        """List link pairs closer than ``margin`` (``margin=0`` -> real overlap).

        ``obstacles`` is an iterable of body ids; self-collision is included
        when the robot was loaded with ``self_collision=True``.

        ``include_self=False`` reports robot-vs-environment only. Keep the two
        separable: a barrier that models the environment says nothing about the
        arm folding onto itself, so mixing them makes an out-of-model gap look
        like the environment model being wrong.
        """
        if q is not None:
            saved = self.get_config()
            self.set_config(q)
        try:
            found: list[Contact] = []
            for body in obstacles or []:
                if body == self.body_id:
                    continue
                for c in p.getClosestPoints(
                    bodyA=self.body_id,
                    bodyB=body,
                    distance=margin,
                    physicsClientId=self.client,
                ):
                    if (body, c[3]) in self._ignored_env_pairs:
                        continue
                    found.append(
                        Contact(self.link_names.get(c[3], str(c[3])), body, c[4], c[8])
                    )
            if self.self_collision and include_self:
                found.extend(self._self_contacts(margin))
            return found
        finally:
            if q is not None:
                self.set_config(saved)

    def in_collision(
        self, q=None, obstacles=None, margin: float = 0.0, include_self: bool = True
    ) -> bool:
        return (
            len(
                self.collision_report(
                    q, obstacles=obstacles, margin=margin, include_self=include_self
                )
            )
            > 0
        )

    def clearance(
        self, q=None, obstacles=None, max_distance: float = 0.5, include_self: bool = True
    ) -> float:
        """Smallest robot-to-obstacle distance, negative when penetrating."""
        report = self.collision_report(
            q, obstacles=obstacles, margin=max_distance, include_self=include_self
        )
        if not report:
            return max_distance
        return min(c.distance for c in report)

    def self_clearance(self, q=None, max_distance: float = 0.5) -> float:
        """Smallest distance between two non-adjacent links of the arm."""
        if not self.self_collision:
            return max_distance
        if q is not None:
            saved = self.get_config()
            self.set_config(q)
        try:
            contacts = self._self_contacts(max_distance)
            return min((c.distance for c in contacts), default=max_distance)
        finally:
            if q is not None:
                self.set_config(saved)
