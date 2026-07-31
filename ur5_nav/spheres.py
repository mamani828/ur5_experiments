"""The spherized UR5, mirrored on the Python side.

``ompl::cbf::ClearanceBarrier`` does not see the arm's meshes — it sees the 40
collision spheres of VAMP's ``ur5_spherized.urdf``, and evaluates

    h_i(q) = d(p_i(q)) - r_i - margin

This module loads that same URDF into PyBullet and reports the same centres and
radii, so a barrier value computed here is the one the planner will compute. That
makes it possible to check, before ever invoking the planner, that a start state
is feasible and that the sphere model is a conservative stand-in for the meshes.

The sphere model does *not* enclose the links: VAMP's own coverage check finds
mesh vertices up to 30.5 mm outside it, which is part of what
``ClearanceBarrier::defaultMargin`` (0.06) exists to absorb.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pybullet as p

from .robot import ARM_JOINT_NAMES, HOME_CONFIG

SPHERIZED_URDF = "/home/mani/vamp/resources/ur5/ur5_spherized.urdf"

# ClearanceBarrier::defaultMargin -- covers sphere under-coverage, SDF
# discretisation and step linearisation.
DEFAULT_MARGIN = 0.06


@dataclass
class SphereModel:
    """Centres and radii of the collision spheres at one configuration."""

    centers: np.ndarray  # (n, 3)
    radii: np.ndarray  # (n,)
    links: list[str]

    def barrier(self, distances: np.ndarray, margin: float = DEFAULT_MARGIN) -> np.ndarray:
        """``h_i = d_i - r_i - margin`` given the field value at each centre."""
        return np.asarray(distances, dtype=float) - self.radii - margin


class SpherizedUR5:
    """The spherized arm, posed by configuration.

    Loaded into its own PyBullet client so it never interferes with the scene
    being measured.
    """

    def __init__(self, client: int, urdf: str = SPHERIZED_URDF) -> None:
        if not os.path.exists(urdf):
            raise FileNotFoundError(f"spherized UR5 URDF not found: {urdf}")
        self.client = client
        p.setAdditionalSearchPath(os.path.dirname(os.path.abspath(urdf)), physicsClientId=client)
        self.body_id = p.loadURDF(urdf, useFixedBase=True, physicsClientId=client)

        name_to_joint = {}
        for j in range(p.getNumJoints(self.body_id, physicsClientId=client)):
            info = p.getJointInfo(self.body_id, j, physicsClientId=client)
            name_to_joint[info[1].decode()] = j
        self.arm_joints = [name_to_joint[n] for n in ARM_JOINT_NAMES]

        # Sphere placements are fixed in their link frames, so gather them once.
        self._shapes: list[tuple[int, tuple, tuple, float, str]] = []
        for link in range(-1, p.getNumJoints(self.body_id, physicsClientId=client)):
            link_name = (
                "base_link"
                if link == -1
                else p.getJointInfo(self.body_id, link, physicsClientId=client)[12].decode()
            )
            for shape in p.getCollisionShapeData(self.body_id, link, physicsClientId=client):
                if shape[2] != p.GEOM_SPHERE:
                    continue
                self._shapes.append((link, shape[5], shape[6], shape[3][0], link_name))

    @property
    def n_spheres(self) -> int:
        return len(self._shapes)

    def at(self, q=HOME_CONFIG) -> SphereModel:
        """Sphere centres in world coordinates at configuration ``q``."""
        for joint, value in zip(self.arm_joints, np.asarray(q, dtype=float)):
            p.resetJointState(self.body_id, joint, float(value), 0.0, physicsClientId=self.client)

        centers, radii, links = [], [], []
        for link, local_pos, local_orn, radius, link_name in self._shapes:
            if link == -1:
                frame_pos, frame_orn = p.getBasePositionAndOrientation(
                    self.body_id, physicsClientId=self.client
                )
            else:
                # getCollisionShapeData places shapes in the link's inertial
                # frame, which is getLinkState's [0]/[1] -- not [4]/[5].
                state = p.getLinkState(
                    self.body_id, link, computeForwardKinematics=True, physicsClientId=self.client
                )
                frame_pos, frame_orn = state[0], state[1]
            center, _ = p.multiplyTransforms(frame_pos, frame_orn, local_pos, local_orn)
            centers.append(center)
            radii.append(radius)
            links.append(link_name)
        return SphereModel(np.asarray(centers, dtype=float), np.asarray(radii), links)

    def worst_sphere(self, q, primitives, margin: float = DEFAULT_MARGIN):
        """The least-clearance sphere at ``q``: ``(h, link, centre, radius)``."""
        from .sdf import scene_distance

        model = self.at(q)
        h = model.barrier(scene_distance(primitives, model.centers), margin)
        i = int(np.argmin(h))
        return float(h[i]), model.links[i], model.centers[i], float(model.radii[i])
