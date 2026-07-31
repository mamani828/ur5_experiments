"""Simulator session management: connection, ground, camera, image capture."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pybullet as p
import pybullet_data


@dataclass
class SimSession:
    """A PyBullet connection with a ground plane and convenience helpers.

    The UR5 URDF resolves its meshes through ``package://meshes/...``, which
    PyBullet only finds via the *additional search path*. Since that path is a
    single global slot, everything from ``pybullet_data`` is loaded by absolute
    path instead and the search path is reserved for the robot's asset root.
    """

    gui: bool = True
    timestep: float = 1.0 / 240.0
    gravity: float = -9.81
    ground: bool = True
    client: int = field(init=False, default=-1)
    plane_id: int = field(init=False, default=-1)

    def __post_init__(self) -> None:
        if self.gui and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print("[sim] no display detected, falling back to DIRECT mode")
            self.gui = False

        self.client = p.connect(p.GUI if self.gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("failed to connect to PyBullet")

        p.setTimeStep(self.timestep, physicsClientId=self.client)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.client)

        if self.gui:
            # The side panels eat framerate and screen space; the RGB preview
            # windows in particular are expensive at every step.
            for flag in (
                p.COV_ENABLE_GUI,
                p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
            ):
                p.configureDebugVisualizer(flag, 0, physicsClientId=self.client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.client)
            self.set_camera()

        if self.ground:
            plane = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
            self.plane_id = p.loadURDF(plane, physicsClientId=self.client)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self.client >= 0 and p.isConnected(self.client):
            p.disconnect(physicsClientId=self.client)
        self.client = -1

    def __enter__(self) -> "SimSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            p.stepSimulation(physicsClientId=self.client)

    # -- viewing -----------------------------------------------------------

    def set_camera(
        self,
        distance: float = 2.4,
        yaw: float = 50.0,
        pitch: float = -30.0,
        target: tuple[float, float, float] = (0.0, 0.0, 0.9),
    ) -> None:
        if self.gui:
            p.resetDebugVisualizerCamera(
                distance, yaw, pitch, target, physicsClientId=self.client
            )

    def capture(
        self,
        width: int = 960,
        height: int = 720,
        distance: float = 2.4,
        yaw: float = 50.0,
        pitch: float = -30.0,
        target: tuple[float, float, float] = (0.0, 0.0, 0.9),
    ) -> np.ndarray:
        """Render an offscreen RGB frame. Works in both GUI and DIRECT mode."""
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=0,
            upAxisIndex=2,
        )
        proj = p.computeProjectionMatrixFOV(
            fov=60.0, aspect=width / height, nearVal=0.05, farVal=10.0
        )
        # ER_TINY_RENDERER works headlessly; the hardware renderer needs a GL
        # context that DIRECT mode does not have.
        renderer = p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER
        _, _, rgba, _, _ = p.getCameraImage(
            width,
            height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=renderer,
            physicsClientId=self.client,
        )
        return np.reshape(np.asarray(rgba, dtype=np.uint8), (height, width, 4))[:, :, :3]

    def save_frame(self, path: str, **kwargs) -> str:
        """Write a captured frame to ``path`` as PNG."""
        rgb = self.capture(**kwargs)
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("saving frames requires Pillow (pip install pillow)") from exc
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        Image.fromarray(rgb).save(path)
        return path

    # -- debug drawing -----------------------------------------------------

    def draw_marker(
        self,
        position,
        rgb=(0.1, 0.8, 0.2),
        radius: float = 0.03,
        alpha: float = 0.6,
    ) -> int:
        """Add a non-colliding translucent sphere, e.g. to show a goal."""
        vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius, rgbaColor=(*rgb, alpha), physicsClientId=self.client
        )
        return p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis,
            basePosition=list(position),
            physicsClientId=self.client,
        )

    def draw_frame(self, position, orientation, length: float = 0.1, width: float = 2.0) -> list[int]:
        """Draw an RGB axis triad at a pose."""
        rot = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        origin = np.asarray(position, dtype=float)
        ids = []
        for axis, color in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
            end = origin + rot[:, axis] * length
            ids.append(
                p.addUserDebugLine(
                    origin, end, color, lineWidth=width, physicsClientId=self.client
                )
            )
        return ids

    def draw_trace(self, points, rgb=(1.0, 0.5, 0.0), width: float = 2.0) -> list[int]:
        """Connect a sequence of points with debug lines."""
        ids = []
        for a, b in zip(points[:-1], points[1:]):
            ids.append(
                p.addUserDebugLine(a, b, rgb, lineWidth=width, physicsClientId=self.client)
            )
        return ids
