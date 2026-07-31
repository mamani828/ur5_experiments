"""Bake a workspace signed distance field from a PyBullet scene.

The consumer is ``ompl::sdf::GridSDF``, whose only requirement is a signed
distance: negative inside an obstacle, zero on the surface, positive outside,
and roughly 1-Lipschitz. Rather than sample PyBullet's collision queries, this
recovers the *analytic primitives* behind the scene with
``getCollisionShapeData`` and evaluates their exact distance functions. Every
environment in :mod:`ur5_nav.envs` is built from boxes, spheres and cylinders,
and ``plane.urdf`` is itself a large box, so the whole scene is covered exactly.

That matters for a CBF: the barrier consumes the field's *gradient*, so a field
that is only approximately a distance function produces a barrier whose
derivative is wrong in a way no amount of grid refinement fixes.

The grid layout mirrors ``GridSDF``'s baking constructor exactly -- node counts
``max(2, ceil(extent / voxel) + 1)``, spacing ``extent / (dims - 1)``, x fastest
in the flat array -- so the C++ side adopts the values with no resampling.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import pybullet as p

# Must match ompl::sdf::GridSDF::magic / formatVersion.
GRID_MAGIC = b"OMPLSDF1"
GRID_VERSION = 1

# ompl::robots::UR5::reachableBounds() -- the box the arm's spheres can occupy.
# Using the same bounds here keeps `inBounds()` meaning the same thing on both
# sides, and out-of-bounds queries clamp rather than extrapolate.
REACHABLE_BOUNDS = (
    np.array([-1.15, -1.15, -0.15]),
    np.array([1.15, 1.15, 2.20]),
)


@dataclass
class Primitive:
    """One analytic obstacle, placed in world coordinates.

    ``kind`` is ``"box"``, ``"sphere"`` or ``"cylinder"``. ``size`` holds the
    half-extents, the radius, or ``(radius, half_height)`` respectively.
    Cylinders are along the local z axis.
    """

    kind: str
    size: np.ndarray
    position: np.ndarray
    rotation: np.ndarray  # 3x3, world <- local
    body: int = -1

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Exact signed distance from each row of ``points`` to this solid."""
        local = (points - self.position) @ self.rotation  # R^T x, vectorised

        if self.kind == "sphere":
            return np.linalg.norm(local, axis=1) - float(self.size[0])

        if self.kind == "box":
            q = np.abs(local) - self.size
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            inside = np.minimum(q.max(axis=1), 0.0)
            return outside + inside

        if self.kind == "cylinder":
            radius, half_height = float(self.size[0]), float(self.size[1])
            radial = np.linalg.norm(local[:, :2], axis=1) - radius
            axial = np.abs(local[:, 2]) - half_height
            outside = np.sqrt(
                np.maximum(radial, 0.0) ** 2 + np.maximum(axial, 0.0) ** 2
            )
            inside = np.minimum(np.maximum(radial, axial), 0.0)
            return outside + inside

        raise ValueError(f"unsupported primitive kind {self.kind!r}")


_SHAPE_KINDS = {
    p.GEOM_BOX: "box",
    p.GEOM_SPHERE: "sphere",
    p.GEOM_CYLINDER: "cylinder",
}


def scene_primitives(client: int, bodies) -> list[Primitive]:
    """Recover the analytic primitives making up ``bodies``.

    Raises on any shape that is not a box, sphere or cylinder -- silently
    dropping an obstacle would hand the planner a field that says free space
    where there is geometry, which is the one failure mode a CBF cannot survive.
    """
    out: list[Primitive] = []
    for body in bodies:
        base_pos, base_orn = p.getBasePositionAndOrientation(body, physicsClientId=client)
        for shape in p.getCollisionShapeData(body, -1, physicsClientId=client):
            geom_type, dims = shape[2], np.asarray(shape[3], dtype=float)
            local_pos, local_orn = shape[5], shape[6]

            kind = _SHAPE_KINDS.get(geom_type)
            if kind is None:
                raise ValueError(
                    f"body {body} has collision shape type {geom_type}, which has no "
                    "analytic distance here; add a case or approximate it explicitly"
                )

            # Compose the shape's local frame with the body pose.
            pos, orn = p.multiplyTransforms(base_pos, base_orn, local_pos, local_orn)
            rotation = np.asarray(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

            if kind == "box":
                # getCollisionShapeData reports *full* extents for a box.
                size = dims / 2.0
            elif kind == "sphere":
                size = dims[:1]
            else:
                # Cylinders come back as (length, radius, 0).
                size = np.array([dims[1], dims[0] / 2.0])

            out.append(Primitive(kind, size, np.asarray(pos, dtype=float), rotation, body))
    return out


def scene_distance(primitives, points: np.ndarray) -> np.ndarray:
    """Signed distance to the union of ``primitives``.

    The union of exact fields is exact outside the solids; inside overlapping
    solids the elementwise minimum under-reports depth, which errs toward
    reporting *less* clearance and so is the safe direction for a barrier.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if not primitives:
        return np.full(len(points), np.inf)
    return np.stack([prim.distance(points) for prim in primitives]).min(axis=0)


def open_mount_hole(primitives, mount_body: int, half_width: float = 0.25) -> list[Primitive]:
    """Replace the mounting box with four boxes leaving a square opening.

    A spherized UR5 has immobile collision spheres at its own base: `base_link`
    sits exactly on the table top and `shoulder_link` 0.09 m above it. Against a
    field that includes the table both are permanently in violation (h = -0.14
    for the base at the default margin) -- and because `base_link` hangs off
    frame 0, which no joint moves, its constraint row is identically zero. A
    violated row of zeros is an infeasible QP: the filter cannot steer away from
    something no control affects, so every step returns `Blocked` and the planner
    never leaves the start state.

    Opening a hole where the arm is bolted down states the same fact that
    `UR5.ignore_collisions_with` states on the PyBullet side: the tabletop under
    the robot is its mount, not an obstacle.

    Four boxes rather than a subtracted cylinder, because the union of exact box
    fields is exact, whereas `max(d, -d_cylinder)` under-reports clearance along
    the concave rim it creates -- badly enough here to need a much wider opening
    for the same barrier value. It also touches nothing but the table: the four
    boxes inherit the original's z range, so obstacles standing *on* the table
    are untouched no matter how close to the base they sit.

    The PyBullet scene keeps its solid table; only the field the planner sees has
    the opening. That is the intended asymmetry -- PyBullet excludes the same
    contact by link instead, which a field indexed by position cannot do.
    """
    table = next((prim for prim in primitives if prim.body == mount_body), None)
    if table is None:
        raise ValueError(f"body {mount_body} is not among the primitives")
    if table.kind != "box" or not np.allclose(table.rotation, np.eye(3)):
        raise ValueError("the mounting surface must be an axis-aligned box")
    if not np.allclose(table.position[:2], 0.0):
        raise ValueError("the mounting surface must be centred on the base axis")

    x, y, z = table.size
    if half_width >= min(x, y):
        raise ValueError("opening is wider than the mounting surface")

    a = half_width
    cz = table.position[2]
    # Four slabs tiling the frame: two spanning full y, two filling the gap.
    spans = [
        ((x - a) / 2.0, y, (a + x) / 2.0, 0.0),
        ((x - a) / 2.0, y, -(a + x) / 2.0, 0.0),
        (a, (y - a) / 2.0, 0.0, (a + y) / 2.0),
        (a, (y - a) / 2.0, 0.0, -(a + y) / 2.0),
    ]
    frame = [
        Primitive(
            kind="box",
            size=np.array([hx, hy, z]),
            position=np.array([px, py, cz]),
            rotation=np.eye(3),
            body=mount_body,
        )
        for hx, hy, px, py in spans
    ]
    return [prim for prim in primitives if prim.body != mount_body] + frame


def grid_dimensions(lower, upper, voxel: float) -> np.ndarray:
    """Node counts, matching ``GridSDF::gridDimensions``."""
    extent = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    return np.maximum(2, np.ceil(extent / voxel).astype(int) + 1)


@dataclass
class BakedGrid:
    """A baked field, ready to hand to ``ompl::sdf::GridSDF``."""

    lower: np.ndarray
    upper: np.ndarray
    dims: np.ndarray
    values: np.ndarray  # shape (nx, ny, nz)

    @property
    def spacing(self) -> np.ndarray:
        extent = self.upper - self.lower
        return np.where(extent > 0.0, extent / np.maximum(self.dims - 1, 1), 0.0)

    def save(self, path: str) -> str:
        """Write the binary layout ``GridSDF::load`` reads."""
        with open(path, "wb") as handle:
            handle.write(GRID_MAGIC)
            handle.write(struct.pack("<I", GRID_VERSION))
            handle.write(struct.pack("<3I", *(int(n) for n in self.dims)))
            handle.write(struct.pack("<3d", *(float(v) for v in self.lower)))
            handle.write(struct.pack("<3d", *(float(v) for v in self.upper)))
            # Fortran order == x fastest, matching index(i,j,k) = i + nx*(j + ny*k).
            handle.write(self.values.astype("<f8").tobytes(order="F"))
        return path

    @classmethod
    def load(cls, path: str) -> "BakedGrid":
        with open(path, "rb") as handle:
            if handle.read(8) != GRID_MAGIC:
                raise ValueError(f"{path} is not an SDF grid")
            (version,) = struct.unpack("<I", handle.read(4))
            if version != GRID_VERSION:
                raise ValueError(f"{path} has unsupported version {version}")
            dims = np.array(struct.unpack("<3I", handle.read(12)), dtype=int)
            lower = np.array(struct.unpack("<3d", handle.read(24)))
            upper = np.array(struct.unpack("<3d", handle.read(24)))
            count = int(np.prod(dims))
            values = np.frombuffer(handle.read(count * 8), dtype="<f8", count=count)
        return cls(lower, upper, dims, values.reshape(dims, order="F").copy())

    def node_points(self) -> np.ndarray:
        """World coordinates of every node, in flat (Fortran) order."""
        axes = [
            self.lower[d] + np.arange(self.dims[d]) * self.spacing[d] for d in range(3)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.ravel(order="F") for m in mesh], axis=1)

    def query(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Trilinear value and its exact gradient -- mirrors ``GridSDF`` queries.

        Used to check the C++ side agrees, and to plot slices without a round
        trip through the planner.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        spacing = np.where(self.spacing > 0.0, self.spacing, 1.0)
        cell = (points - self.lower) / spacing
        cell = np.clip(cell, 0.0, self.dims - 1.0)
        base = np.minimum(np.floor(cell).astype(int), self.dims - 2)
        frac = cell - base

        values = np.zeros(len(points))
        gradient = np.zeros((len(points), 3))
        weights1d = np.stack([1.0 - frac, frac], axis=0)  # (2, N, 3)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    corner = self.values[
                        base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz
                    ]
                    offsets = (dx, dy, dz)
                    w = [weights1d[offsets[d], :, d] for d in range(3)]
                    values += w[0] * w[1] * w[2] * corner
                    signs = [1.0 if o else -1.0 for o in offsets]
                    for d in range(3):
                        other = [w[k] for k in range(3) if k != d]
                        gradient[:, d] += (
                            signs[d] * other[0] * other[1] * corner / spacing[d]
                        )
        return values, gradient


def bake(
    client: int,
    bodies,
    voxel: float = 0.02,
    bounds=REACHABLE_BOUNDS,
    exclude=(),
    primitives=None,
) -> tuple[BakedGrid, list[Primitive]]:
    """Bake the scene in ``client`` into a grid.

    ``exclude`` drops bodies from the field -- the robot itself, or goal markers,
    which are visual-only anyway. Pass ``primitives`` to bake an already-adjusted
    list (e.g. after :func:`open_mount_hole`) instead of re-reading the scene.
    Returns the grid and the primitives it came from, so a caller can compare
    interpolated against exact.
    """
    lower, upper = (np.asarray(b, dtype=float) for b in bounds)
    if primitives is None:
        keep = [b for b in bodies if b not in set(exclude)]
        primitives = scene_primitives(client, keep)

    dims = grid_dimensions(lower, upper, voxel)
    grid = BakedGrid(lower, upper, dims, np.zeros(tuple(dims)))

    axes = [lower[d] + np.arange(dims[d]) * grid.spacing[d] for d in range(3)]
    # Evaluate a z-slice at a time: one (nx*ny, 3) array per slice keeps peak
    # memory flat for fine voxels while staying fully vectorised.
    xy = np.stack(np.meshgrid(axes[0], axes[1], indexing="ij"), axis=-1).reshape(-1, 2)
    for k, z in enumerate(axes[2]):
        points = np.column_stack([xy, np.full(len(xy), z)])
        grid.values[:, :, k] = scene_distance(primitives, points).reshape(
            dims[0], dims[1]
        )
    return grid, primitives


def probe_pybullet(client: int, point, bodies, probe_radius: float = 0.005) -> float:
    """Signed distance measured by PyBullet itself, for cross-checking.

    Places a small sphere at ``point`` and takes the closest-point distance to
    each body, adding the probe radius back. Slow -- this is a validation tool,
    not something to bake with.
    """
    shape = p.createCollisionShape(p.GEOM_SPHERE, radius=probe_radius, physicsClientId=client)
    probe = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=shape, basePosition=list(point),
        physicsClientId=client,
    )
    try:
        best = np.inf
        for body in bodies:
            points = p.getClosestPoints(
                bodyA=probe, bodyB=body, distance=1.0, physicsClientId=client
            )
            for contact in points:
                best = min(best, contact[8] + probe_radius)
        return float(best)
    finally:
        p.removeBody(probe, physicsClientId=client)
