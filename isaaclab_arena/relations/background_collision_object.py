# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Passive collision object that represents all fixed scene background geometry."""

from __future__ import annotations

import numpy as np
import trimesh
from typing import TYPE_CHECKING

from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


class BackgroundCollisionObject:
    """Single fixed collision-only object built from multiple background meshes."""

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        name: str = "__background_collision_mesh__",
    ) -> None:
        self.name = name
        self.relations = []
        self.use_collision_mesh_as_is = True
        self._pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        self._mesh = mesh
        self._bounding_box = _bounding_box_from_mesh(self._mesh)

    @property
    def is_anchor(self) -> bool:
        """Background collision geometry is fixed but not part of the relation graph."""
        return False

    def get_relations(self) -> list:
        """Return no relations; this object is collision-only."""
        return self.relations

    def get_spatial_relations(self) -> list:
        """Return no spatial relations."""
        return []

    def get_initial_pose(self) -> Pose:
        """Return identity pose because the mesh is already baked into world coordinates."""
        return self._pose

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return the world-space mesh bounds in the identity local frame."""
        return self._bounding_box

    def get_world_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return the world-space bounds of the combined background mesh."""
        return self._bounding_box

    def get_collision_mesh(self) -> trimesh.Trimesh:
        """Return the combined background mesh in world coordinates."""
        return self._mesh


def make_background_collision_object(objects: list[ObjectBase]) -> BackgroundCollisionObject | None:
    """Build one fixed collision object from the full background mesh."""
    mesh = _combine_background_meshes(objects)
    if mesh is None:
        return None
    return BackgroundCollisionObject(mesh)


def _combine_background_meshes(objects: list[ObjectBase]) -> trimesh.Trimesh | None:
    manager = WarpMeshAndSphereCache(device="cpu")
    meshes = []
    for obj in objects:
        mesh = manager.get_collision_mesh(obj)
        if mesh is None:
            continue
        pose = obj.get_initial_pose()
        assert isinstance(pose, Pose), f"Background object '{obj.name}' must have a fixed Pose."
        meshes.append(_mesh_in_world_frame(mesh, pose))
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def _mesh_in_world_frame(mesh: trimesh.Trimesh, pose: Pose) -> trimesh.Trimesh:
    transformed = mesh.copy()
    qx, qy, qz, qw = pose.rotation_xyzw
    transform = trimesh.transformations.quaternion_matrix([qw, qx, qy, qz])
    transform[:3, 3] = np.asarray(pose.position_xyz, dtype=np.float64)
    transformed.apply_transform(transform)
    return transformed


def _bounding_box_from_mesh(mesh: trimesh.Trimesh) -> AxisAlignedBoundingBox:
    bounds = mesh.bounds
    return AxisAlignedBoundingBox(
        min_point=tuple(float(v) for v in bounds[0]),
        max_point=tuple(float(v) for v in bounds[1]),
    )
