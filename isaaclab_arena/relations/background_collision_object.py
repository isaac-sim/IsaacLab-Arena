# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Passive collision object that represents fixed scene geometry."""

from __future__ import annotations

import trimesh
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.trimesh import bounding_box_from_mesh, mesh_in_world_frame

if TYPE_CHECKING:
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.relations import RelationBase


class FixedCollisionObject:
    """Collision-only obstacle wrapping a single world-frame mesh with an identity pose."""

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        name: str = "fixed_collision_mesh",
    ) -> None:
        self.name = name
        self.collision_mode = CollisionMode.MESH
        # Whole-scene fixed meshes should use raw geometry rather than convex-hull repair.
        self.repair_collision_mesh_non_watertight = False
        self._pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        self._mesh = mesh
        self._bounding_box = bounding_box_from_mesh(self._mesh)

    @property
    def is_anchor(self) -> bool:
        """Return False; this collision-only object is not a relation anchor."""
        return False

    def get_relations(self) -> list[RelationBase]:
        """Return no relations; this object is collision-only."""
        return []

    def get_spatial_relations(self) -> list[RelationBase]:
        """Return no spatial relations; the object is collision-only."""
        return []

    def get_initial_pose(self) -> Pose:
        """Return identity pose because the mesh is already baked into world coordinates."""
        return self._pose

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return the mesh bounds; identical to the world bounds since the mesh is in world frame."""
        return self._bounding_box

    def get_world_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return the mesh bounds in world frame."""
        return self._bounding_box

    def get_collision_mesh(self) -> trimesh.Trimesh:
        """Return the mesh in world coordinates."""
        return self._mesh


def make_fixed_collision_objects(objects: Sequence[CollisionObject]) -> list[CollisionObject]:
    """Combine the objects' collision meshes into one FixedCollisionObject.

    Objects in BBOX mode or without an extractable mesh are returned unchanged;
    a whole-scene Background that cannot aggregate is an error.
    """
    from isaaclab_arena.assets.background import Background

    mesh, skipped_objects = _combine_fixed_meshes(objects)
    collision_objects: list[CollisionObject] = []
    if mesh is not None:
        collision_objects.append(FixedCollisionObject(mesh))
    if skipped_objects:
        aabb_fallback_objects = [obj for obj in skipped_objects if not isinstance(obj, Background)]
        bbox_backgrounds = [
            obj for obj in skipped_objects if isinstance(obj, Background) and obj.collision_mode == CollisionMode.BBOX
        ]
        skipped_backgrounds = [
            obj for obj in skipped_objects if isinstance(obj, Background) and obj.collision_mode != CollisionMode.BBOX
        ]
        assert not bbox_backgrounds, (
            "Whole-scene Background assets cannot use explicit BBOX collision because their AABBs span the full scene: "
            f"{[obj.name for obj in bbox_backgrounds]}."
        )
        assert not skipped_backgrounds, (
            "Cannot build background collision mesh; mesh extraction failed for whole-scene Background assets "
            f"{[obj.name for obj in skipped_backgrounds]}."
        )
        if aabb_fallback_objects:
            print(
                "Fixed collision mesh extraction failed for "
                f"{[obj.name for obj in aabb_fallback_objects]}; keeping them as individual AABB collision obstacles."
            )
            collision_objects.extend(aabb_fallback_objects)
    return collision_objects


def _combine_fixed_meshes(objects: Sequence[CollisionObject]) -> tuple[trimesh.Trimesh | None, list[CollisionObject]]:
    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache

    manager = WarpMeshAndSphereCache(device="cpu")
    meshes = []
    skipped_objects = []
    for obj in objects:
        if obj.collision_mode == CollisionMode.BBOX:
            skipped_objects.append(obj)
            continue
        mesh = manager.get_collision_mesh(obj)
        if mesh is None:
            skipped_objects.append(obj)
            continue
        pose = obj.get_initial_pose()
        if pose is None and isinstance(obj, Background):
            pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        assert isinstance(pose, Pose), f"Fixed collision object '{obj.name}' must have a fixed Pose."
        meshes.append(mesh_in_world_frame(mesh, pose))
    if not meshes:
        return None, skipped_objects
    return trimesh.util.concatenate(meshes), skipped_objects
