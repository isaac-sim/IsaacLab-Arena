# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


class CollisionMode(Enum):
    """Collision-detection method for no-overlap constraints."""

    BBOX = "bbox"
    """Axis-aligned bounding box penetration (fast, conservative)."""

    MESH = "mesh"
    """Sphere-to-SDF queries against actual mesh geometry (accurate, slower)."""


def get_object_collision_mode(obj: CollisionObject, default: CollisionMode) -> CollisionMode:
    """Return an object's collision mode, falling back to the solver default."""
    return default if obj.collision_mode is None else obj.collision_mode


def object_uses_mesh_collision(obj: CollisionObject, default: CollisionMode) -> bool:
    """Return True when the object's effective collision mode is MESH."""
    return get_object_collision_mode(obj, default) == CollisionMode.MESH


def pair_is_covered_by_mesh_collision(
    subject: CollisionObject,
    obstacle: CollisionObject,
    subject_bbox: AxisAlignedBoundingBox,
    obstacle_bbox: AxisAlignedBoundingBox,
    mesh_manager: WarpMeshAndSphereCache,
    default_collision_mode: CollisionMode,
    obstacle_is_fixed: bool,
) -> bool:
    """Return whether the mesh loss represents a collision pair."""
    subject_has_mesh = (
        object_uses_mesh_collision(subject, default_collision_mode)
        and mesh_manager.get_collision_mesh(subject) is not None
    )
    obstacle_has_mesh = (
        object_uses_mesh_collision(obstacle, default_collision_mode)
        and mesh_manager.get_collision_mesh(obstacle) is not None
    )
    subject_has_mesh_or_invariant_bbox = subject_has_mesh or subject_bbox.is_batch_invariant()
    if obstacle_is_fixed:
        return obstacle_has_mesh and subject_has_mesh_or_invariant_bbox
    obstacle_has_mesh_or_invariant_bbox = obstacle_has_mesh or obstacle_bbox.is_batch_invariant()
    return (
        (subject_has_mesh or obstacle_has_mesh)
        and subject_has_mesh_or_invariant_bbox
        and obstacle_has_mesh_or_invariant_bbox
    )
