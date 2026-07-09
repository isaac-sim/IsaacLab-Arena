# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


class CollisionMode(Enum):
    """Collision-detection method for no-overlap constraints.

    Individual objects may override this default. A pair uses mesh collision when either
    object resolves to MESH and provides a mesh; the meshless side is approximated by its AABB.
    """

    BBOX = "bbox"
    """Axis-aligned bounding box overlap volume (fast, conservative)."""

    MESH = "mesh"
    """Sphere-to-SDF queries against actual mesh geometry (accurate, slower)."""


def get_object_collision_mode(obj: ObjectBase, default: CollisionMode) -> CollisionMode:
    """Return an object's collision mode, falling back to the solver default."""
    return default if obj.collision_mode is None else obj.collision_mode


def object_uses_mesh_collision(obj: ObjectBase, default: CollisionMode) -> bool:
    """Return True when the object's effective collision mode is MESH."""
    return get_object_collision_mode(obj, default) == CollisionMode.MESH
