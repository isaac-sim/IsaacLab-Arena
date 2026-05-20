# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounding-box helpers for heterogeneous placement.

Keeps num_envs and per-env geometry logic out of ObjectBase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_arena.assets.object_set import RigidObjectSet
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


def has_heterogeneous_objects(objects: list[ObjectBase]) -> bool:
    """True if any object in the list is a RigidObjectSet."""
    return any(isinstance(obj, RigidObjectSet) for obj in objects)


def assign_variants_for_envs(objects: list[ObjectBase], num_envs: int) -> None:
    """Assign per-env variants on every RigidObjectSet in the list.

    Placers call this at the boundary before any per-env geometry reads.
    """
    for obj in objects:
        if isinstance(obj, RigidObjectSet):
            obj.assign_variants(num_envs)


def get_bounding_box_per_env(obj: ObjectBase, num_envs: int) -> AxisAlignedBoundingBox:
    """Return bounding boxes expanded to (num_envs, 3).

    RigidObjectSet delegates to its own get_bounding_box_per_env.
    All other objects broadcast their single bbox.
    """
    if isinstance(obj, RigidObjectSet):
        return obj.get_bounding_box_per_env(num_envs)

    bbox = obj.get_bounding_box()
    return AxisAlignedBoundingBox(
        min_point=bbox.min_point.expand(num_envs, 3),
        max_point=bbox.max_point.expand(num_envs, 3),
    )
