# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounding-box helpers for heterogeneous placement.

Keeps num_envs and per-env geometry logic out of ObjectBase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


def is_heterogeneous(obj: ObjectBase) -> bool:
    """True if obj has different bounding boxes per environment (e.g. multi-variant RigidObjectSet)."""
    return getattr(obj, "has_env_specific_bboxes", False)


def has_heterogeneous_objects(objects: list[ObjectBase]) -> bool:
    """True if any object in the list varies across environments."""
    return any(is_heterogeneous(obj) for obj in objects)


def get_bounding_box_per_env(obj: ObjectBase, num_envs: int) -> AxisAlignedBoundingBox:
    """Return bounding boxes expanded to (num_envs, 3).

    Heterogeneous objects delegate to their own get_bounding_box_per_env.
    Homogeneous objects just broadcast the single bbox.
    """
    if is_heterogeneous(obj):
        return obj.get_bounding_box_per_env(num_envs)

    bbox = obj.get_bounding_box()
    return AxisAlignedBoundingBox(
        min_point=bbox.min_point.expand(num_envs, 3),
        max_point=bbox.max_point.expand(num_envs, 3),
    )
