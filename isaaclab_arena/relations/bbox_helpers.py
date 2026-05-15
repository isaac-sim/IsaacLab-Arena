# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounding-box utilities for the relation solver / object placer.

These helpers centralise the logic that expands single-object bounding boxes
to per-environment tensors and detects whether any object carries
env-specific geometry (e.g. RigidObjectSet with multiple variants).

Placing this logic here (instead of on ObjectBase) keeps num_envs
and other placement/simulation details out of the asset layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


def object_has_env_specific_bboxes(obj: ObjectBase) -> bool:
    """Whether obj produces different bounding boxes per environment.

    Returns True for multi-variant RigidObjectSet instances (which
    override get_bounding_box_per_env). All other objects return False.
    """
    return getattr(obj, "has_env_specific_bboxes", False)


def any_object_has_env_specific_bboxes(objects: list[ObjectBase]) -> bool:
    """Whether any object in the list has env-specific bounding boxes."""
    return any(object_has_env_specific_bboxes(obj) for obj in objects)


def get_bounding_box_per_env(obj: ObjectBase, num_envs: int) -> AxisAlignedBoundingBox:
    """Return per-environment local bounding boxes for obj.

    For objects with env-specific variants (RigidObjectSet), delegates to the
    object's own get_bounding_box_per_env override. For all other objects,
    expands the single local bbox to (num_envs, 3).

    Args:
        obj: The object to query.
        num_envs: Number of environments.

    Returns:
        AxisAlignedBoundingBox with min_point / max_point of shape (num_envs, 3).
    """
    override = getattr(obj, "get_bounding_box_per_env", None)
    if override is not None:
        return override(num_envs)

    bbox = obj.get_bounding_box()
    return AxisAlignedBoundingBox(
        min_point=bbox.min_point.expand(num_envs, 3),
        max_point=bbox.max_point.expand(num_envs, 3),
    )
