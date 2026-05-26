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

VARIANT_SEED_STRIDE = 1_000_003


def has_heterogeneous_objects(objects: list[ObjectBase]) -> bool:
    """Return whether placement must use env-specific object geometry."""
    from isaaclab_arena.assets.object_set import RigidObjectSet

    return any(isinstance(obj, RigidObjectSet) for obj in objects)


def assign_variants_for_envs(objects: list[ObjectBase], num_envs: int, placement_seed: int | None = None) -> None:
    """Assign per-env variants on every RigidObjectSet in the list.

    Placers call this once they know the real environment count, before
    requesting per-env bounding boxes. Objects without variants are ignored.
    """
    from isaaclab_arena.assets.object_set import RigidObjectSet

    variant_set_idx = 0
    for obj in objects:
        if isinstance(obj, RigidObjectSet):
            variant_seed = None if placement_seed is None else placement_seed + VARIANT_SEED_STRIDE * variant_set_idx
            obj.assign_variants(num_envs, variant_seed=variant_seed)
            variant_set_idx += 1


def get_bounding_box_per_env(obj: ObjectBase, num_envs: int) -> AxisAlignedBoundingBox:
    """Return bounding boxes expanded to (num_envs, 3).

    RigidObjectSet delegates to its own get_bounding_box_per_env.
    All other objects broadcast their single bbox.
    """
    from isaaclab_arena.assets.object_set import RigidObjectSet

    if isinstance(obj, RigidObjectSet):
        return obj.get_bounding_box_per_env(num_envs)

    bbox = obj.get_bounding_box()
    return AxisAlignedBoundingBox(
        min_point=bbox.min_point.expand(num_envs, 3),
        max_point=bbox.max_point.expand(num_envs, 3),
    )
