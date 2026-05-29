# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounding-box helpers for heterogeneous placement.

Keeps num_envs and per-env geometry logic out of ObjectBase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


def has_heterogeneous_objects(objects: list[ObjectBase]) -> bool:
    """Return whether placement must use env-specific object geometry."""
    from isaaclab_arena.assets.object_set import RigidObjectSet

    return any(isinstance(obj, RigidObjectSet) for obj in objects)


def assign_variants_for_envs(objects: list[ObjectBase], num_envs: int, placement_seed: int | None = None) -> None:
    """Assign per-env variants on every RigidObjectSet in the list.

    Placers call this once they know the real environment count, before
    requesting per-env bounding boxes. Non-RigidObjectSet objects are skipped.
    Seeded assignments offset each set by its index so multiple sets do not
    reuse the same random sequence.
    """
    from isaaclab_arena.assets.object_set import RigidObjectSet

    variant_set_idx = 0
    for obj in objects:
        if isinstance(obj, RigidObjectSet):
            variant_seed = None if placement_seed is None else placement_seed + variant_set_idx
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


@dataclass(frozen=True)
class PerEnvBoundingBoxes:
    """Per-env object bboxes, exposed in three layouts:

    - get_bounding_boxes_for_env_id: one dict for a single env, bboxes (1, 3).
    - get_bounding_boxes_for_all_envs: list[dict] of length num_envs, each bbox (1, 3).
    - get_bounding_boxes_for_solver_candidates: one dict tiled to
      (num_envs * candidates_per_env, 3), grouped contiguously by env.
    """

    object_bboxes: dict[ObjectBase, AxisAlignedBoundingBox]
    num_envs: int

    def __post_init__(self) -> None:
        assert self.num_envs >= 1, f"num_envs must be >= 1, got {self.num_envs}"
        for obj, bbox in self.object_bboxes.items():
            assert (
                bbox.min_point.shape[0] == self.num_envs
            ), f"Object '{obj.name}' bbox min_point has {bbox.min_point.shape[0]} envs, expected {self.num_envs}."
            assert (
                bbox.max_point.shape[0] == self.num_envs
            ), f"Object '{obj.name}' bbox max_point has {bbox.max_point.shape[0]} envs, expected {self.num_envs}."

    def get_bounding_boxes_for_env_id(self, env_id: int) -> dict[ObjectBase, AxisAlignedBoundingBox]:
        """Return object bboxes for a single env (each (1, 3)), used for per-env initialization and validation."""
        return {
            obj: AxisAlignedBoundingBox(
                min_point=bbox.min_point[env_id : env_id + 1],
                max_point=bbox.max_point[env_id : env_id + 1],
            )
            for obj, bbox in self.object_bboxes.items()
        }

    def get_bounding_boxes_for_all_envs(self) -> list[dict[ObjectBase, AxisAlignedBoundingBox]]:
        """Return one-env bbox dicts for every env.

        The outer list has length num_envs. Each bbox has min_point/max_point
        shape (1, 3).
        """
        return [self.get_bounding_boxes_for_env_id(env_id) for env_id in range(self.num_envs)]

    def get_bounding_boxes_for_solver_candidates(
        self, candidates_per_env: int
    ) -> dict[ObjectBase, AxisAlignedBoundingBox]:
        """Return bboxes tiled to one row per solver candidate.

        Each bbox has shape (num_envs * candidates_per_env, 3). Rows are grouped
        contiguously by env: rows [i * candidates_per_env : (i + 1) * candidates_per_env]
        all hold env i's bbox. Callers recover the env via candidate_idx // candidates_per_env.
        """
        return {
            obj: AxisAlignedBoundingBox(
                min_point=bbox.min_point.repeat_interleave(candidates_per_env, dim=0),
                max_point=bbox.max_point.repeat_interleave(candidates_per_env, dim=0),
            )
            for obj, bbox in self.object_bboxes.items()
        }


def build_per_env_bounding_boxes(objects: list[ObjectBase], num_envs: int) -> PerEnvBoundingBoxes:
    """Build per-env bboxes for each placement object."""
    object_bboxes = {obj: get_bounding_box_per_env(obj, num_envs) for obj in objects}
    return PerEnvBoundingBoxes(object_bboxes=object_bboxes, num_envs=num_envs)
