# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-environment bounding-box helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.utils.bounding_box import OrientedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


def has_heterogeneous_objects(objects: list[PlaceableAsset]) -> bool:
    """Return whether placement must use env-specific object geometry."""
    from isaaclab_arena.assets.object_set import RigidObjectSet

    return any(isinstance(obj, RigidObjectSet) for obj in objects)


def assign_variants_for_envs(objects: list[PlaceableAsset], num_envs: int, placement_seed: int | None = None) -> None:
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


def get_bounding_box_per_env(obj: PlaceableAsset, num_envs: int) -> OrientedBoundingBox:
    """Return one local bounding box per environment."""
    from isaaclab_arena.assets.object_set import RigidObjectSet

    if isinstance(obj, RigidObjectSet):
        return obj.get_bounding_box_per_env(num_envs)

    bbox = obj.get_bounding_box()
    return OrientedBoundingBox(
        center=bbox.center.expand(num_envs, 3),
        half_extents=bbox.half_extents.expand(num_envs, 3),
        rotation_xyzw=bbox.rotation_xyzw.expand(num_envs, 4),
    )


@dataclass(frozen=True)
class PerEnvBoundingBoxes:
    """Local object bounding boxes for each environment."""

    object_bboxes: dict[PlaceableAsset, OrientedBoundingBox]
    """Boxes with center/half-extents shape (N, 3) and rotation shape (N, 4)."""

    num_envs: int
    """Number of environments N."""

    def __post_init__(self) -> None:
        assert self.num_envs >= 1, f"num_envs must be >= 1, got {self.num_envs}"
        for obj, bbox in self.object_bboxes.items():
            assert (
                bbox.center.shape[0] == self.num_envs
            ), f"Object '{obj.name}' bbox center has {bbox.center.shape[0]} envs, expected {self.num_envs}."
            assert (
                bbox.half_extents.shape[0] == self.num_envs
            ), f"Object '{obj.name}' bbox half_extents has {bbox.half_extents.shape[0]} envs, expected {self.num_envs}."
            assert bbox.rotation_xyzw.shape[0] == self.num_envs, (
                f"Object '{obj.name}' bbox rotation_xyzw has {bbox.rotation_xyzw.shape[0]} envs, expected"
                f" {self.num_envs}."
            )

    def get_bounding_boxes_for_env_id(self, env_id: int) -> dict[PlaceableAsset, OrientedBoundingBox]:
        """Return object bboxes for one env with N=1."""
        return {obj: bbox[env_id] for obj, bbox in self.object_bboxes.items()}

    def get_bounding_boxes_for_all_envs(self) -> list[dict[PlaceableAsset, OrientedBoundingBox]]:
        """Return num_envs one-env bbox dicts, each with N=1."""
        return [self.get_bounding_boxes_for_env_id(env_id) for env_id in range(self.num_envs)]

    def get_bounding_boxes_for_solver_candidates(
        self, candidates_per_env: int
    ) -> dict[PlaceableAsset, OrientedBoundingBox]:
        """Return bboxes tiled to one row per solver candidate.

        Each bbox has N=num_envs * candidates_per_env. Rows are grouped contiguously
        by env; callers recover the env via candidate_idx // candidates_per_env.
        """
        return {
            obj: OrientedBoundingBox(
                center=bbox.center.repeat_interleave(candidates_per_env, dim=0),
                half_extents=bbox.half_extents.repeat_interleave(candidates_per_env, dim=0),
                rotation_xyzw=bbox.rotation_xyzw.repeat_interleave(candidates_per_env, dim=0),
            )
            for obj, bbox in self.object_bboxes.items()
        }


def build_per_env_bounding_boxes(objects: list[PlaceableAsset], num_envs: int) -> PerEnvBoundingBoxes:
    """Build per-env local OBB geometry for each placement object."""
    object_bboxes = {obj: get_bounding_box_per_env(obj, num_envs) for obj in objects}
    return PerEnvBoundingBoxes(object_bboxes=object_bboxes, num_envs=num_envs)
