# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-environment bounds and rotation fitting for placement candidates."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relations import RotateAroundSolution, get_relation
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, quaternion_to_90_deg_z_quarters
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.yaw import rotate_quat_by_yaw, yaw_from_quat_xyzw

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_candidate_batch import PlacementCandidateBatch


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


def get_bounding_box_per_env(obj: PlaceableAsset, num_envs: int) -> AxisAlignedBoundingBox:
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
    """Object bounds for N environments."""

    object_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox]
    """Per-object min/max tensors of shape (N, 3), in environment order."""
    num_envs: int
    """Number of environments N."""

    def __post_init__(self) -> None:
        assert self.num_envs >= 1, f"num_envs must be >= 1, got {self.num_envs}"
        for obj, bbox in self.object_bboxes.items():
            assert (
                bbox.min_point.shape[0] == self.num_envs
            ), f"Object '{obj.name}' bbox min_point has {bbox.min_point.shape[0]} envs, expected {self.num_envs}."
            assert (
                bbox.max_point.shape[0] == self.num_envs
            ), f"Object '{obj.name}' bbox max_point has {bbox.max_point.shape[0]} envs, expected {self.num_envs}."

    def get_bounding_boxes_for_env_id(self, env_id: int) -> dict[PlaceableAsset, AxisAlignedBoundingBox]:
        """Return object bboxes for a single env (each (1, 3)), used for per-env initialization and validation."""
        return {
            obj: AxisAlignedBoundingBox(
                min_point=bbox.min_point[env_id : env_id + 1],
                max_point=bbox.max_point[env_id : env_id + 1],
            )
            for obj, bbox in self.object_bboxes.items()
        }

    def get_bounding_boxes_for_all_envs(self) -> list[dict[PlaceableAsset, AxisAlignedBoundingBox]]:
        """Return one-env bbox dicts for every env.

        The outer list has length num_envs. Each bbox has min_point/max_point
        shape (1, 3).
        """
        return [self.get_bounding_boxes_for_env_id(env_id) for env_id in range(self.num_envs)]


def build_per_env_bounding_boxes(objects: list[PlaceableAsset], num_envs: int) -> PerEnvBoundingBoxes:
    """Build per-env base bboxes for each placement object.

    Anchor bounds include their fixed quarter-turn rotation. Movable-object bounds remain
    unrotated until candidate orientations are applied.
    """
    object_bboxes = {obj: get_bounding_box_per_env(obj, num_envs) for obj in objects}
    for obj, bbox in object_bboxes.items():
        if obj.is_anchor:
            pose = obj.get_initial_pose()
            assert isinstance(pose, Pose), f"Anchor '{obj.name}' must have a fixed Pose"
            try:
                quarters = quaternion_to_90_deg_z_quarters(obj.get_bounding_box_rotation())
            except AssertionError as error:
                raise AssertionError(f"Anchor '{obj.name}': {error}") from error
            object_bboxes[obj] = bbox.rotated_90_around_z(quarters)
    return PerEnvBoundingBoxes(object_bboxes=object_bboxes, num_envs=num_envs)


def update_candidate_bounds(
    batch: PlacementCandidateBatch,
    env_bboxes: list[dict[PlaceableAsset, AxisAlignedBoundingBox]],
) -> None:
    """Replace each candidate's bboxes in place using base geometry and its current orientations."""
    objects = list(env_bboxes[0])
    base_bounds = {
        obj: AxisAlignedBoundingBox(
            torch.cat([env_bboxes[candidate.env_id][obj].min_point for candidate in batch.candidates]),
            torch.cat([env_bboxes[candidate.env_id][obj].max_point for candidate in batch.candidates]),
        )
        for obj in objects
    }
    rotated = rotate_candidate_bboxes(objects, base_bounds, [candidate.orientations for candidate in batch.candidates])
    for index, candidate in enumerate(batch.candidates):
        candidate.bboxes = {obj: bounds[index] for obj, bounds in rotated.items()}


def rotate_candidate_bboxes(
    objects: list[PlaceableAsset],
    candidate_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
    orientations_per_candidate: list[dict[PlaceableAsset, float]],
) -> dict[PlaceableAsset, AxisAlignedBoundingBox]:
    """Enclose RotateAroundSolution roll/pitch and candidate world yaw in axis-aligned bounds.

    Supply base bounds: movable objects are unrotated; anchors include their fixed rotation.
    Returned bounds remain relative to object origins. Inputs are not modified.
    """
    num_candidates = len(orientations_per_candidate)
    rotated: dict[PlaceableAsset, AxisAlignedBoundingBox] = {}
    for obj in objects:
        bbox = candidate_bboxes[obj]
        marker = get_relation(obj, RotateAroundSolution)
        marker_rotation = marker.get_rotation_xyzw() if marker is not None else (0.0, 0.0, 0.0, 1.0)
        has_roll_pitch = marker is not None and (marker.roll_rad != 0.0 or marker.pitch_rad != 0.0)
        # orientations carries absolute world yaw; subtract the marker's own yaw to get the delta to compose.
        marker_yaw = yaw_from_quat_xyzw(marker_rotation)
        extra_yaws = [orientations_per_candidate[c].get(obj, marker_yaw) - marker_yaw for c in range(num_candidates)]
        # Preserve the original bounds exactly when no rotation is applied.
        if not has_roll_pitch and marker_yaw == 0.0 and all(yaw == 0.0 for yaw in extra_yaws):
            rotated[obj] = bbox
        else:
            quats = [rotate_quat_by_yaw(marker_rotation, yaw) for yaw in extra_yaws]
            quat_tensor = torch.tensor(quats, dtype=torch.float32, device=bbox.min_point.device)
            rotated[obj] = bbox.rotated_by_quat(quat_tensor)
    return rotated
