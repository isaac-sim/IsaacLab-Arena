# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relations import ClutterOn, On, get_relation
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


def get_initial_pose_or_assert_fail(obj: PlaceableAsset) -> Pose:
    """Return obj's initial pose, asserting it is a fixed pose rather than a distribution."""
    initial_pose = obj.get_initial_pose()
    assert isinstance(
        initial_pose, Pose
    ), f"Object '{obj.name}' must have a fixed Pose before placement, got {type(initial_pose).__name__}."
    return initial_pose


def get_world_bbox_at_initial_pose(
    obj: PlaceableAsset, asset_to_bbox: dict[PlaceableAsset, AxisAlignedBoundingBox]
) -> AxisAlignedBoundingBox:
    """Return obj's local bbox translated to its fixed initial pose."""
    return asset_to_bbox[obj].translated(get_initial_pose_or_assert_fail(obj).position_xyz)


def sample_uniform_or_midpoint(low: float, high: float, generator: torch.Generator | None = None) -> float:
    """Sample uniformly from [low, high], falling back to the midpoint when the interval is empty.

    An empty interval means the caller's constraints cannot all hold, which happens for example
    when a child is wider than the surface it sits on. The midpoint keeps the seed centred on the
    intended region and lets the solver resolve the conflict.
    """
    if low >= high:
        return (low + high) / 2.0
    return low + (high - low) * torch.rand(1, generator=generator).item()


def get_child_bbox_given_parent_position(
    obj: PlaceableAsset,
    parent_world_bbox: AxisAlignedBoundingBox,
    asset_to_bbox: dict[PlaceableAsset, AxisAlignedBoundingBox],
) -> AxisAlignedBoundingBox:
    """Return the positions at which obj sits on parent_world_bbox, as a box.

    X and Y span the parent's footprint inset by the child's extents, or, for ``ClutterOn``, its
    release region inset the same way. An axis the child is too
    large for has no such position and collapses to the parent's center. Z is a single value: the
    height at which the child's bottom face meets the parent's top surface plus the relation's
    clearance.

    Args:
        obj: The object being seeded; must carry an ``On`` relation.
        parent_world_bbox: World-space bbox of the parent, shape (1, 3).
        asset_to_bbox: Local bounding box per object for the env this candidate belongs to.
    """
    on_relation = get_relation(obj, On)
    if isinstance(on_relation, ClutterOn):
        # Clutter is released over a sub-region of the parent's surface, not all of it.
        parent_world_bbox = on_relation.get_release_region_bbox(parent_world_bbox)
    child_bbox = asset_to_bbox[obj]

    child_min, child_max = child_bbox.min_point[0], child_bbox.max_point[0]
    if on_relation.overlap:
        # Intersection compares the child's far edge with the parent's near edge.
        child_min, child_max = child_max, child_min

    parent_min, parent_max = parent_world_bbox.min_point[0], parent_world_bbox.max_point[0]
    footprint_min = parent_min[:2] - child_min[:2]
    footprint_max = parent_max[:2] - child_max[:2]
    child_fits = footprint_min < footprint_max
    parent_center_xy = (parent_min[:2] + parent_max[:2]) / 2.0
    min_xy = torch.where(child_fits, footprint_min, parent_center_xy)
    max_xy = torch.where(child_fits, footprint_max, parent_center_xy)

    resting_z = parent_max[2] + on_relation.clearance_m - child_bbox.min_point[0, 2]
    return AxisAlignedBoundingBox(
        min_point=torch.cat([min_xy, resting_z.unsqueeze(0)]),
        max_point=torch.cat([max_xy, resting_z.unsqueeze(0)]),
    )


def sample_position_in_bbox(
    bbox: AxisAlignedBoundingBox, generator: torch.Generator | None = None
) -> tuple[float, float, float]:
    """Sample a position uniformly from bbox, each axis independently.

    An axis whose bounds coincide samples to that value.

    Args:
        bbox: Box of candidate positions, shape (1, 3).
        generator: RNG for reproducible sampling. None uses PyTorch's global RNG.
    """
    position = [
        sample_uniform_or_midpoint(float(bbox.min_point[0, axis]), float(bbox.max_point[0, axis]), generator)
        for axis in range(3)
    ]
    return (position[0], position[1], position[2])
