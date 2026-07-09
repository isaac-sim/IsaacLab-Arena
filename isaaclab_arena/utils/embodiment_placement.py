# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""USD-derived placement footprints for robot embodiments."""

from __future__ import annotations

from dataclasses import dataclass, field

from isaaclab_arena.utils.bounding_box import (
    AxisAlignedBoundingBox,
    quaternion_to_90_deg_z_quarters,
    union_bounding_boxes,
)
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd_helpers import compute_local_bounding_box_from_usd


@dataclass(frozen=True)
class PlacementUsdSource:
    """One USD asset contributing to an embodiment placement footprint."""

    usd_path: str
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    offset_pose: Pose = field(default_factory=Pose.identity)


def compute_embodiment_placement_bbox(sources: list[PlacementUsdSource]) -> AxisAlignedBoundingBox:
    """Union local bounding boxes from one or more USD sources relative to the embodiment origin."""
    assert sources, "At least one PlacementUsdSource is required"
    boxes: list[AxisAlignedBoundingBox] = []
    for source in sources:
        bbox = compute_local_bounding_box_from_usd(source.usd_path, source.scale)
        if source.offset_pose.position_xyz != (0.0, 0.0, 0.0) or source.offset_pose.rotation_xyzw != (
            0.0,
            0.0,
            0.0,
            1.0,
        ):
            quarters = quaternion_to_90_deg_z_quarters(source.offset_pose.rotation_xyzw)
            bbox = bbox.rotated_90_around_z(quarters).translated(source.offset_pose.position_xyz)
        boxes.append(bbox)
    return union_bounding_boxes(*boxes)
