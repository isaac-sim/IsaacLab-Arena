# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Clutter release planning above relation-placed supports."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.clutter_drop_poses import (
    ClutterDropParams,
    ClutterRegion,
    MemberDropParams,
    OccupiedFootprint,
    compute_drop_poses,
    refit_bbox_to_rotation,
)
from isaaclab_arena.relations.clutter_groups import ClutterGroup, assert_group_parameters_agree
from isaaclab_arena.relations.placement_events import get_pose_from_layout, get_rotation_xyzw
from isaaclab_arena.relations.relations import ClutteredOn, IsAnchor
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


_QUARTER_TURN_TOLERANCE_RAD = 1e-3


def region_above_support(
    support_position: tuple[float, float, float],
    support_bbox: AxisAlignedBoundingBox,
    spread: float = 1.0,
    env_index: int = 0,
    support_rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> ClutterRegion:
    """Return the top-face region of a horizontal, axis-aligned support.

    Args:
        support_position: Support position in environment frame E, shape (3,).
        support_bbox: Object-local bounds, min/max shape (N, 3); N is the environment count.
        spread: Usable fraction of the support footprint, in (0, 1].
        env_index: Environment row in support_bbox.
        support_rotation_xyzw: Support-to-E quaternion, shape (4,); yaw must be a quarter turn.
    """
    x, y, z, w = support_rotation_xyzw
    assert abs(x) < 1e-3 and abs(y) < 1e-3, (
        f"Clutter support rotation must be yaw-only, got (x={x:.4f}, y={y:.4f}). A tilted "
        "support has no single top-surface height for a pile to rest on."
    )
    yaw = 2.0 * math.atan2(z, w)

    minimum = support_bbox.min_point[env_index]
    maximum = support_bbox.max_point[env_index]
    half_x = float(maximum[0] - minimum[0]) * 0.5
    half_y = float(maximum[1] - minimum[1]) * 0.5
    local_center_x = float(maximum[0] + minimum[0]) * 0.5
    local_center_y = float(maximum[1] + minimum[1]) * 0.5

    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    center_x = support_position[0] + local_center_x * cos_yaw - local_center_y * sin_yaw
    center_y = support_position[1] + local_center_x * sin_yaw + local_center_y * cos_yaw

    quarter_turns = round(yaw / (math.pi / 2.0))
    assert abs(yaw - quarter_turns * (math.pi / 2.0)) < _QUARTER_TURN_TOLERANCE_RAD, (
        f"Clutter support is turned {math.degrees(yaw):.3f} degrees, which is not a quarter turn "
        "about Z. Clutter requires an axis-aligned support surface."
    )
    if quarter_turns % 2:
        half_x, half_y = half_y, half_x

    region = ClutterRegion(
        min_x=center_x - half_x,
        min_y=center_y - half_y,
        max_x=center_x + half_x,
        max_y=center_y + half_y,
        floor_z=support_position[2] + float(support_bbox.top_surface_z[env_index]),
    )
    return region.scaled(spread) if spread != 1.0 else region


def _placed_bounding_box(
    asset: PlaceableAsset, layout: PlacementResult, bbox: AxisAlignedBoundingBox
) -> AxisAlignedBoundingBox:
    """Return object bounds rotated into the environment frame."""
    return refit_bbox_to_rotation(bbox, get_pose_from_layout(asset, layout).rotation_xyzw)


def resting_extents(
    asset: PlaceableAsset, layout: PlacementResult, bbox: AxisAlignedBoundingBox, env_index: int = 0
) -> tuple[float, float, float, float, float]:
    """Return rotated (min_x, min_y, max_x, max_y, min_z) offsets from the object origin."""
    placed = _placed_bounding_box(asset, layout, bbox)
    minimum, maximum = placed.min_point[env_index], placed.max_point[env_index]
    return (
        float(minimum[0]),
        float(minimum[1]),
        float(maximum[0]),
        float(maximum[1]),
        float(minimum[2]),
    )


def support_pose_from_layout(
    support: PlaceableAsset, layout: PlacementResult
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Return the support position and bounding-box rotation in the environment frame."""
    declared = support.get_initial_pose()

    if support in layout.positions:
        position = layout.positions[support]
    else:
        assert isinstance(declared, Pose), f"Clutter support '{support.name}' needs a solved position or a fixed Pose"
        position = declared.position_xyz

    if support in layout.rotations or support in layout.orientations:
        rotation = get_pose_from_layout(support, layout).rotation_xyzw
    elif support.has_relation(IsAnchor) or support not in layout.positions:
        assert isinstance(declared, Pose), f"Clutter support '{support.name}' needs a fixed Pose"
        rotation = declared.rotation_xyzw
    else:
        rotation = get_pose_from_layout(support, layout).rotation_xyzw

    # ObjectReference bounds already include the prim's rotation within its parent USD.
    # Only the parent's placement rotation remains to be applied to those bounds.
    from isaaclab_arena.assets.object_reference import ObjectReference

    if isinstance(support, ObjectReference):
        parent_pose = support.parent_asset.get_initial_pose()
        assert parent_pose is None or isinstance(parent_pose, Pose), "Clutter support parent needs a fixed Pose"
        rotation = parent_pose.rotation_xyzw if parent_pose is not None else (0.0, 0.0, 0.0, 1.0)

    return tuple(float(value) for value in position), tuple(float(value) for value in rotation)


def region_for_support(
    support: PlaceableAsset,
    layout: PlacementResult,
    bounding_boxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
    spread: float = 1.0,
    env_index: int = 0,
) -> ClutterRegion:
    """Return the support region for a layout and environment."""
    position, rotation = support_pose_from_layout(support, layout)
    return region_above_support(position, bounding_boxes[support], spread, env_index, rotation)


def occupied_footprints_in_region(
    region: ClutterRegion,
    layout: PlacementResult,
    bounding_boxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
    exclude: set[PlaceableAsset],
    env_index: int = 0,
) -> list[OccupiedFootprint]:
    """Return placed footprints intersecting the region above its floor."""
    footprints = []
    for asset, position in layout.positions.items():
        if asset in exclude or asset not in bounding_boxes:
            continue
        bbox = _placed_bounding_box(asset, layout, bounding_boxes[asset])
        minimum, maximum = bbox.min_point[env_index], bbox.max_point[env_index]
        top_z = float(position[2]) + float(maximum[2])
        if top_z <= region.floor_z:
            continue
        center = (
            float(position[0]) + float(maximum[0] + minimum[0]) * 0.5,
            float(position[1]) + float(maximum[1] + minimum[1]) * 0.5,
        )
        half_extents = (float(maximum[0] - minimum[0]) * 0.5, float(maximum[1] - minimum[1]) * 0.5)
        if (
            center[0] + half_extents[0] <= region.min_x
            or center[0] - half_extents[0] >= region.max_x
            or center[1] + half_extents[1] <= region.min_y
            or center[1] - half_extents[1] >= region.max_y
        ):
            continue
        footprints.append(OccupiedFootprint(center=center, half_extents=half_extents, top_z=top_z))
    return footprints


def _member_drop_params(member: PlaceableAsset) -> MemberDropParams:
    """Return a member's release parameters."""
    relation = next(r for r in member.get_relations() if isinstance(r, ClutteredOn))
    return MemberDropParams(clearance_m=relation.clearance_m, gap_m=relation.gap_m, random_yaw=relation.random_yaw)


def plan_group_drops_into_layout(
    layout: PlacementResult,
    group: ClutterGroup,
    member_bboxes: list[AxisAlignedBoundingBox],
    support_position: tuple[float, float, float],
    support_bbox: AxisAlignedBoundingBox,
    generator: torch.Generator,
    env_index: int = 0,
    occupied: list[OccupiedFootprint] | None = None,
    support_rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> None:
    """Write one group's release poses into the layout."""
    assert len(member_bboxes) == len(
        group.members
    ), f"Clutter group '{group.name}' has {len(group.members)} members but {len(member_bboxes)} bounding boxes."
    assert_group_parameters_agree(group)

    relation = group.relation
    region = region_above_support(support_position, support_bbox, relation.spread, env_index, support_rotation_xyzw)
    params = ClutterDropParams(drop_order=relation.drop_order)
    poses = compute_drop_poses(
        member_bboxes,
        region,
        params,
        generator,
        occupied=occupied,
        base_rotations_xyzw=[get_rotation_xyzw(member) for member in group.members],
        member_params=[_member_drop_params(member) for member in group.members],
    )

    for member, pose in zip(group.members, poses):
        layout.positions[member] = pose.position_xyz
        layout.rotations[member] = pose.rotation_xyzw


def plan_clutter_drops(
    layout: PlacementResult,
    groups: list[ClutterGroup],
    bounding_boxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
    generator: torch.Generator,
    env_index: int = 0,
) -> None:
    """Write release poses for each group, avoiding earlier groups."""
    for group in groups:
        support = group.support
        assert support in bounding_boxes, f"Clutter support '{support.name}' has no bounding box."
        missing = [member.name for member in group.members if member not in bounding_boxes]
        assert not missing, f"Clutter group '{group.name}' has members without bounding boxes: {missing}"
        support_position, support_rotation = support_pose_from_layout(support, layout)
        region = region_for_support(support, layout, bounding_boxes, group.relation.spread, env_index)
        occupied = occupied_footprints_in_region(
            region,
            layout,
            bounding_boxes,
            exclude={support, *group.members},
            env_index=env_index,
        )
        plan_group_drops_into_layout(
            layout=layout,
            group=group,
            member_bboxes=[bounding_boxes[member] for member in group.members],
            support_position=support_position,
            support_bbox=bounding_boxes[support],
            generator=generator,
            env_index=env_index,
            occupied=occupied,
            support_rotation_xyzw=support_rotation,
        )
