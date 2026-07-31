# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Planning of clutter drop poses into solved layouts.

A clutter pile cannot be optimised into place, so its members are instead dropped above
their support and left to settle. This module plans where each member is released; the
settling and the capture of resting poses are done by the existing in-sim validation pass.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.clutter_drop_poses import ClutterDropParams, ClutterRegion, compute_drop_poses
from isaaclab_arena.relations.clutter_groups import ClutterGroup, assert_group_parameters_agree

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


def region_above_support(
    support_position: tuple[float, float, float],
    support_bbox: AxisAlignedBoundingBox,
    spread: float = 1.0,
    env_index: int = 0,
) -> ClutterRegion:
    """Return the drop region covering a support's top face.

    Bounding boxes hold extents local to the object origin, so the support's world position
    offsets them. The floor of the region is the support's top surface.

    Args:
        support_position: World position of the support, in the environment-local frame.
        support_bbox: The support's bounding box, batched per environment.
        spread: Fraction of the footprint to use, shrunk about the centre.
        env_index: Which environment's extents to read.
    """
    minimum = support_bbox.min_point[env_index]
    maximum = support_bbox.max_point[env_index]
    region = ClutterRegion(
        min_x=support_position[0] + float(minimum[0]),
        min_y=support_position[1] + float(minimum[1]),
        max_x=support_position[0] + float(maximum[0]),
        max_y=support_position[1] + float(maximum[1]),
        floor_z=support_position[2] + float(support_bbox.top_surface_z[env_index]),
    )
    return region.scaled(spread) if spread != 1.0 else region


def plan_group_drops_into_layout(
    layout: PlacementResult,
    group: ClutterGroup,
    member_bboxes: list[AxisAlignedBoundingBox],
    support_position: tuple[float, float, float],
    support_bbox: AxisAlignedBoundingBox,
    generator: torch.Generator,
    env_index: int = 0,
) -> None:
    """Write one group's drop poses into a solved layout.

    The layout then carries release poses rather than solved ones, so writing it to sim and
    stepping physics produces the pile. Resting poses are read back by the validation pass.

    Args:
        layout: The layout to write drop poses into.
        group: The pile being poured.
        member_bboxes: Bounding box per member, in ``group.members`` order.
        support_position: World position of the support, in the environment-local frame.
        support_bbox: The support's bounding box.
        generator: Seeded RNG, so a given seed reproduces a given pile.
        env_index: Which environment's extents to read.
    """
    assert len(member_bboxes) == len(
        group.members
    ), f"Clutter group '{group.name}' has {len(group.members)} members but {len(member_bboxes)} bounding boxes."
    assert_group_parameters_agree(group)

    relation = group.relation
    region = region_above_support(support_position, support_bbox, relation.spread, env_index)
    params = ClutterDropParams(
        clearance_m=relation.clearance_m,
        gap_m=relation.gap_m,
        random_yaw=relation.random_yaw,
    )
    poses = compute_drop_poses(member_bboxes, region, params, generator)

    for member, pose in zip(group.members, poses):
        layout.positions[member] = pose.position
        layout.rotations[member] = pose.rotation_xyzw


def plan_clutter_drops(
    layout: PlacementResult,
    groups: list[ClutterGroup],
    bounding_boxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
    generator: torch.Generator,
    env_index: int = 0,
) -> None:
    """Write every group's drop poses into a solved layout.

    Args:
        layout: The layout to write drop poses into.
        groups: The piles to pour, in the order returned by group resolution.
        bounding_boxes: Bounding box per asset, covering every support and member.
        generator: Seeded RNG shared across groups.
        env_index: Which environment's extents to read.
    """
    for group in groups:
        support = group.support
        assert support in bounding_boxes, f"Clutter support '{support.name}' has no bounding box."
        missing = [member.name for member in group.members if member not in bounding_boxes]
        assert not missing, f"Clutter group '{group.name}' has members without bounding boxes: {missing}"
        # An anchored support keeps its declared pose; a solved one is already in the layout.
        support_position = layout.positions.get(support) or support.get_initial_pose().position_xyz
        plan_group_drops_into_layout(
            layout=layout,
            group=group,
            member_bboxes=[bounding_boxes[member] for member in group.members],
            support_position=tuple(float(value) for value in support_position),
            support_bbox=bounding_boxes[support],
            generator=generator,
            env_index=env_index,
        )
