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

import math
import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.clutter_drop_poses import (
    ClutterDropParams,
    ClutterRegion,
    OccupiedFootprint,
    compute_drop_poses,
)
from isaaclab_arena.relations.clutter_groups import ClutterGroup, assert_group_parameters_agree
from isaaclab_arena.relations.placement_events import IDENTITY_ROTATION_XYZW, get_rotation_xyzw

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
    """Return the drop region covering a support's top face.

    Bounding boxes hold extents local to the object origin, so the support's world position
    offsets them and its yaw turns them. The floor of the region is the support's top surface.

    A region is axis-aligned, so a support turned off-axis cannot be covered exactly. Its
    enclosing box would be larger than the support itself and would drop clutter past the
    real edge, so an off-axis support instead gets the largest axis-aligned square that fits
    inside the footprint's inscribed circle. That is conservative in the safe direction and
    rotation-invariant. Quarter turns are exact, since they only swap the extents.

    Args:
        support_position: World position of the support, in the environment-local frame.
        support_bbox: The support's bounding box, batched per environment.
        spread: Fraction of the footprint to use, shrunk about the centre.
        env_index: Which environment's extents to read.
        support_rotation_xyzw: The support's world rotation. Must be a yaw-only rotation, since
            a tilted support has no single top surface height.
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
    local_centre_x = float(maximum[0] + minimum[0]) * 0.5
    local_centre_y = float(maximum[1] + minimum[1]) * 0.5

    # The footprint centre orbits the origin with the support.
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    centre_x = support_position[0] + local_centre_x * cos_yaw - local_centre_y * sin_yaw
    centre_y = support_position[1] + local_centre_x * sin_yaw + local_centre_y * cos_yaw

    quarter_turns = round(yaw / (math.pi / 2.0))
    if abs(yaw - quarter_turns * (math.pi / 2.0)) <= _QUARTER_TURN_TOLERANCE_RAD:
        if quarter_turns % 2:
            half_x, half_y = half_y, half_x
    else:
        half_x = half_y = min(half_x, half_y) / math.sqrt(2.0)

    region = ClutterRegion(
        min_x=centre_x - half_x,
        min_y=centre_y - half_y,
        max_x=centre_x + half_x,
        max_y=centre_y + half_y,
        floor_z=support_position[2] + float(support_bbox.top_surface_z[env_index]),
    )
    return region.scaled(spread) if spread != 1.0 else region


def _placed_bounding_box(
    asset: PlaceableAsset, layout: PlacementResult, bbox: AxisAlignedBoundingBox
) -> AxisAlignedBoundingBox:
    """Return an asset's bounding box refitted to the orientation the layout gave it.

    Layout boxes carry object geometry only; orientation is applied per candidate elsewhere,
    so a box read straight from the layout describes the asset unrotated.
    """
    rotation = layout.rotations.get(asset)
    if rotation is None:
        yaw = layout.orientations.get(asset)
        if yaw is None:
            return bbox
        rotation = (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))
    return bbox.rotated_by_quat(torch.tensor([rotation], dtype=torch.float32))


def support_pose_from_layout(
    support: PlaceableAsset, layout: PlacementResult
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Return a support's world position and rotation as resolved from a layout.

    An anchored support keeps its declared pose; a solved one is already in the layout. A
    support that is neither has no pose yet, which is a caller error rather than a default.
    """
    declared = support.get_initial_pose()
    position = layout.positions.get(support)
    if position is None:
        assert declared is not None, (
            f"Clutter support '{support.name}' has neither a solved position nor a declared "
            "initial pose, so there is nothing to pour onto."
        )
        position = declared.position_xyz

    rotation = layout.rotations.get(support)
    if rotation is None:
        yaw = layout.orientations.get(support)
        if yaw is not None:
            rotation = (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))
        elif declared is not None:
            rotation = declared.rotation_xyzw
        else:
            rotation = IDENTITY_ROTATION_XYZW

    return tuple(float(value) for value in position), tuple(float(value) for value in rotation)


def region_for_support(
    support: PlaceableAsset,
    layout: PlacementResult,
    bounding_boxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
    spread: float = 1.0,
    env_index: int = 0,
) -> ClutterRegion:
    """Return a support's drop region, resolved from the layout that placed it.

    Every consumer of a support's region goes through here. Deriving it twice invites the
    two derivations to disagree -- a pour that accounts for the support's yaw and a check
    that does not will reject exactly the layouts the pour got right.
    """
    position, rotation = support_pose_from_layout(support, layout)
    return region_above_support(position, bounding_boxes[support], spread, env_index, rotation)


def occupied_footprints_in_region(
    region: ClutterRegion,
    layout: PlacementResult,
    bounding_boxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
    exclude: set[PlaceableAsset],
    env_index: int = 0,
) -> list[OccupiedFootprint]:
    """Return the footprints of already-placed objects standing in ``region``.

    Nothing re-checks overlap once a pile has settled, and doing so on boxes would be
    meaningless anyway, so the pour has to avoid what is already there rather than discover
    the collision afterwards. Objects resting below the region's floor are ignored: they are
    under the surface being poured onto, not on it.

    Footprints are taken from each occupant's *placed* orientation. A solver-assigned yaw
    changes an elongated object's footprint substantially, and understating it would drop
    clutter into the very object this exists to avoid.
    """
    footprints = []
    for asset, position in layout.positions.items():
        if asset in exclude or asset not in bounding_boxes:
            continue
        bbox = _placed_bounding_box(asset, layout, bounding_boxes[asset])
        minimum, maximum = bbox.min_point[env_index], bbox.max_point[env_index]
        top_z = float(position[2]) + float(maximum[2])
        if top_z <= region.floor_z:
            continue
        centre = (
            float(position[0]) + float(maximum[0] + minimum[0]) * 0.5,
            float(position[1]) + float(maximum[1] + minimum[1]) * 0.5,
        )
        half_extents = (float(maximum[0] - minimum[0]) * 0.5, float(maximum[1] - minimum[1]) * 0.5)
        if (
            centre[0] + half_extents[0] <= region.min_x
            or centre[0] - half_extents[0] >= region.max_x
            or centre[1] + half_extents[1] <= region.min_y
            or centre[1] - half_extents[1] >= region.max_y
        ):
            continue
        footprints.append(OccupiedFootprint(centre=centre, half_extents=half_extents, top_z=top_z))
    return footprints


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
    region = region_above_support(support_position, support_bbox, relation.spread, env_index, support_rotation_xyzw)
    params = ClutterDropParams(
        clearance_m=relation.clearance_m,
        gap_m=relation.gap_m,
        random_yaw=relation.random_yaw,
    )
    poses = compute_drop_poses(
        member_bboxes,
        region,
        params,
        generator,
        occupied=occupied,
        base_rotations_xyzw=[get_rotation_xyzw(member) for member in group.members],
    )

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
        support_position, support_rotation = support_pose_from_layout(support, layout)
        region = region_for_support(support, layout, bounding_boxes, group.relation.spread, env_index)
        # Members of earlier groups are already in the layout and must be avoided too.
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
