# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Noninterpenetrating release poses for physics-settled clutter."""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass
from enum import Enum

from isaaclab.utils.math import quat_mul

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.random import get_random_rotation


class XySampling(str, Enum):
    """How XY positions are drawn from the region."""

    UNIFORM = "uniform"
    """Sample each object independently. Objects may co-locate and occlude one another."""

    GRID_CELLS = "grid_cells"
    """Jitter shuffled grid centers by each object's footprint size, clipped to the region."""


class DropOrder(str, Enum):
    """Release order within a clutter group."""

    AS_LISTED = "as_listed"
    """Keep the caller's order."""

    FLATTEST_FIRST = "flattest_first"
    """Release objects in ascending base-rotated height."""

    SHUFFLE = "shuffle"
    """Random member order."""


@dataclass(frozen=True)
class ClutterRegion:
    """Axis-aligned support footprint and surface height, in the environment frame E."""

    min_x: float
    """Minimum X in E, in metres."""

    min_y: float
    """Minimum Y in E, in metres."""

    max_x: float
    """Maximum X in E, in metres."""

    max_y: float
    """Maximum Y in E, in metres."""

    floor_z: float
    """Z of the surface objects are dropped onto."""

    def __post_init__(self) -> None:
        assert self.max_x > self.min_x, f"region needs max_x > min_x, got {self.min_x}, {self.max_x}"
        assert self.max_y > self.min_y, f"region needs max_y > min_y, got {self.min_y}, {self.max_y}"

    def scaled(self, factor: float) -> ClutterRegion:
        """Return this region scaled about its center. Lower factors heap the pile tighter."""
        assert factor > 0.0, f"scale factor must be positive, got {factor}"
        cx, cy = (self.min_x + self.max_x) * 0.5, (self.min_y + self.max_y) * 0.5
        half_x = (self.max_x - self.min_x) * 0.5 * factor
        half_y = (self.max_y - self.min_y) * 0.5 * factor
        return ClutterRegion(cx - half_x, cy - half_y, cx + half_x, cy + half_y, self.floor_z)


@dataclass(frozen=True)
class ClutterDropParams:
    """Release sampling parameters shared by a clutter group."""

    xy_sampling: XySampling = XySampling.GRID_CELLS
    """XY sampling distribution."""

    drop_order: DropOrder = DropOrder.AS_LISTED
    """Member release order."""

    max_yaw_attempts: int = 8
    """Maximum yaw samples per object."""

    def __post_init__(self) -> None:
        assert self.max_yaw_attempts >= 1, "max_yaw_attempts must be positive"


@dataclass(frozen=True)
class MemberDropParams:
    """Release clearance and yaw parameters for one object."""

    clearance_m: float = 0.01
    """Gap between this object's lowest point and the surface it is dropped onto."""

    gap_m: float = 0.03
    """Extra vertical gap when this object must clear one already placed below it."""

    random_yaw: bool = True
    """Whether to sample world-Z yaw on top of the base rotation."""


@dataclass(frozen=True)
class OccupiedFootprint:
    """An occupied XY footprint in environment frame E. Lengths are in metres."""

    center: tuple[float, float]
    """Footprint center in E, shape (2,)."""

    half_extents: tuple[float, float]
    """XY half-extents, shape (2,)."""

    top_z: float
    """Height of its highest point, which clutter above it must clear."""


@dataclass(frozen=True)
class DropPose:
    """Object pose in environment frame E before settling. Quaternion order is (x, y, z, w)."""

    position_xyz: tuple[float, float, float]
    """Object position in E, shape (3,), in metres."""

    rotation_xyzw: tuple[float, float, float, float]
    """Object-to-E quaternion, shape (4,)."""

    drop_index: int
    """Zero-based release index."""


@dataclass(frozen=True)
class _Footprint:
    """XY half-extents and center offsets from the object origin, in metres."""

    half_x: float
    """X half-extent."""

    half_y: float
    """Y half-extent."""

    offset_x: float
    """X center offset."""

    offset_y: float
    """Y center offset."""

    def center_at(self, x: float, y: float) -> tuple[float, float]:
        """Return the footprint center when the origin is placed at ``(x, y)``."""
        return x + self.offset_x, y + self.offset_y


def _footprint_of(bbox: AxisAlignedBoundingBox) -> _Footprint:
    """Half-extents and origin offset of a bounding box's XY footprint."""
    minimum, maximum = bbox.min_point[0], bbox.max_point[0]
    return _Footprint(
        half_x=float(maximum[0] - minimum[0]) * 0.5,
        half_y=float(maximum[1] - minimum[1]) * 0.5,
        offset_x=float(maximum[0] + minimum[0]) * 0.5,
        offset_y=float(maximum[1] + minimum[1]) * 0.5,
    )


def refit_bbox_to_rotation(
    bbox: AxisAlignedBoundingBox, rotation_xyzw: tuple[float, float, float, float]
) -> AxisAlignedBoundingBox:
    """Return the axis-aligned bounds after rotating about the object origin."""
    if rotation_xyzw == (0.0, 0.0, 0.0, 1.0):
        return bbox
    return bbox.rotated_by_quat(torch.tensor([rotation_xyzw], dtype=torch.float32))


def _resolve_order(
    bounding_boxes: list[AxisAlignedBoundingBox],
    drop_order: DropOrder,
    generator: torch.Generator | None,
    base_rotations_xyzw: list[tuple[float, float, float, float]],
) -> list[int]:
    """Return object indices in release order, using base-rotated heights."""
    indices = list(range(len(bounding_boxes)))
    if drop_order is DropOrder.AS_LISTED:
        return indices
    if drop_order is DropOrder.FLATTEST_FIRST:
        heights = [
            float(refit_bbox_to_rotation(bbox, rotation).size[0][2])
            for bbox, rotation in zip(bounding_boxes, base_rotations_xyzw)
        ]
        return sorted(indices, key=lambda i: heights[i])
    permutation = torch.randperm(len(indices), generator=generator)
    return [indices[int(i)] for i in permutation]


def _grid_cell_centers(
    count: int, region: ClutterRegion, generator: torch.Generator | None
) -> list[tuple[float, float]]:
    """Return shuffled cell centers on a grid matching the region aspect ratio."""
    width, depth = region.max_x - region.min_x, region.max_y - region.min_y
    num_cols = max(1, min(count, round(math.sqrt(count * width / depth))))
    num_rows = math.ceil(count / num_cols)

    centers = []
    for i in range(count):
        row, col = divmod(i, num_cols)
        centers.append((
            region.min_x + width * (col + 0.5) / num_cols,
            region.min_y + depth * (row + 0.5) / num_rows,
        ))
    permutation = torch.randperm(count, generator=generator)
    return [centers[int(i)] for i in permutation]


def _footprint_fits(footprint: _Footprint, region: ClutterRegion) -> bool:
    """Whether a footprint of these half-extents can sit wholly inside ``region``."""
    return (
        region.min_x + footprint.half_x <= region.max_x - footprint.half_x
        and region.min_y + footprint.half_y <= region.max_y - footprint.half_y
    )


def _sample_orientation_that_fits(
    bbox: AxisAlignedBoundingBox,
    region: ClutterRegion,
    params: ClutterDropParams,
    member: MemberDropParams,
    generator: torch.Generator | None,
    base_rotation_xyzw: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float, float], AxisAlignedBoundingBox, _Footprint]:
    """Sample a release yaw whose full-rotation bounds fit inside the region."""
    attempts = params.max_yaw_attempts if member.random_yaw else 1
    for _ in range(attempts):
        rotation = base_rotation_xyzw
        if member.random_yaw:
            yaw = get_random_rotation(generator)
            yaw_rotation = torch.tensor((0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))
            # Left multiplication applies the extra rotation about world Z.
            rotation = tuple(quat_mul(yaw_rotation, torch.tensor(base_rotation_xyzw, dtype=torch.float32)).tolist())
        rotated = refit_bbox_to_rotation(bbox, rotation)
        footprint = _footprint_of(rotated)
        if _footprint_fits(footprint, region):
            return rotation, rotated, footprint
    raise AssertionError(
        f"object footprint {2 * footprint.half_x:.3f}x{2 * footprint.half_y:.3f} m does not fit in region "
        f"{region.max_x - region.min_x:.3f}x{region.max_y - region.min_y:.3f} m "
        f"after {attempts} orientation attempt(s)"
    )


def _sample_xy(
    center: tuple[float, float] | None,
    footprint: _Footprint,
    region: ClutterRegion,
    generator: torch.Generator | None,
) -> tuple[float, float]:
    """Sample an object origin that keeps its offset footprint inside the region."""
    min_x = region.min_x + footprint.half_x - footprint.offset_x
    max_x = region.max_x - footprint.half_x - footprint.offset_x
    min_y = region.min_y + footprint.half_y - footprint.offset_y
    max_y = region.max_y - footprint.half_y - footprint.offset_y
    unit = torch.rand(2, generator=generator)
    if center is None:
        x = min_x + float(unit[0]) * (max_x - min_x)
        y = min_y + float(unit[1]) * (max_y - min_y)
    else:
        x = center[0] - footprint.offset_x + (float(unit[0]) - 0.5) * footprint.half_x
        y = center[1] - footprint.offset_y + (float(unit[1]) - 0.5) * footprint.half_y
    return min(max(x, min_x), max_x), min(max(y, min_y), max_y)


def _footprints_overlap(
    a_center: tuple[float, float],
    a_half: tuple[float, float],
    b_center: tuple[float, float],
    b_half: tuple[float, float],
) -> bool:
    """Whether two axis-aligned XY footprints overlap. Centers, not origins."""
    return (
        abs(a_center[0] - b_center[0]) < a_half[0] + b_half[0]
        and abs(a_center[1] - b_center[1]) < a_half[1] + b_half[1]
    )


def compute_drop_poses(
    bounding_boxes: list[AxisAlignedBoundingBox],
    region: ClutterRegion,
    params: ClutterDropParams | None = None,
    generator: torch.Generator | None = None,
    occupied: list[OccupiedFootprint] | None = None,
    base_rotations_xyzw: list[tuple[float, float, float, float]] | None = None,
    member_params: list[MemberDropParams] | None = None,
) -> list[DropPose]:
    """Return noninterpenetrating release poses for one clutter group.

    Args:
        bounding_boxes: N object-local boxes, each with min/max shape (1, 3).
        region: Support footprint in the environment frame E.
        params: Group sampling parameters.
        generator: Random generator.
        occupied: Existing footprints in frame E.
        base_rotations_xyzw: N authored quaternions, each shape (4,). Defaults to identity.
        member_params: N per-object sampling parameters, in input order.

    Returns:
        N drop poses in input order.
    """
    assert bounding_boxes, "compute_drop_poses needs at least one bounding box"
    for i, bbox in enumerate(bounding_boxes):
        assert bbox.num_envs == 1, f"bounding_boxes[{i}] must be single-env (N=1), got N={bbox.num_envs}"
    params = params or ClutterDropParams()
    if member_params is None:
        member_params = [MemberDropParams()] * len(bounding_boxes)
    assert len(member_params) == len(
        bounding_boxes
    ), f"compute_drop_poses got {len(member_params)} member params for {len(bounding_boxes)} objects"
    if base_rotations_xyzw is None:
        base_rotations_xyzw = [(0.0, 0.0, 0.0, 1.0) for _ in bounding_boxes]
    assert len(base_rotations_xyzw) == len(
        bounding_boxes
    ), "base_rotations_xyzw must contain one rotation per bounding box"

    order = _resolve_order(bounding_boxes, params.drop_order, generator, base_rotations_xyzw)
    centers = (
        _grid_cell_centers(len(order), region, generator)
        if params.xy_sampling is XySampling.GRID_CELLS
        else [None] * len(order)
    )

    poses: dict[int, DropPose] = {}
    placed: list[tuple[tuple[float, float], tuple[float, float], float]] = [
        (item.center, item.half_extents, item.top_z) for item in (occupied or [])
    ]

    for drop_index, (object_index, center) in enumerate(zip(order, centers)):
        rotation, rotated, footprint = _sample_orientation_that_fits(
            bounding_boxes[object_index],
            region,
            params,
            member_params[object_index],
            generator,
            base_rotations_xyzw[object_index],
        )
        x, y = _sample_xy(center, footprint, region, generator)
        footprint_center = footprint.center_at(x, y)
        half_extents = (footprint.half_x, footprint.half_y)

        member = member_params[object_index]
        support_z = region.floor_z + member.clearance_m
        for other_center, other_half, other_top_z in placed:
            if _footprints_overlap(footprint_center, half_extents, other_center, other_half):
                support_z = max(support_z, other_top_z + member.gap_m)

        z = support_z - float(rotated.bottom_surface_z[0])
        placed.append((footprint_center, half_extents, z + float(rotated.top_surface_z[0])))
        poses[object_index] = DropPose(position_xyz=(x, y, z), rotation_xyzw=rotation, drop_index=drop_index)

    return [poses[index] for index in range(len(bounding_boxes))]
