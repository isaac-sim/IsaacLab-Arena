# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Drop-pose generation for clutter placement.

A pile cannot be produced by the relation solver: ``On`` requires each child's whole
footprint to rest on one Z plane and a global pairwise no-overlap loss forbids contact,
whereas a pile is defined by objects touching and resting on one another. Clutter is
therefore placed by dropping objects into a region and letting the simulator settle them.

This module computes the *drop* poses only: the pre-settle layout. It is pure geometry
with no simulator dependency, so it is cheap and directly testable. Poses are guaranteed
free of mutual penetration at spawn, which lets them double as the scene's spawn poses.

The two properties that make the layout safe:

* Each object's sampled XY is constrained so its yaw-rotated footprint lies inside the
  region, rather than only its origin.
* Objects are lifted only over the footprints they actually overlap (a per-column ladder),
  so an object with a clear column starts just above the floor instead of above the whole
  pile. This keeps drop heights: and therefore impact energy and scatter: small.
"""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass
from enum import Enum

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.random import get_random_rotation
from isaaclab_arena.utils.yaw import rotate_quat_by_yaw


class XySampling(str, Enum):
    """How XY positions are drawn from the region."""

    UNIFORM = "uniform"
    """Sample each object independently. Objects may co-locate and occlude one another."""

    GRID_CELLS = "grid_cells"
    """Give each object its own jittered cell, so the group spreads across the region."""


class DropOrder(str, Enum):
    """Order objects are planned in, which decides which of them end up underneath."""

    AS_LISTED = "as_listed"
    """Keep the caller's order."""

    FLATTEST_FIRST = "flattest_first"
    """Shortest object first. Flat objects reach the floor before the region gets lumpy,
    which keeps them lying flat instead of coming to rest on an edge."""

    SHUFFLE = "shuffle"
    """Randomise, so no object is systematically at the bottom across layouts."""


@dataclass(frozen=True)
class ClutterRegion:
    """Axis-aligned XY region with a floor, in the frame drop poses are returned in."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float
    floor_z: float
    """Z of the surface objects are dropped onto."""

    def __post_init__(self) -> None:
        assert self.max_x > self.min_x, f"region needs max_x > min_x, got {self.min_x}, {self.max_x}"
        assert self.max_y > self.min_y, f"region needs max_y > min_y, got {self.min_y}, {self.max_y}"

    def scaled(self, factor: float) -> ClutterRegion:
        """Return this region scaled about its centre. Lower factors heap the pile tighter."""
        assert factor > 0.0, f"scale factor must be positive, got {factor}"
        cx, cy = (self.min_x + self.max_x) * 0.5, (self.min_y + self.max_y) * 0.5
        half_x = (self.max_x - self.min_x) * 0.5 * factor
        half_y = (self.max_y - self.min_y) * 0.5 * factor
        return ClutterRegion(cx - half_x, cy - half_y, cx + half_x, cy + half_y, self.floor_z)


@dataclass(frozen=True)
class ClutterDropParams:
    """Tuning that describes a pile as a whole."""

    xy_sampling: XySampling = XySampling.GRID_CELLS
    drop_order: DropOrder = DropOrder.AS_LISTED

    max_yaw_attempts: int = 8
    """How many yaws to try before declaring an object unplaceable. A long object turned
    diagonally needs much more room than the same object axis-aligned, so an unlucky draw
    should be resampled rather than failing the whole layout."""


@dataclass(frozen=True)
class MemberDropParams:
    """Tuning that describes one member of a pile.

    These read per member rather than per pile because each answers a question about a single
    object: how far above its support it starts, how far it clears whatever it lands on, and
    whether it is turned. A heavier tray can take more clearance than a mug beside it.
    """

    clearance_m: float = 0.01
    """Gap between this object's lowest point and the surface it is dropped onto."""

    gap_m: float = 0.03
    """Extra vertical gap when this object must clear one already placed below it."""

    random_yaw: bool = True
    """Sample a random Z-yaw for this object. Disable for a reproducible axis-aligned drop."""


@dataclass(frozen=True)
class OccupiedFootprint:
    """Something already standing in the region that clutter must not be dropped into."""

    centre: tuple[float, float]
    """XY centre of the footprint, in the region's frame."""

    half_extents: tuple[float, float]
    """Half-width and half-depth of the footprint."""

    top_z: float
    """Height of its highest point, which clutter above it must clear."""


@dataclass(frozen=True)
class DropPose:
    """A pre-settle pose for one object."""

    position: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    drop_index: int
    """Position in the planning sequence; 0 is placed first and tends to end up lowest."""


@dataclass(frozen=True)
class _Footprint:
    """A bounding box's XY footprint, described relative to the object origin.

    Local boxes are not centred on the origin, USD pivots sit wherever the asset author
    put them: so the footprint's own centre has to be carried separately from its size.
    Treating the origin as the centre would let an object hang off the region by its offset.
    """

    half_x: float
    half_y: float
    offset_x: float
    offset_y: float
    """XY of the footprint centre in the object's local frame."""

    def centre_at(self, x: float, y: float) -> tuple[float, float]:
        """Return the footprint centre when the origin is placed at ``(x, y)``."""
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


def _rotated_by(
    bbox: AxisAlignedBoundingBox, rotation_xyzw: tuple[float, float, float, float]
) -> AxisAlignedBoundingBox:
    """Return the box refitted to a rotation, or the box itself when there is nothing to turn."""
    if rotation_xyzw == (0.0, 0.0, 0.0, 1.0):
        return bbox
    return bbox.rotated_by_quat(torch.tensor([rotation_xyzw], dtype=torch.float32))


def _resolve_order(
    bounding_boxes: list[AxisAlignedBoundingBox],
    drop_order: DropOrder,
    generator: torch.Generator | None,
    base_rotations_xyzw: list[tuple[float, float, float, float]],
) -> list[int]:
    """Return indices into ``bounding_boxes`` in the order they should be dropped.

    Heights are measured after the authored base rotation, since that is the pose the object is
    released in: a thin plate stood on its edge is tall, whatever its local extents say. The yaw
    sampled later turns it about Z and cannot change its height, so ranking here is safe.
    """
    indices = list(range(len(bounding_boxes)))
    if drop_order is DropOrder.AS_LISTED:
        return indices
    if drop_order is DropOrder.FLATTEST_FIRST:
        heights = [
            float(_rotated_by(bbox, rotation).size[0][2]) for bbox, rotation in zip(bounding_boxes, base_rotations_xyzw)
        ]
        return sorted(indices, key=lambda i: heights[i])
    permutation = torch.randperm(len(indices), generator=generator)
    return [indices[int(i)] for i in permutation]


def _grid_cell_centres(
    count: int, region: ClutterRegion, generator: torch.Generator | None
) -> list[tuple[float, float]]:
    """One cell centre per object, tiling the region and shuffled.

    The grid is chosen to keep cells as square as the region allows, so neither axis is
    over-subdivided. Shuffling stops objects listed together from landing side by side.
    """
    width, depth = region.max_x - region.min_x, region.max_y - region.min_y
    num_cols = max(1, min(count, round(math.sqrt(count * width / depth)))) if depth > 0.0 else count
    num_rows = math.ceil(count / num_cols)

    centres = []
    for i in range(count):
        row, col = divmod(i, num_cols)
        centres.append((
            region.min_x + width * (col + 0.5) / num_cols,
            region.min_y + depth * (row + 0.5) / num_rows,
        ))
    permutation = torch.randperm(count, generator=generator)
    return [centres[int(i)] for i in permutation]


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
    """Sample a yaw whose rotated footprint fits the region, retrying unlucky draws.

    Yaw changes an elongated object's footprint substantially: a long object turned
    diagonally can need far more room than the same object axis-aligned: so a single
    unlucky draw is not evidence that the object cannot be placed. Retry before giving up,
    and only fail when no sampled orientation fits.
    """
    attempts = params.max_yaw_attempts if member.random_yaw else 1
    widest: _Footprint | None = None
    for _ in range(attempts):
        yaw = get_random_rotation(generator) if member.random_yaw else 0.0
        rotation = rotate_quat_by_yaw(base_rotation_xyzw, yaw)
        # Refit the box to the sampled yaw so footprint and height match the placed object.
        rotated = _rotated_by(bbox, rotation)
        footprint = _footprint_of(rotated)
        if _footprint_fits(footprint, region):
            return rotation, rotated, footprint
        # Report the draw that came closest to needing the whole region, not the first miss.
        if widest is None or max(footprint.half_x, footprint.half_y) > max(widest.half_x, widest.half_y):
            widest = footprint

    half_x = widest.half_x if widest is not None else 0.0
    half_y = widest.half_y if widest is not None else 0.0
    raise AssertionError(
        f"object footprint {2 * half_x:.3f}x{2 * half_y:.3f} m does not fit in region "
        f"{region.max_x - region.min_x:.3f}x{region.max_y - region.min_y:.3f} m "
        f"after {attempts} orientation attempt(s)"
    )


def _sample_xy(
    centre: tuple[float, float] | None,
    footprint: _Footprint,
    region: ClutterRegion,
    generator: torch.Generator | None,
) -> tuple[float, float]:
    """Sample an object origin whose footprint stays inside ``region``.

    The admissible box is the region inset by the object's own half-extents and then shifted
    by its origin offset, so the whole footprint lands inside rather than just the origin.
    When a cell centre is given it names where the footprint should sit, and the origin is
    offset accordingly before jittering and clamping.
    """
    min_x = region.min_x + footprint.half_x - footprint.offset_x
    max_x = region.max_x - footprint.half_x - footprint.offset_x
    min_y = region.min_y + footprint.half_y - footprint.offset_y
    max_y = region.max_y - footprint.half_y - footprint.offset_y
    unit = torch.rand(2, generator=generator)
    if centre is None:
        x = min_x + float(unit[0]) * (max_x - min_x)
        y = min_y + float(unit[1]) * (max_y - min_y)
    else:
        # Jitter by half a footprint so cell-mates stay distinguishable without escaping.
        x = centre[0] - footprint.offset_x + (float(unit[0]) - 0.5) * footprint.half_x
        y = centre[1] - footprint.offset_y + (float(unit[1]) - 0.5) * footprint.half_y
    return min(max(x, min_x), max_x), min(max(y, min_y), max_y)


def _footprints_overlap(
    a_centre: tuple[float, float],
    a_half: tuple[float, float],
    b_centre: tuple[float, float],
    b_half: tuple[float, float],
) -> bool:
    """Whether two axis-aligned XY footprints overlap. Centres, not origins."""
    return (
        abs(a_centre[0] - b_centre[0]) < a_half[0] + b_half[0]
        and abs(a_centre[1] - b_centre[1]) < a_half[1] + b_half[1]
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
    """Compute pre-settle drop poses for one pile.

    Objects are placed so that no two overlap in 3D: an object whose footprint is clear
    starts just above the floor, and one that would sit over an already-placed object is
    lifted to clear it. The simulator turns this layout into a pile by settling it.

    Args:
        bounding_boxes: Object-local bounding box per object, single-environment (``N=1``).
        region: Where objects may land, in the frame the returned poses use.
        params: Tuning; defaults to :class:`ClutterDropParams`.
        generator: Seeded RNG for reproducible layouts.
        occupied: Footprints already standing in the region, such as objects the solver placed
            on the same surface. Clutter is released above them rather than inside them.
        member_params: Per-object tuning, in ``bounding_boxes`` order. Defaults to one
            :class:`MemberDropParams` for every object.
        base_rotations_xyzw: Source-authored rotation per object. Sampled yaw is composed on
            top of each rotation. Defaults to identity for every object.

    Returns:
        One :class:`DropPose` per input object, in input order.
    """
    assert bounding_boxes, "compute_drop_poses needs at least one bounding box"
    for i, bbox in enumerate(bounding_boxes):
        assert bbox.num_envs == 1, f"bounding_boxes[{i}] must be single-env (N=1), got N={bbox.num_envs}"
    params = params or ClutterDropParams()
    member_params = member_params or [MemberDropParams()] * len(bounding_boxes)
    assert len(member_params) == len(
        bounding_boxes
    ), f"compute_drop_poses got {len(member_params)} member params for {len(bounding_boxes)} objects"
    if base_rotations_xyzw is None:
        base_rotations_xyzw = [(0.0, 0.0, 0.0, 1.0) for _ in bounding_boxes]
    assert len(base_rotations_xyzw) == len(
        bounding_boxes
    ), "base_rotations_xyzw must contain one rotation per bounding box"

    usable = region
    order = _resolve_order(bounding_boxes, params.drop_order, generator, base_rotations_xyzw)
    centres = (
        _grid_cell_centres(len(order), usable, generator)
        if params.xy_sampling is XySampling.GRID_CELLS
        else [None] * len(order)
    )

    poses: list[DropPose | None] = [None] * len(bounding_boxes)
    # Seed the ladder with what is already standing here, so the first drop clears it too.
    placed: list[tuple[tuple[float, float], tuple[float, float], float]] = [
        (item.centre, item.half_extents, item.top_z) for item in (occupied or [])
    ]

    for drop_index, (object_index, centre) in enumerate(zip(order, centres)):
        rotation, rotated, footprint = _sample_orientation_that_fits(
            bounding_boxes[object_index],
            usable,
            params,
            member_params[object_index],
            generator,
            base_rotations_xyzw[object_index],
        )
        x, y = _sample_xy(centre, footprint, usable, generator)
        footprint_centre = footprint.centre_at(x, y)
        half_extents = (footprint.half_x, footprint.half_y)

        # Lift only over the footprints actually overlapped, not over everything placed so far.
        member = member_params[object_index]
        support_z = region.floor_z + member.clearance_m
        for other_centre, other_half, other_top_z in placed:
            if _footprints_overlap(footprint_centre, half_extents, other_centre, other_half):
                support_z = max(support_z, other_top_z + member.gap_m)

        # Offset the origin so the object's lowest point sits at support_z.
        z = support_z - float(rotated.bottom_surface_z[0])
        placed.append((footprint_centre, half_extents, z + float(rotated.top_surface_z[0])))
        poses[object_index] = DropPose(position=(x, y, z), rotation_xyzw=rotation, drop_index=drop_index)

    assert all(pose is not None for pose in poses), "every object must receive a drop pose"
    return [pose for pose in poses if pose is not None]
