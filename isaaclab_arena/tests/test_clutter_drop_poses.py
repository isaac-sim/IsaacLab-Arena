# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for clutter drop-pose generation.

Pure geometry, so these run without a SimulationApp.
"""

import math
import torch

import pytest

from isaaclab_arena.relations.clutter_drop_poses import (
    ClutterDropParams,
    ClutterRegion,
    DropOrder,
    DropPose,
    XySampling,
    compute_drop_poses,
)
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

REGION = ClutterRegion(min_x=-0.18, min_y=-0.23, max_x=0.18, max_y=0.23, floor_z=0.8)


def make_bbox(size_x: float, size_y: float, size_z: float, origin_offset_z: float = 0.0):
    """Object-local bbox of the given size, optionally shifted so the origin is not centred."""
    half_x, half_y, half_z = size_x / 2.0, size_y / 2.0, size_z / 2.0
    return AxisAlignedBoundingBox(
        min_point=(-half_x, -half_y, -half_z + origin_offset_z),
        max_point=(half_x, half_y, half_z + origin_offset_z),
    )


def seeded(seed: int = 0) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def footprint_half_extents(bbox, rotation_xyzw):
    rotated = bbox.rotated_by_quat(torch.tensor([rotation_xyzw], dtype=torch.float32))
    size = rotated.size[0]
    return float(size[0]) / 2.0, float(size[1]) / 2.0


def assert_no_penetration(bboxes: list, poses: list[DropPose]) -> None:
    """No two placed objects may overlap in all three axes simultaneously."""
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            bbox_i = bboxes[i].rotated_by_quat(torch.tensor([poses[i].rotation_xyzw], dtype=torch.float32))
            bbox_j = bboxes[j].rotated_by_quat(torch.tensor([poses[j].rotation_xyzw], dtype=torch.float32))
            overlaps = []
            for axis in range(3):
                lo_i = poses[i].position[axis] + float(bbox_i.min_point[0][axis])
                hi_i = poses[i].position[axis] + float(bbox_i.max_point[0][axis])
                lo_j = poses[j].position[axis] + float(bbox_j.min_point[0][axis])
                hi_j = poses[j].position[axis] + float(bbox_j.max_point[0][axis])
                overlaps.append(lo_i < hi_j and lo_j < hi_i)
            assert not all(overlaps), f"objects {i} and {j} interpenetrate at spawn"


# --------------------------------------------------------------------------- basics


def test_returns_one_pose_per_object_in_input_order():
    bboxes = [make_bbox(0.05, 0.05, 0.04), make_bbox(0.1, 0.03, 0.02), make_bbox(0.02, 0.02, 0.08)]
    poses = compute_drop_poses(bboxes, REGION, generator=seeded())

    assert len(poses) == len(bboxes)
    assert sorted(pose.drop_index for pose in poses) == [0, 1, 2]


def test_rejects_empty_input():
    with pytest.raises(AssertionError):
        compute_drop_poses([], REGION, generator=seeded())


def test_rejects_batched_bounding_box():
    batched = AxisAlignedBoundingBox(
        min_point=torch.tensor([[-0.1, -0.1, -0.1], [-0.1, -0.1, -0.1]]),
        max_point=torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]),
    )
    with pytest.raises(AssertionError, match="single-env"):
        compute_drop_poses([batched], REGION, generator=seeded())


# --------------------------------------------------------------------- containment


@pytest.mark.parametrize("sampling", [XySampling.UNIFORM, XySampling.GRID_CELLS])
def test_rotated_footprint_stays_inside_region(sampling):
    """The whole footprint must land inside the region, not merely the object origin."""
    bboxes = [make_bbox(0.12, 0.04, 0.03) for _ in range(6)]
    params = ClutterDropParams(xy_sampling=sampling)
    poses = compute_drop_poses(bboxes, REGION, params, generator=seeded(3))

    for bbox, pose in zip(bboxes, poses):
        half_x, half_y = footprint_half_extents(bbox, pose.rotation_xyzw)
        assert pose.position[0] - half_x >= REGION.min_x - 1e-6
        assert pose.position[0] + half_x <= REGION.max_x + 1e-6
        assert pose.position[1] - half_y >= REGION.min_y - 1e-6
        assert pose.position[1] + half_y <= REGION.max_y + 1e-6


def test_object_too_large_for_region_fails_closed():
    oversized = make_bbox(1.0, 1.0, 0.05)
    with pytest.raises(AssertionError, match="does not fit in region"):
        compute_drop_poses([oversized], REGION, generator=seeded())


def test_clutter_spread_narrows_the_usable_area():
    bboxes = [make_bbox(0.02, 0.02, 0.02) for _ in range(8)]
    tight = ClutterDropParams(clutter_spread=0.25, xy_sampling=XySampling.UNIFORM)
    poses = compute_drop_poses(bboxes, REGION, tight, generator=seeded(5))

    centre_x = (REGION.min_x + REGION.max_x) / 2.0
    half_width = (REGION.max_x - REGION.min_x) / 2.0
    for pose in poses:
        assert abs(pose.position[0] - centre_x) <= half_width * 0.25 + 1e-6


# ------------------------------------------------------------------ drop heights


def test_clear_column_starts_just_above_the_floor():
    """An object with nothing beneath it must not be lifted over unrelated objects."""
    bbox = make_bbox(0.04, 0.04, 0.06)
    poses = compute_drop_poses([bbox], REGION, ClutterDropParams(random_yaw=False), generator=seeded())

    expected_bottom = REGION.floor_z + ClutterDropParams().clearance_m
    assert poses[0].position[2] + float(bbox.min_point[0][2]) == pytest.approx(expected_bottom)


def test_origin_offset_is_respected_so_the_base_clears_the_floor():
    """Placement uses the box's own bottom offset, not an assumption that the origin is centred."""
    offset_bbox = make_bbox(0.04, 0.04, 0.06, origin_offset_z=0.03)
    poses = compute_drop_poses([offset_bbox], REGION, ClutterDropParams(random_yaw=False), generator=seeded())

    lowest_point = poses[0].position[2] + float(offset_bbox.min_point[0][2])
    assert lowest_point == pytest.approx(REGION.floor_z + ClutterDropParams().clearance_m)


def test_overlapping_footprints_stack_and_disjoint_ones_do_not():
    """The ladder is per column: only objects sharing a column are lifted over each other."""
    tall = make_bbox(0.02, 0.02, 0.10)
    # A region only wide enough for one column forces every object to overlap.
    narrow = ClutterRegion(min_x=-0.01, min_y=-0.01, max_x=0.01, max_y=0.01, floor_z=0.5)
    stacked = compute_drop_poses(
        [tall, tall, tall],
        narrow,
        ClutterDropParams(xy_sampling=XySampling.UNIFORM, random_yaw=False),
        generator=seeded(1),
    )
    heights = sorted(pose.position[2] for pose in stacked)
    assert heights[1] > heights[0] and heights[2] > heights[1], "objects sharing a column must stack"

    # Widely separated cells: nothing is lifted.
    wide = ClutterRegion(min_x=-2.0, min_y=-2.0, max_x=2.0, max_y=2.0, floor_z=0.5)
    spread = compute_drop_poses(
        [make_bbox(0.02, 0.02, 0.02) for _ in range(4)],
        wide,
        ClutterDropParams(random_yaw=False),
        generator=seeded(2),
    )
    assert len({round(pose.position[2], 6) for pose in spread}) == 1, "disjoint columns must share a height"


@pytest.mark.parametrize("sampling", [XySampling.UNIFORM, XySampling.GRID_CELLS])
def test_no_interpenetration_at_spawn(sampling):
    """The core guarantee: a drop layout is always penetration-free."""
    bboxes = [make_bbox(0.06, 0.03, 0.02), make_bbox(0.03, 0.03, 0.09), make_bbox(0.08, 0.05, 0.03)] * 3
    params = ClutterDropParams(xy_sampling=sampling)
    for seed in range(5):
        poses = compute_drop_poses(bboxes, REGION, params, generator=seeded(seed))
        assert_no_penetration(bboxes, poses)


# ------------------------------------------------------------------- drop order


def test_flattest_first_drops_shortest_object_first():
    tall, flat, medium = make_bbox(0.03, 0.03, 0.20), make_bbox(0.05, 0.05, 0.01), make_bbox(0.04, 0.04, 0.07)
    poses = compute_drop_poses(
        [tall, flat, medium], REGION, ClutterDropParams(drop_order=DropOrder.FLATTEST_FIRST), generator=seeded()
    )
    assert poses[1].drop_index == 0, "flattest object must be dropped first"
    assert poses[0].drop_index == 2, "tallest object must be dropped last"


def test_as_listed_preserves_caller_order():
    bboxes = [make_bbox(0.03, 0.03, 0.20), make_bbox(0.05, 0.05, 0.01), make_bbox(0.04, 0.04, 0.07)]
    poses = compute_drop_poses(bboxes, REGION, ClutterDropParams(drop_order=DropOrder.AS_LISTED), generator=seeded())
    assert [pose.drop_index for pose in poses] == [0, 1, 2]


def test_shuffle_varies_order_across_seeds():
    bboxes = [make_bbox(0.03, 0.03, 0.02) for _ in range(8)]
    params = ClutterDropParams(drop_order=DropOrder.SHUFFLE)
    orders = {
        tuple(pose.drop_index for pose in compute_drop_poses(bboxes, REGION, params, generator=seeded(seed)))
        for seed in range(6)
    }
    assert len(orders) > 1, "shuffling must produce different orders across seeds"


# ---------------------------------------------------------------- reproducibility


def test_same_seed_reproduces_layout_exactly():
    bboxes = [make_bbox(0.05, 0.03, 0.04) for _ in range(6)]
    first = compute_drop_poses(bboxes, REGION, generator=seeded(11))
    second = compute_drop_poses(bboxes, REGION, generator=seeded(11))
    assert first == second


def test_different_seeds_produce_different_layouts():
    bboxes = [make_bbox(0.05, 0.03, 0.04) for _ in range(6)]
    first = compute_drop_poses(bboxes, REGION, generator=seeded(11))
    second = compute_drop_poses(bboxes, REGION, generator=seeded(12))
    assert first != second


# ---------------------------------------------------------------------- rotation


def test_random_yaw_disabled_gives_identity_rotation():
    poses = compute_drop_poses(
        [make_bbox(0.05, 0.05, 0.05)], REGION, ClutterDropParams(random_yaw=False), generator=seeded()
    )
    assert poses[0].rotation_xyzw == (0.0, 0.0, 0.0, 1.0)


def test_rotation_is_a_unit_yaw_quaternion():
    poses = compute_drop_poses([make_bbox(0.05, 0.05, 0.05) for _ in range(5)], REGION, generator=seeded(7))
    for pose in poses:
        x, y, z, w = pose.rotation_xyzw
        assert math.isclose(math.sqrt(x * x + y * y + z * z + w * w), 1.0, rel_tol=1e-6)
        assert (x, y) == (0.0, 0.0), "clutter yaw must rotate about Z only"


def test_yaw_widens_footprint_and_placement_accounts_for_it():
    """A long object turned 45 degrees needs more room, and must still land inside."""
    elongated = [make_bbox(0.16, 0.02, 0.02) for _ in range(4)]
    poses = compute_drop_poses(elongated, REGION, generator=seeded(4))
    for bbox, pose in zip(elongated, poses):
        half_x, half_y = footprint_half_extents(bbox, pose.rotation_xyzw)
        assert pose.position[0] - half_x >= REGION.min_x - 1e-6
        assert pose.position[0] + half_x <= REGION.max_x + 1e-6
        assert pose.position[1] - half_y >= REGION.min_y - 1e-6
        assert pose.position[1] + half_y <= REGION.max_y + 1e-6


# ------------------------------------------------------------------------ region


def test_region_rejects_inverted_bounds():
    with pytest.raises(AssertionError):
        ClutterRegion(min_x=0.2, min_y=-0.1, max_x=-0.2, max_y=0.1, floor_z=0.0)


def test_region_scaled_keeps_centre_and_floor():
    region = ClutterRegion(min_x=0.0, min_y=0.0, max_x=1.0, max_y=2.0, floor_z=0.7)
    scaled = region.scaled(0.5)
    assert (scaled.min_x, scaled.max_x) == pytest.approx((0.25, 0.75))
    assert (scaled.min_y, scaled.max_y) == pytest.approx((0.5, 1.5))
    assert scaled.floor_z == region.floor_z
