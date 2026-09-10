# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for clutter drop-pose generation."""

import math
import torch

import pytest

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena_examples.relations.clutter.drop_poses import (
    ClutterDropParams,
    ClutterRegion,
    DropOrder,
    DropPose,
    MemberDropParams,
    XySampling,
    compute_drop_poses,
)

REGION = ClutterRegion(min_x=-0.18, min_y=-0.23, max_x=0.18, max_y=0.23, floor_z=0.8)


def make_bbox(
    size_x: float,
    size_y: float,
    size_z: float,
    origin_offset_z: float = 0.0,
    origin_offset_x: float = 0.0,
    origin_offset_y: float = 0.0,
):
    """Object-local bbox of the given size, optionally shifted so the origin is not centered."""
    half_x, half_y, half_z = size_x / 2.0, size_y / 2.0, size_z / 2.0
    return AxisAlignedBoundingBox(
        min_point=(-half_x + origin_offset_x, -half_y + origin_offset_y, -half_z + origin_offset_z),
        max_point=(half_x + origin_offset_x, half_y + origin_offset_y, half_z + origin_offset_z),
    )


def assert_footprints_inside(bboxes: list, poses: list[DropPose], region: ClutterRegion) -> None:
    """Every placed object's rotated footprint must lie wholly within the region."""
    for index, (bbox, pose) in enumerate(zip(bboxes, poses)):
        rotated = bbox.rotated_by_quat(torch.tensor([pose.rotation_xyzw], dtype=torch.float32))
        min_x = pose.position_xyz[0] + float(rotated.min_point[0][0])
        max_x = pose.position_xyz[0] + float(rotated.max_point[0][0])
        min_y = pose.position_xyz[1] + float(rotated.min_point[0][1])
        max_y = pose.position_xyz[1] + float(rotated.max_point[0][1])
        assert min_x >= region.min_x - 1e-6, f"object {index} spans x={min_x:.4f} past {region.min_x}"
        assert max_x <= region.max_x + 1e-6, f"object {index} spans x={max_x:.4f} past {region.max_x}"
        assert min_y >= region.min_y - 1e-6, f"object {index} spans y={min_y:.4f} past {region.min_y}"
        assert max_y <= region.max_y + 1e-6, f"object {index} spans y={max_y:.4f} past {region.max_y}"


def axis_aligned(count: int = 1) -> list:
    """Per-member params with yaw sampling off, for deterministic axis-aligned drops."""
    return [MemberDropParams(random_yaw=False)] * count


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
                lo_i = poses[i].position_xyz[axis] + float(bbox_i.min_point[0][axis])
                hi_i = poses[i].position_xyz[axis] + float(bbox_i.max_point[0][axis])
                lo_j = poses[j].position_xyz[axis] + float(bbox_j.min_point[0][axis])
                hi_j = poses[j].position_xyz[axis] + float(bbox_j.max_point[0][axis])
                overlaps.append(lo_i < hi_j and lo_j < hi_i)
            assert not all(overlaps), f"objects {i} and {j} interpenetrate at spawn"


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


@pytest.mark.parametrize("sampling", [XySampling.UNIFORM, XySampling.GRID_CELLS])
@pytest.mark.parametrize("size_xy", [(0.12, 0.04), (0.16, 0.02)])
def test_rotated_footprint_stays_inside_region(sampling, size_xy):
    bboxes = [make_bbox(size_xy[0], size_xy[1], 0.03) for _ in range(6)]
    params = ClutterDropParams(xy_sampling=sampling)
    poses = compute_drop_poses(bboxes, REGION, params, generator=seeded(3))

    for bbox, pose in zip(bboxes, poses):
        half_x, half_y = footprint_half_extents(bbox, pose.rotation_xyzw)
        assert pose.position_xyz[0] - half_x >= REGION.min_x - 1e-6
        assert pose.position_xyz[0] + half_x <= REGION.max_x + 1e-6
        assert pose.position_xyz[1] - half_y >= REGION.min_y - 1e-6
        assert pose.position_xyz[1] + half_y <= REGION.max_y + 1e-6


def test_object_too_large_for_region_fails_closed():
    oversized = make_bbox(1.0, 1.0, 0.05)
    with pytest.raises(AssertionError, match="does not fit in region"):
        compute_drop_poses([oversized], REGION, generator=seeded())


def test_unlucky_yaw_is_resampled_rather_than_failing():
    narrow = ClutterRegion(min_x=-0.18, min_y=-0.06, max_x=0.18, max_y=0.06, floor_z=0.0)
    elongated = make_bbox(0.28, 0.04, 0.02)
    params = ClutterDropParams(max_yaw_attempts=64)

    for seed in range(5):
        poses = compute_drop_poses([elongated], narrow, params, generator=seeded(seed))
        half_x, half_y = footprint_half_extents(elongated, poses[0].rotation_xyzw)
        assert half_y <= (narrow.max_y - narrow.min_y) / 2.0 + 1e-6, "chosen yaw must fit the narrow axis"
        assert poses[0].position_xyz[1] - half_y >= narrow.min_y - 1e-6
        assert poses[0].position_xyz[1] + half_y <= narrow.max_y + 1e-6


def test_no_orientation_fits_reports_attempt_count():
    oversized = make_bbox(1.0, 1.0, 0.05)
    params = ClutterDropParams(max_yaw_attempts=3)
    with pytest.raises(AssertionError, match="after 3 orientation attempt"):
        compute_drop_poses([oversized], REGION, params, generator=seeded())


def test_axis_aligned_layouts_try_a_single_orientation():
    oversized = make_bbox(1.0, 1.0, 0.05)
    params = ClutterDropParams(max_yaw_attempts=8)
    with pytest.raises(AssertionError, match="after 1 orientation attempt"):
        compute_drop_poses([oversized], REGION, params, generator=seeded(), member_params=axis_aligned())


def test_clear_column_starts_just_above_the_floor():
    bbox = make_bbox(0.04, 0.04, 0.06)
    poses = compute_drop_poses([bbox], REGION, ClutterDropParams(), generator=seeded())

    expected_bottom = REGION.floor_z + MemberDropParams().clearance_m
    assert poses[0].position_xyz[2] + float(bbox.min_point[0][2]) == pytest.approx(expected_bottom)


def test_origin_offset_is_respected_so_the_base_clears_the_floor():
    offset_bbox = make_bbox(0.04, 0.04, 0.06, origin_offset_z=0.03)
    poses = compute_drop_poses([offset_bbox], REGION, ClutterDropParams(), generator=seeded())

    lowest_point = poses[0].position_xyz[2] + float(offset_bbox.min_point[0][2])
    assert lowest_point == pytest.approx(REGION.floor_z + MemberDropParams().clearance_m)


def test_overlapping_footprints_stack_and_disjoint_ones_do_not():
    tall = make_bbox(0.02, 0.02, 0.10)
    narrow = ClutterRegion(min_x=-0.01, min_y=-0.01, max_x=0.01, max_y=0.01, floor_z=0.5)
    stacked = compute_drop_poses(
        [tall, tall, tall],
        narrow,
        ClutterDropParams(xy_sampling=XySampling.UNIFORM),
        member_params=axis_aligned(3),
        generator=seeded(1),
    )
    heights = sorted(pose.position_xyz[2] for pose in stacked)
    assert heights[1] > heights[0] and heights[2] > heights[1], "objects sharing a column must stack"

    wide = ClutterRegion(min_x=-2.0, min_y=-2.0, max_x=2.0, max_y=2.0, floor_z=0.5)
    spread = compute_drop_poses(
        [make_bbox(0.02, 0.02, 0.02) for _ in range(4)],
        wide,
        ClutterDropParams(),
        member_params=axis_aligned(4),
        generator=seeded(2),
    )
    assert len({round(pose.position_xyz[2], 6) for pose in spread}) == 1, "disjoint columns must share a height"


@pytest.mark.parametrize("sampling", [XySampling.UNIFORM, XySampling.GRID_CELLS])
def test_no_interpenetration_at_spawn(sampling):
    bboxes = [make_bbox(0.06, 0.03, 0.02), make_bbox(0.03, 0.03, 0.09), make_bbox(0.08, 0.05, 0.03)] * 3
    params = ClutterDropParams(xy_sampling=sampling)
    for seed in range(5):
        poses = compute_drop_poses(bboxes, REGION, params, generator=seeded(seed))
        assert_no_penetration(bboxes, poses)


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


def test_random_yaw_disabled_gives_identity_rotation():
    poses = compute_drop_poses(
        [make_bbox(0.05, 0.05, 0.05)], REGION, ClutterDropParams(), generator=seeded(), member_params=axis_aligned()
    )
    assert poses[0].rotation_xyzw == (0.0, 0.0, 0.0, 1.0)


def test_base_rotation_tilts_object_and_refits_drop_height():
    pitch_quarter_turn = (0.0, math.sin(math.pi / 4.0), 0.0, math.cos(math.pi / 4.0))
    upright = make_bbox(0.04, 0.04, 0.20)

    poses = compute_drop_poses(
        [upright],
        REGION,
        ClutterDropParams(),
        generator=seeded(),
        base_rotations_xyzw=[pitch_quarter_turn],
        member_params=axis_aligned(),
    )

    pose = poses[0]
    assert pose.rotation_xyzw == pytest.approx(pitch_quarter_turn)
    rotated = upright.rotated_by_quat(torch.tensor([pose.rotation_xyzw], dtype=torch.float32))
    assert float(rotated.size[0][0]) == pytest.approx(0.20)
    assert float(rotated.size[0][2]) == pytest.approx(0.04)
    assert pose.position_xyz[2] + float(rotated.bottom_surface_z[0]) == pytest.approx(
        REGION.floor_z + MemberDropParams().clearance_m
    )


def test_rotation_is_a_unit_yaw_quaternion():
    poses = compute_drop_poses([make_bbox(0.05, 0.05, 0.05) for _ in range(5)], REGION, generator=seeded(7))
    for pose in poses:
        x, y, z, w = pose.rotation_xyzw
        assert math.isclose(math.sqrt(x * x + y * y + z * z + w * w), 1.0, rel_tol=1e-6)
        assert (x, y) == (0.0, 0.0), "clutter yaw must rotate about Z only"


def test_region_rejects_inverted_bounds():
    with pytest.raises(AssertionError):
        ClutterRegion(min_x=0.2, min_y=-0.1, max_x=-0.2, max_y=0.1, floor_z=0.0)


def test_region_scaled_keeps_center_and_floor():
    region = ClutterRegion(min_x=0.0, min_y=0.0, max_x=1.0, max_y=2.0, floor_z=0.7)
    scaled = region.scaled(0.5)
    assert (scaled.min_x, scaled.max_x) == pytest.approx((0.25, 0.75))
    assert (scaled.min_y, scaled.max_y) == pytest.approx((0.5, 1.5))
    assert scaled.floor_z == region.floor_z


def test_flattest_first_ranks_by_height_after_the_authored_rotation():
    plate = make_bbox(0.20, 0.20, 0.01)
    block = make_bbox(0.06, 0.06, 0.05)
    upright = (0.0, math.sin(math.pi / 4.0), 0.0, math.cos(math.pi / 4.0))
    identity = (0.0, 0.0, 0.0, 1.0)

    params = ClutterDropParams(drop_order=DropOrder.FLATTEST_FIRST)
    poses = compute_drop_poses(
        [plate, block],
        REGION,
        params,
        seeded(),
        base_rotations_xyzw=[upright, identity],
        member_params=axis_aligned(2),
    )

    assert poses[1].drop_index < poses[0].drop_index, "pitched plate was ranked by its local thickness"


@pytest.mark.parametrize("sampling", [XySampling.GRID_CELLS, XySampling.UNIFORM])
@pytest.mark.parametrize(
    "offsets",
    [
        [(0.03, 0.03)] * 6,
        [(0.04, 0.0)] * 6,
        [(0.025, -0.025), (-0.03, 0.0), (0.0, 0.03), (0.0, 0.0)],
        [(0.0, 0.0)] * 6,
    ],
)
def test_off_center_origins_stay_contained_and_apart(sampling, offsets):
    bboxes = [make_bbox(0.05, 0.05, 0.03, origin_offset_x=dx, origin_offset_y=dy) for dx, dy in offsets]
    params = ClutterDropParams(xy_sampling=sampling)
    for seed in range(3):
        poses = compute_drop_poses(bboxes, REGION, params, generator=seeded(seed))
        assert_footprints_inside(bboxes, poses, REGION)
        assert_no_penetration(bboxes, poses)


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_sampled_yaw_rotates_a_tilted_base_about_world_z(seed):
    from isaaclab.utils.math import matrix_from_quat

    bbox = make_bbox(0.06, 0.08, 0.1)
    base = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))
    yaw_pose = compute_drop_poses([bbox], REGION, generator=torch.Generator().manual_seed(seed))[0]
    tilted_pose = compute_drop_poses(
        [bbox], REGION, generator=torch.Generator().manual_seed(seed), base_rotations_xyzw=[base]
    )[0]
    yaw_rotation = matrix_from_quat(torch.tensor(yaw_pose.rotation_xyzw))
    base_rotation = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    actual = matrix_from_quat(torch.tensor(tilted_pose.rotation_xyzw))
    torch.testing.assert_close(actual, yaw_rotation @ base_rotation, atol=1e-6, rtol=0)
    assert abs(float(actual[1, 2])) > 0.01


def test_explicit_empty_member_parameters_are_rejected():
    with pytest.raises(AssertionError, match="0 member params for 1 objects"):
        compute_drop_poses([make_bbox(0.05, 0.05, 0.05)], REGION, member_params=[])


def test_disabled_yaw_preserves_authored_quaternion_exactly():
    base = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    pose = compute_drop_poses(
        [make_bbox(0.05, 0.06, 0.07)],
        REGION,
        base_rotations_xyzw=[base],
        member_params=[MemberDropParams(random_yaw=False)],
    )[0]
    assert pose.rotation_xyzw == base
