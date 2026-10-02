# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check occupancy and packing edge cases without launching Isaac Sim."""

import math
import torch

import pytest

from isaaclab_arena_environments.return_to_service.measurements import (
    box_contained,
    box_contained_on_support,
    box_corners,
    box_overlaps,
    point_velocity,
    sphere_overlaps_cylinder,
)


@pytest.fixture(params=(torch.float32, torch.float64))
def dtype(request):
    return request.param


def _pose(position=(0, 0, 0), quaternion=(0, 0, 0, 1), dtype=torch.float64):
    return torch.tensor([*position, *quaternion], dtype=dtype)


def test_point_velocity_matches_finite_differences_and_broadcasts_batches():
    generator = torch.Generator().manual_seed(816)
    reference = torch.randn((2, 4, 3), generator=generator, dtype=torch.float64)
    point = torch.randn((2, 4, 3), generator=generator, dtype=torch.float64)
    velocity = torch.randn((4, 3), generator=generator, dtype=torch.float64)
    omega = torch.tensor((0.7, -1.1, 0.4), dtype=torch.float64)
    wx, wy, wz = omega.tolist()
    rotation_generator = torch.tensor(((0, -wz, wy), (wz, 0, -wx), (-wy, wx, 0)), dtype=torch.float64)

    def trajectory(time):
        rotation = torch.matrix_exp(time * rotation_generator)
        return reference + time * velocity + (point - reference) @ rotation.T

    dt = 1e-5
    finite_difference = (trajectory(dt) - trajectory(-dt)) / (2 * dt)
    measured = point_velocity(velocity, omega, point, reference)
    assert measured.shape == (2, 4, 3) and measured.dtype == torch.float64
    torch.testing.assert_close(measured, finite_difference, atol=1e-9, rtol=1e-9)


def test_point_velocity_is_invariant_to_world_origin_and_rotates_with_the_frame(dtype):
    velocity = torch.tensor((0.2, -0.3, 0.1), dtype=dtype)
    omega = torch.tensor((-0.7, 1.2, 0.5), dtype=dtype)
    point = torch.tensor(((0.1, 0.4, -0.2), (0.7, -0.2, 0.6)), dtype=dtype)
    reference = torch.tensor((-0.3, 0.1, 0.2), dtype=dtype)
    rotation = torch.matrix_exp(torch.tensor(((0.0, -0.4, 0.7), (0.4, 0.0, 0.2), (-0.7, -0.2, 0.0)), dtype=dtype))
    shift = torch.tensor((2.0, -3.0, 5.0), dtype=dtype)
    expected = point_velocity(velocity, omega, point, reference) @ rotation.T
    actual = point_velocity(
        velocity @ rotation.T, omega @ rotation.T, point @ rotation.T + shift, reference @ rotation.T + shift
    )
    torch.testing.assert_close(actual, expected)


def test_noncentered_corner_bounds_preserve_dtype_and_device(dtype):
    reference = torch.zeros((2, 7), dtype=dtype)
    bounds = ((0.1, -0.3, 0.2), (0.8, 0.4, 0.6))
    corners = box_corners(bounds, reference)
    assert corners.shape == (8, 3)
    assert corners.dtype == dtype and corners.device == reference.device
    assert torch.allclose(corners.amin(dim=0), reference.new_tensor(bounds[0]))
    assert torch.allclose(corners.amax(dim=0), reference.new_tensor(bounds[1]))
    assert len(torch.unique(corners, dim=0)) == 8


def test_containment_checks_rotated_corners_not_just_object_center(dtype):
    bounds = ((-0.7, -0.7, -0.1), (0.7, 0.7, 0.1))
    region = ((-0.8, -0.8, -0.2), (0.8, 0.8, 0.2))
    quarter_turn = (0, 0, math.sin(math.pi / 8), math.cos(math.pi / 8))
    poses = torch.stack((_pose(dtype=dtype), _pose(quaternion=quarter_turn, dtype=dtype)))
    assert box_contained(poses, bounds, region).tolist() == [True, False]


def test_pose_uses_xyzw_and_rotates_noncentered_bounds(dtype):
    bounds = ((0.1, -0.01, -0.01), (0.2, 0.01, 0.01))
    region = ((-0.011, 0.099, -0.011), (0.011, 0.201, 0.011))
    pose = _pose(quaternion=(0, 0, math.sqrt(0.5), math.sqrt(0.5)), dtype=dtype)
    assert box_contained(pose, bounds, region).item()
    assert box_overlaps(pose, bounds, region).item()
    assert not box_overlaps(_pose(dtype=dtype), bounds, region).item()
    pose[3:] *= -1
    assert box_contained(pose, bounds, region).item()


def test_partial_insertion_and_boundary_contact_are_occupied(dtype):
    bounds = ((-0.010, -0.010, -0.010), (0.010, 0.010, 0.010))
    region = ((-0.020, -0.020, -0.020), (0.020, 0.020, 0.020))
    poses = torch.stack([_pose((x, 0, 0), dtype=dtype) for x in (0, 0.025, 0.030, 0.031)])
    assert box_overlaps(poses, bounds, region).tolist() == [True, True, True, False]
    assert box_contained(poses, bounds, region).tolist() == [True, False, False, False]


def test_resting_contact_allowance_is_bounded_and_changes_only_the_support_face(dtype):
    bounds = ((-0.01, -0.01, 0.0), (0.01, 0.01, 0.01))
    region = torch.tensor(((-0.02, -0.02, 0.0), (0.02, 0.02, 0.02)), dtype=dtype)
    unchanged = region.clone()
    positions = (
        (0, 0, -1.647e-6),  # Measured resting brush penetration in the physical packing trial.
        (0, 0, -49e-6),
        (0, 0, -52e-6),
        (0.010002, 0, 0),
        (-0.010002, 0, 0),
        (0, 0.010002, 0),
        (0, -0.010002, 0),
        (0, 0, 0.010002),
    )
    poses = torch.stack([_pose(position, dtype=dtype) for position in positions])
    result = box_contained_on_support(poses, bounds, region, floor_allowance_m=50e-6)
    assert result.tolist() == [True, True, False, False, False, False, False, False]
    assert torch.equal(region, unchanged)
    assert not box_contained(poses[0], bounds, region)


def test_support_allowance_preserves_batches_and_checks_tilted_corners(dtype):
    bounds = ((-0.01, -0.01, 0.0), (0.01, 0.01, 0.01))
    region = ((-0.02, -0.02, 0.0), (0.02, 0.02, 0.02))
    poses = _pose(dtype=dtype).expand(2, 3, 7).clone()
    poses[1, 2, 3:] = poses.new_tensor((0, math.sin(0.01), 0, math.cos(0.01)))
    result = box_contained_on_support(poses, bounds, region, floor_allowance_m=50e-6)
    assert result.tolist() == [[True, True, True], [True, True, False]]


@pytest.mark.parametrize("allowance", (-1e-6, float("nan"), float("inf")))
def test_invalid_support_allowance_is_rejected(allowance):
    bounds = ((-0.01, -0.01, 0), (0.01, 0.01, 0.01))
    with pytest.raises(AssertionError):
        box_contained_on_support(_pose(), bounds, bounds, floor_allowance_m=allowance)


def test_separating_edge_axis_rejects_a_false_overlap(dtype):
    # A linear-program feasibility check of the twelve box half-spaces confirms
    # this pair is disjoint, although every face-normal projection overlaps.
    pose = torch.tensor(
        [-0.5765011240, 0.3804832869, -0.2554200747, -0.7198719507, -0.4809437944, -0.1020370155, 0.4899651913],
        dtype=dtype,
    )
    bounds = ((-0.8, -0.15, -0.1), (0.8, 0.15, 0.1))
    region = ((-0.25, -0.45, -0.2), (0.25, 0.45, 0.2))
    assert not box_overlaps(pose, bounds, region).item()


def test_multiple_leading_batch_dimensions_are_preserved(dtype):
    poses = _pose(dtype=dtype).expand(2, 3, 7).clone()
    poses[1, 2, 0] = 0.5
    bounds = ((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01))
    region = ((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1))
    result = box_overlaps(poses, bounds, region)
    assert result.shape == (2, 3)
    assert result.tolist() == [[True, True, True], [True, True, False]]
    assert torch.equal(result, box_contained(poses, bounds, region))


def test_cylinder_rejects_its_bounding_box_corner_and_tracks_partial_exit(dtype):
    bounds = ((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01))
    poses = torch.stack(
        [_pose(position, dtype=dtype) for position in ((0, 0, 0), (0, 0.9, 0.9), (1.009, 0, 0), (1.030, 0, 0))]
    )
    result = sphere_overlaps_cylinder(poses, bounds, (-1, 1), (0, 0), 1)
    assert result.tolist() == [True, False, True, False]


def test_cylinder_uses_combined_radial_and_endcap_distance(dtype):
    bounds = ((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01))
    poses = torch.stack([_pose((distance, distance, 0), dtype=dtype) for distance in (1.012, 1.014)])
    assert sphere_overlaps_cylinder(poses, bounds, (-1, 1), (0, 0), 1).tolist() == [True, False]


def test_conservative_sphere_rotates_the_local_bounds_center(dtype):
    bounds = ((0.09, -0.005, -0.005), (0.11, 0.005, 0.005))
    pose = _pose(quaternion=(0, 0, math.sqrt(0.5), math.sqrt(0.5)), dtype=dtype)
    assert sphere_overlaps_cylinder(pose, bounds, (-0.02, 0.02), (0.1, 0), 0.02).item()
    assert not sphere_overlaps_cylinder(_pose(dtype=dtype), bounds, (-0.02, 0.02), (0.1, 0), 0.02).item()


@pytest.mark.parametrize("num_envs", (1, 4))
def test_heterogeneous_bounds_broadcast_over_environments(dtype, num_envs):
    bounds = torch.tensor(
        [
            ((-0.02, -0.01, -0.03), (0.02, 0.01, 0.03)),
            ((0.04, -0.01, 0.01), (0.09, 0.02, 0.03)),
            ((-0.01, -0.06, -0.02), (0.01, 0.06, 0.02)),
        ],
        dtype=dtype,
    )
    poses = _pose(dtype=dtype).expand(num_envs, 3, 7).clone()
    poses[..., 0] = torch.arange(num_envs, dtype=dtype)[:, None] * 0.08
    poses[:, 1, 3:] = poses.new_tensor((0, 0, math.sqrt(0.5), math.sqrt(0.5)))
    poses[:, 2, 1] = 0.18
    # Each environment has its own region, shared by its three objects.
    regions = poses.new_tensor(((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1))).expand(num_envs, 1, 2, 3).clone()
    regions[:, 0, 1, 0] += torch.arange(num_envs, dtype=dtype) * 0.04
    corners = box_corners(bounds, poses)
    assert corners.shape == (3, 8, 3)
    torch.testing.assert_close(corners.amin(dim=-2), bounds[:, 0])
    torch.testing.assert_close(corners.amax(dim=-2), bounds[:, 1])

    for predicate in (box_contained, box_overlaps):
        batched = predicate(poses, bounds, regions)
        expected = torch.empty((num_envs, 3), dtype=torch.bool)
        for env_id in range(num_envs):
            for object_id in range(3):
                expected[env_id, object_id] = predicate(poses[env_id, object_id], bounds[object_id], regions[env_id, 0])
        assert torch.equal(batched, expected)
        assert batched.any() and not batched.all()

    cylinder = ((-0.1, 0.1), (0, 0), 0.1)
    batched = sphere_overlaps_cylinder(poses, bounds, *cylinder)
    expected = torch.empty((num_envs, 3), dtype=torch.bool)
    for env_id in range(num_envs):
        for object_id in range(3):
            expected[env_id, object_id] = sphere_overlaps_cylinder(
                poses[env_id, object_id], bounds[object_id], *cylinder
            )
    assert torch.equal(batched, expected)
    assert batched.any() and not batched.all()
