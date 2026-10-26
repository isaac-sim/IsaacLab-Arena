# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for AxisAlignedBoundingBox and OrientedBoundingBox with always-tensor API."""

import math
import torch

import pytest

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, OrientedBoundingBox

_CORNER_SIGNS = torch.tensor(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    dtype=torch.float32,
)
"""Box-local corner signs in get_corners_at() order."""


def test_bounding_box_single_env_properties():
    """Single env: properties return tensors with leading dim 1."""
    aabb = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(2.0, 4.0, 0.2))
    assert aabb.num_envs == 1
    assert isinstance(aabb.min_point, torch.Tensor)
    assert aabb.min_point.shape == (1, 3)
    torch.testing.assert_close(aabb.min_point, torch.tensor([[0.0, 0.0, 0.0]]))
    torch.testing.assert_close(aabb.max_point[0, 2], torch.tensor(0.2), atol=1e-6, rtol=0)
    torch.testing.assert_close(aabb.size, torch.tensor([[2.0, 4.0, 0.2]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(aabb.center, torch.tensor([[1.0, 2.0, 0.1]]), atol=1e-6, rtol=0)
    assert isinstance(aabb.top_surface_z, torch.Tensor)
    assert aabb.top_surface_z.shape == (1,)
    torch.testing.assert_close(aabb.top_surface_z, torch.tensor([0.2]), atol=1e-6, rtol=0)


def test_bounding_box_single_env_transforms():
    """Single env: translated, scaled, centered, rotated return AABBs with correct values."""
    aabb = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(2.0, 1.0, 0.5))
    moved = aabb.translated((1.0, 2.0, 0.5))
    torch.testing.assert_close(moved.min_point, torch.tensor([[1.0, 2.0, 0.5]]))
    torch.testing.assert_close(moved.max_point, torch.tensor([[3.0, 3.0, 1.0]]))

    scaled = aabb.scaled((2.0, 0.5, 3.0))
    torch.testing.assert_close(scaled.max_point, torch.tensor([[4.0, 0.5, 1.5]]))

    centered = AxisAlignedBoundingBox(min_point=(2.0, 4.0, 0.0), max_point=(4.0, 6.0, 2.0)).centered()
    torch.testing.assert_close(centered.center, torch.tensor([[0.0, 0.0, 0.0]]), atol=1e-6, rtol=0)

    rotated = aabb.rotated_90_around_z(1)
    torch.testing.assert_close(rotated.min_point, torch.tensor([[-1.0, 0.0, 0.0]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(rotated.max_point, torch.tensor([[0.0, 2.0, 0.5]]), atol=1e-6, rtol=0)


def test_rotated_around_z_single_angle():
    """90° matches rotated_90_around_z; 45° inflates a centered box to its conservative enclosure."""
    off_origin = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(2.0, 1.0, 0.5))
    rot90 = off_origin.rotated_around_z(math.pi / 2)
    torch.testing.assert_close(rot90.min_point, off_origin.rotated_90_around_z(1).min_point, atol=1e-6, rtol=0)
    torch.testing.assert_close(rot90.min_point, torch.tensor([[-1.0, 0.0, 0.0]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(rot90.max_point, torch.tensor([[0.0, 2.0, 0.5]]), atol=1e-6, rtol=0)

    # Half-extents (0.2, 0.1); enclosing half-extent = a|cos| + b|sin| = 0.3*cos(45°) on each axis.
    centered = AxisAlignedBoundingBox(min_point=(-0.2, -0.1, -0.05), max_point=(0.2, 0.1, 0.05))
    rot45 = centered.rotated_around_z(math.pi / 4)
    half = (0.2 + 0.1) * math.cos(math.pi / 4)
    torch.testing.assert_close(rot45.min_point, torch.tensor([[-half, -half, -0.05]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(rot45.max_point, torch.tensor([[half, half, 0.05]]), atol=1e-6, rtol=0)


def test_rotated_around_z_off_center_arbitrary_angle():
    """An off-center box at 30° enclosed by hand-computed corner extents (center shifts, Z fixed)."""
    box = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(2.0, 1.0, 0.5))
    rot = box.rotated_around_z(math.pi / 6)
    cos, sin = math.cos(math.pi / 6), math.sin(math.pi / 6)
    corners = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
    xs = [x * cos - y * sin for x, y in corners]
    ys = [x * sin + y * cos for x, y in corners]
    torch.testing.assert_close(rot.min_point, torch.tensor([[min(xs), min(ys), 0.0]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(rot.max_point, torch.tensor([[max(xs), max(ys), 0.5]]), atol=1e-6, rtol=0)


def test_rotated_by_quat_encloses_pitched_box():
    """A 90° pitch about Y swaps the X and Z extents, unlike the Z-only rotated_around_z."""
    box = AxisAlignedBoundingBox(min_point=(-0.1, -0.2, -0.5), max_point=(0.1, 0.2, 0.5))
    # Pitch quaternion about Y by 90°, as (x, y, z, w).
    pitch_quat = (0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))
    rotated = box.rotated_by_quat(pitch_quat)
    torch.testing.assert_close(rotated.min_point, torch.tensor([[-0.5, -0.2, -0.1]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(rotated.max_point, torch.tensor([[0.5, 0.2, 0.1]]), atol=1e-6, rtol=0)


def test_rotated_around_z_batched_angles_broadcasts_single_box():
    """An (M,) angle tensor broadcasts an N=1 box to M enclosing boxes (one per angle)."""
    aabb = AxisAlignedBoundingBox(min_point=(-0.2, -0.1, 0.0), max_point=(0.2, 0.1, 0.5))
    angles = torch.tensor([0.0, math.pi / 2])
    rotated = aabb.rotated_around_z(angles)
    assert rotated.num_envs == 2
    # Angle 0: unchanged.
    torch.testing.assert_close(rotated.min_point[0], torch.tensor([-0.2, -0.1, 0.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(rotated.max_point[0], torch.tensor([0.2, 0.1, 0.5]), atol=1e-6, rtol=0)
    # Angle 90°: X/Y extents swap for this origin-centered box.
    torch.testing.assert_close(rotated.min_point[1], torch.tensor([-0.1, -0.2, 0.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(rotated.max_point[1], torch.tensor([0.1, 0.2, 0.5]), atol=1e-6, rtol=0)


def test_getitem_selects_single_row():
    """Indexing a batched bbox returns the (N=1) box for that row; out-of-range asserts."""
    boxes = AxisAlignedBoundingBox(
        min_point=torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
        max_point=torch.tensor([[1.0, 1.0, 1.0], [4.0, 5.0, 6.0]]),
    )
    first = boxes[0]
    assert first.num_envs == 1
    torch.testing.assert_close(first.min_point, torch.tensor([[0.0, 0.0, 0.0]]))
    torch.testing.assert_close(first.max_point, torch.tensor([[1.0, 1.0, 1.0]]))

    last = boxes[1]
    torch.testing.assert_close(last.min_point, torch.tensor([[1.0, 2.0, 3.0]]))
    torch.testing.assert_close(last.max_point, torch.tensor([[4.0, 5.0, 6.0]]))

    with pytest.raises(AssertionError):
        _ = boxes[2]


def test_rotated_around_z_mismatched_box_and_angle_counts_raises():
    """Multiple boxes paired with a different count of multiple angles is ambiguous and must assert."""
    boxes = AxisAlignedBoundingBox(
        min_point=torch.tensor([[-0.2, -0.1, 0.0], [-0.2, -0.1, 0.0]]),
        max_point=torch.tensor([[0.2, 0.1, 0.5], [0.2, 0.1, 0.5]]),
    )
    with pytest.raises(AssertionError):
        boxes.rotated_around_z(torch.tensor([0.0, math.pi / 2, math.pi]))


def test_bounding_box_single_env_overlaps():
    """Single env: overlaps() returns a (1,) bool tensor; margin widens the check."""
    a = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 1.0))
    b = AxisAlignedBoundingBox(min_point=(0.5, 0.5, 0.0), max_point=(1.5, 1.5, 0.5))
    assert a.overlaps(b).item() is True

    c = AxisAlignedBoundingBox(min_point=(1.05, 0.0, 0.0), max_point=(2.0, 1.0, 1.0))
    assert a.overlaps(c).item() is False
    assert a.overlaps(c, margin=0.1).item() is True


def test_bounding_box_single_env_get_corners_at():
    """Single env: get_corners_at() returns (1, 8, 3) tensor offset by pos."""
    aabb = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 1.0))
    corners = aabb.get_corners_at()
    assert corners.shape == (1, 8, 3)
    corners_offset = aabb.get_corners_at(pos=torch.tensor([10.0, 0.0, 0.0]))
    torch.testing.assert_close(corners_offset[0, 0], corners[0, 0] + torch.tensor([10.0, 0.0, 0.0]))


def test_bounding_box_multi_env_properties():
    """Multi-env: properties return (N, 3) or (N,) tensors."""
    aabb = AxisAlignedBoundingBox(
        min_point=torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]]),
        max_point=torch.tensor([[1.0, 1.0, 1.0], [3.0, 4.0, 0.5]]),
    )
    assert aabb.num_envs == 2
    assert isinstance(aabb.min_point, torch.Tensor)
    assert aabb.size.shape == (2, 3)
    torch.testing.assert_close(aabb.size[1], torch.tensor([2.0, 2.0, 0.5]))
    torch.testing.assert_close(aabb.center[0], torch.tensor([0.5, 0.5, 0.5]))
    assert aabb.top_surface_z.shape == (2,)
    torch.testing.assert_close(aabb.top_surface_z, torch.tensor([1.0, 0.5]))


def test_bounding_box_multi_env_transforms():
    """Multi-env: translated, scaled, centered, rotated operate per-env."""
    aabb = AxisAlignedBoundingBox(
        min_point=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        max_point=torch.tensor([[2.0, 1.0, 0.5], [3.0, 1.0, 0.5]]),
    )
    moved = aabb.translated(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    torch.testing.assert_close(moved.min_point[0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(moved.min_point[1], torch.tensor([0.0, 1.0, 0.0]))

    rotated = aabb.rotated_90_around_z(1)
    torch.testing.assert_close(rotated.min_point[0], torch.tensor([-1.0, 0.0, 0.0]))
    torch.testing.assert_close(rotated.max_point[1], torch.tensor([0.0, 3.0, 0.5]))

    centered = aabb.centered()
    torch.testing.assert_close(centered.center, torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))


def test_bounding_box_multi_env_overlaps():
    """Multi-env vs single-env: overlaps() returns (N,) bool tensor via broadcasting."""
    batched = AxisAlignedBoundingBox(
        min_point=torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]),
        max_point=torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 1.0]]),
    )
    other = AxisAlignedBoundingBox(min_point=(0.5, 0.5, 0.0), max_point=(1.5, 1.5, 0.5))
    result = batched.overlaps(other)
    assert isinstance(result, torch.Tensor)
    assert result[0].item() is True
    assert result[1].item() is False


def test_bounding_box_multi_env_get_corners_at():
    """Multi-env: get_corners_at() returns (N, 8, 3) tensor."""
    aabb = AxisAlignedBoundingBox(
        min_point=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        max_point=torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    )
    corners = aabb.get_corners_at()
    assert corners.shape == (2, 8, 3)
    corners_with_pos = aabb.get_corners_at(pos=torch.tensor([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]))
    torch.testing.assert_close(corners_with_pos[1, 0], corners[1, 0] + torch.tensor([20.0, 0.0, 0.0]))


def test_intersected_returns_the_common_box():
    """Two overlapping boxes intersect to the region inside both."""
    first = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 1.0))
    second = AxisAlignedBoundingBox(min_point=(0.5, -1.0, 0.25), max_point=(2.0, 0.75, 0.5))

    result = first.intersected(second)

    assert result.min_point[0].tolist() == [0.5, 0.0, 0.25]
    assert result.max_point[0].tolist() == [1.0, 0.75, 0.5]


def test_intersected_reports_empty_axes_without_clamping():
    """Boxes that miss on an axis give min > max there, so callers can see which axis failed."""
    first = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 1.0))
    second = AxisAlignedBoundingBox(min_point=(2.0, 0.0, 0.0), max_point=(3.0, 1.0, 1.0))

    result = first.intersected(second)

    assert result.min_point[0, 0] > result.max_point[0, 0], "Expected an empty X axis"
    assert result.min_point[0, 0].item() == 2.0 and result.max_point[0, 0].item() == 1.0
    assert result.min_point[0, 1] <= result.max_point[0, 1], "Y overlapped and should be intact"


def test_intersected_keeps_per_env_results_when_only_some_envs_overlap():
    """A batched intersection must not discard the envs that do overlap."""
    first = AxisAlignedBoundingBox(
        min_point=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        max_point=torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    )
    second = AxisAlignedBoundingBox(
        min_point=torch.tensor([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        max_point=torch.tensor([[2.0, 1.0, 1.0], [3.0, 1.0, 1.0]]),
    )

    result = first.intersected(second)

    assert result.is_empty.tolist() == [False, True]
    assert result.min_point[0, 0].item() == 0.5 and result.max_point[0, 0].item() == 1.0


def test_intersected_leaves_infinite_axes_untouched():
    """An unbounded box narrows nothing, which is how unset limits are represented."""
    bounded = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 1.0))
    unbounded = AxisAlignedBoundingBox(
        min_point=(-math.inf, -math.inf, -math.inf), max_point=(math.inf, math.inf, math.inf)
    )

    result = bounded.intersected(unbounded)

    assert result.min_point[0].tolist() == bounded.min_point[0].tolist()
    assert result.max_point[0].tolist() == bounded.max_point[0].tolist()


def test_volume_fraction_within():
    """Containment measures object volume, including partial overlap and degenerate boxes."""
    target = AxisAlignedBoundingBox((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    objects = AxisAlignedBoundingBox(
        min_point=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        max_point=torch.tensor([[1.0, 1.0, 1.0], [1.5, 1.0, 1.0], [1.5, 1.5, 1.5], [2.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
    )
    torch.testing.assert_close(objects.volume_fraction_within(target), torch.tensor([1.0, 0.5, 0.125, 0.0, 0.0]))
    torch.testing.assert_close(target.volume_fraction_within(objects), torch.tensor([1.0, 0.5, 0.125, 0.0, 0.0]))
    large = AxisAlignedBoundingBox((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    torch.testing.assert_close(large.volume_fraction_within(target), torch.tensor([0.125]))
    torch.testing.assert_close(target.volume_fraction_within(large), torch.tensor([1.0]))


def test_volume_fraction_within_partial_targets():
    """Two targets can contain the same corners but very different object volumes."""
    object_bounds = AxisAlignedBoundingBox((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    targets = AxisAlignedBoundingBox(
        min_point=torch.zeros(2, 3),
        max_point=torch.tensor([[0.1, 1.0, 1.0], [0.9, 1.0, 1.0]]),
    )
    # Both targets contain the four x=0 corners, but overlap different volumes.
    torch.testing.assert_close(object_bounds.volume_fraction_within(targets), torch.tensor([0.1, 0.9]))


def test_translated_by_tuple_keeps_cuda_device():
    """A tuple offset is created on the box's device, so CUDA boxes translate without a device mismatch."""
    aabb = AxisAlignedBoundingBox(min_point=(0.1, 0.2, 0.3), max_point=(1.0, 2.0, 3.0)).to("cuda")
    moved = aabb.translated((1.0, 2.0, 3.0))
    offset = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    torch.testing.assert_close(moved.min_point, aabb.min_point + offset)
    torch.testing.assert_close(moved.max_point, aabb.max_point + offset)


def test_rotated_corners_and_enclosing_bounds():
    """An asymmetric 3D rotation gives corners and enclosing bounds independent of quaternion helpers."""
    quaternion = tuple(component / math.sqrt(30) for component in (1, 2, 3, 4))
    box = OrientedBoundingBox((1.0, 2.0, 3.0), (2.0, 1.0, 0.5), quaternion)
    R_W_B = torch.tensor([[2 / 15, -2 / 3, 11 / 15], [14 / 15, 1 / 3, 2 / 15], [-1 / 3, 2 / 3, 2 / 3]])
    corners_B = _CORNER_SIGNS * torch.tensor([2.0, 1.0, 0.5])
    expected = (corners_B @ R_W_B.T + torch.tensor([1.0, 2.0, 3.0])).unsqueeze(0)
    torch.testing.assert_close(box.get_corners_at(), expected)
    torch.testing.assert_close(box.get_corners_at(torch.tensor([2.0, 0.0, 0.0])), expected + torch.tensor([2.0, 0, 0]))
    enclosing = box.to_aabb()
    torch.testing.assert_close(enclosing.min_point, expected.amin(dim=1))
    torch.testing.assert_close(enclosing.max_point, expected.amax(dim=1))


def test_transform_composes_rotations_and_rotates_the_center():
    """A quarter-turn about X followed by Z maps local (x, y, z) to (z, x, y)."""
    s = math.sqrt(0.5)
    box = OrientedBoundingBox((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (s, 0.0, 0.0, s))
    transformed = box.transformed((4.0, 5.0, 6.0), (0.0, 0.0, s, s))
    torch.testing.assert_close(transformed.center_W, torch.tensor([[2.0, 6.0, 9.0]]))
    torch.testing.assert_close(transformed.rotation_W_B_xyzw, torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
    lower, upper = transformed.get_axis_aligned_bounds()
    torch.testing.assert_close(lower, torch.tensor([[-1.0, 5.0, 7.0]]))
    torch.testing.assert_close(upper, torch.tensor([[5.0, 7.0, 11.0]]))


def test_aabb_transform_matches_enclosing_rotation():
    """An exactly transformed AABB has the same axis-aligned bounds as rotated_by_quat."""
    box = AxisAlignedBoundingBox((-2.0, -1.0, -0.5), (2.0, 1.0, 0.5))
    exact = box.transformed((0.0, 0.0, 0.0), _yaw(math.pi / 4))
    enclosing = box.rotated_by_quat(_yaw(math.pi / 4))
    assert type(exact) is OrientedBoundingBox
    lower, upper = exact.get_axis_aligned_bounds()
    torch.testing.assert_close(enclosing.min_point, lower)
    torch.testing.assert_close(enclosing.max_point, upper)


def test_sat_separates_boxes_whose_six_face_axes_overlap():
    """Skew rods separate on an edge-cross-edge axis, despite all face projections overlapping."""
    first = OrientedBoundingBox((0.0, 0.0, 0.0), (2.0, 0.1, 0.1))
    second = OrientedBoundingBox(
        (0.0, 0.2612789059, 0.1847520861),
        (2.0, 0.1, 0.1),
        (-0.1464466094, 0.3535533906, 0.3535533906, 0.8535533906),
    )
    s = math.sqrt(0.5)
    face_axes = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, -s], [-s, s, 0], [0.5, 0.5, s]])
    first_projections = first.get_corners_at()[0] @ face_axes.T
    second_projections = second.get_corners_at()[0] @ face_axes.T
    assert (first_projections.amax(dim=0) > second_projections.amin(dim=0)).all()
    assert (second_projections.amax(dim=0) > first_projections.amin(dim=0)).all()

    assert not first.overlaps(second).item()
    assert not second.overlaps(first).item()
    assert not first.overlaps(second, margin=0.04).item()
    assert first.overlaps(second, margin=0.05).item()

    separating_direction = torch.tensor([0.0, math.sqrt(2 / 3), 1 / math.sqrt(3)])
    overlapping = second.translated(-0.08 * separating_direction)
    depth = 0.2 * (math.sqrt(2 / 3) + 1 / math.sqrt(3)) - 0.24
    assert first.overlaps(overlapping.translated((depth - 1e-4) * separating_direction)).item()
    assert not first.overlaps(overlapping.translated((depth + 1e-4) * separating_direction)).item()


def test_nearly_parallel_thin_boxes_do_not_use_their_enclosing_aabbs():
    """Rods with distinct tiny yaws separate despite their enclosing AABBs overlapping."""
    first = OrientedBoundingBox((0.0, 0.0, 0.0), (100.0, 0.001, 1.0), _yaw(5e-6))
    second = OrientedBoundingBox((0.0, 0.0023, 0.0), (100.0, 0.001, 1.0), _yaw(6e-6))
    assert first.to_aabb().overlaps(second.to_aabb()).item()
    assert not first.overlaps(second).item()
    assert not second.overlaps(first).item()


def test_mixed_box_overlap_is_symmetric_with_strict_margins():
    """AABB/OBB pairs use oriented geometry both ways, and touching at the requested margin is allowed."""
    aabb = AxisAlignedBoundingBox((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    touching = OrientedBoundingBox((2.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    separated = touching.translated((0.125, 0.0, 0.0))
    for left, right in ((aabb, touching), (touching, aabb)):
        assert not left.overlaps(right).item()
        assert left.overlaps(right, margin=0.125).item()
    for left, right in ((aabb, separated), (separated, aabb)):
        assert not left.overlaps(right, margin=0.125).item()
        assert left.overlaps(right, margin=0.25).item()
    with pytest.raises(AssertionError, match="Margin"):
        touching.overlaps(aabb, margin=-0.125)

    small = AxisAlignedBoundingBox((-0.1, 0.3, -0.05), (0.1, 0.5, 0.05))
    rod = OrientedBoundingBox((0.0, 0.0, 0.0), (1.0, 0.05, 0.05), _yaw(math.pi / 4))
    assert small.overlaps(rod.to_aabb()).item()
    assert not small.overlaps(rod).item()
    assert not rod.overlaps(small).item()
    assert small.translated((0.0, -0.4, 0.0)).overlaps(rod).item()


def test_box_batches_keep_rows_through_queries_and_transforms():
    """Singleton extents broadcast without mixing different rows' centers or rotations."""
    s = math.sqrt(0.5)
    boxes = OrientedBoundingBox(
        torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=torch.float64),
        (1.0, 2.0, 0.5),
        torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, s, s]]),
    )
    assert boxes.num_envs == 2 and boxes.center_W.dtype == torch.float32
    target = OrientedBoundingBox((2.5, 0.0, 0.0), (0.1, 0.1, 0.1))
    assert boxes.overlaps(target).tolist() == [False, True]
    assert target.overlaps(boxes).tolist() == [False, True]

    moved = boxes.transformed(
        torch.tensor([[1.0, 0.0, 0.0], [-1.0, 2.0, 0.0]]),
        torch.tensor([[0.0, 0.0, s, s], [s, 0.0, 0.0, s]]),
    )
    torch.testing.assert_close(moved.center_W, torch.tensor([[1.0, 0.0, 0.0], [3.0, 2.0, 0.0]]))
    torch.testing.assert_close(moved.rotation_W_B_xyzw, torch.tensor([[0.0, 0.0, s, s], [0.5, -0.5, 0.5, 0.5]]))
    torch.testing.assert_close(moved[1].center_W, torch.tensor([[3.0, 2.0, 0.0]]))
    with pytest.raises(AssertionError, match="row counts"):
        OrientedBoundingBox(torch.zeros(2, 3), torch.ones(3, 3))


def test_aabb_empty_and_unbounded_bounds_are_rejected_by_finite_queries():
    """Empty and infinite AABBs keep interval overlaps but are rejected wherever finite geometry is needed."""
    bounded = AxisAlignedBoundingBox((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    empty = bounded.intersected(AxisAlignedBoundingBox((2.0, 0.0, 0.0), (3.0, 1.0, 1.0)))
    unbounded = AxisAlignedBoundingBox((-math.inf,) * 3, (math.inf,) * 3)
    assert not bounded.overlaps(empty).item()
    assert bounded.overlaps(unbounded).item()
    oriented = bounded.to_obb()
    for box in (empty, unbounded):
        torch.testing.assert_close(box.to_aabb().min_point, box.min_point, rtol=0, atol=0)
        finite_queries = (
            lambda: box.to_obb(),
            lambda: box.transformed((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            lambda: box.overlap_volume_upper_bound(bounded),
            lambda: bounded.overlap_volume_upper_bound(box),
            lambda: box.overlaps(oriented),
            lambda: oriented.overlaps(box),
        )
        for query in finite_queries:
            with pytest.raises(AssertionError):
                query()


def test_aabb_queries_read_current_bounds_and_conversions_snapshot_them():
    """Shared queries follow edits to AABB bounds, while to_obb() keeps the bounds at conversion time."""
    box = AxisAlignedBoundingBox((0.0, 0.0, 0.0), (2.0, 4.0, 6.0))
    other = AxisAlignedBoundingBox((1.5, 0.0, 0.0), (3.5, 4.0, 6.0)).to_obb()
    snapshot = box.to_obb()
    box.min_point[0, 0] = 0.5
    box.max_point[0, :2] = torch.tensor([3.0, 6.0])
    torch.testing.assert_close(box.overlap_volume_upper_bound(other), torch.tensor([36.0]))
    torch.testing.assert_close(snapshot.overlap_volume_upper_bound(other), torch.tensor([12.0]))
    moved = box.transformed((4.0, 5.0, 6.0), _yaw(math.pi / 2))
    torch.testing.assert_close(moved.center_W, torch.tensor([[1.0, 6.75, 9.0]]))


def test_constructor_rejects_invalid_geometry():
    """Non-finite bounds, negative extents and non-unit quaternions cannot describe an OBB."""
    invalid = [
        ((math.nan, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (math.inf, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (-1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 2.0)),
    ]
    for center, half_extents, rotation in invalid:
        with pytest.raises(AssertionError):
            OrientedBoundingBox(center, half_extents, rotation)


def test_box_copies_inputs_and_returned_properties():
    """Editing input tensors or returned properties leaves a box's geometry unchanged."""
    centers = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    extents = torch.ones(1, 3)
    box = OrientedBoundingBox(centers, extents)
    corners = box.get_corners_at()
    centers.add_(10)
    extents.zero_()
    box.center_W.add_(10)
    box.half_extents.zero_()
    box.rotation_W_B_xyzw.zero_()
    torch.testing.assert_close(box.get_corners_at(), corners)


def test_rounded_quaternions_are_normalized():
    """Rounded authored quarter-turns give the same geometry as unit quaternions."""
    rounded = (0.0, 0.0, 0.707, 0.707)
    exact = _yaw(math.pi / 2)
    box = OrientedBoundingBox((1.0, 2.0, 3.0), (1.0, 2.0, 0.5), rounded)
    reference = OrientedBoundingBox((1.0, 2.0, 3.0), (1.0, 2.0, 0.5), exact)
    torch.testing.assert_close(box.get_corners_at(), reference.get_corners_at())
    torch.testing.assert_close(
        box.transformed((0.0, 0.0, 0.0), rounded).get_corners_at(),
        reference.transformed((0.0, 0.0, 0.0), exact).get_corners_at(),
    )


def test_box_queries_match_on_cuda_and_reject_mixed_devices():
    """CUDA boxes give the CPU results, and inputs on different devices are rejected."""
    boxes = OrientedBoundingBox(
        torch.tensor([[0.0, 0.0, 0.0], [3.0, 2.0, 1.0]]),
        (1.0, 2.0, 0.5),
        torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5]]),
    )
    target = OrientedBoundingBox((0.75, 0.0, 0.0), (0.1, 0.1, 0.1))
    aabb = AxisAlignedBoundingBox((0.1, 0.2, 0.3), (1.0, 2.0, 3.0))
    gpu_boxes, gpu_target, gpu_aabb = boxes.to("cuda"), target.to("cuda"), aabb.to("cuda")
    torch.testing.assert_close(gpu_boxes.get_corners_at().cpu(), boxes.get_corners_at())
    torch.testing.assert_close(gpu_boxes.overlaps(gpu_target).cpu(), boxes.overlaps(target))
    torch.testing.assert_close(
        gpu_boxes.overlap_volume_upper_bound(gpu_target).cpu(), boxes.overlap_volume_upper_bound(target)
    )
    torch.testing.assert_close(
        gpu_aabb.overlap_volume_upper_bound(gpu_boxes).cpu(), aabb.overlap_volume_upper_bound(boxes)
    )
    with pytest.raises(AssertionError, match="device"):
        gpu_boxes.overlaps(target)
    with pytest.raises(AssertionError, match="device"):
        OrientedBoundingBox(torch.zeros(3, device="cuda"), torch.ones(3))


def test_overlap_bound_matches_aabb_value_and_gradients_at_quarter_turns():
    """Axis permutations give the interval-intersection volume and gradients of their axis-aligned boxes."""
    s = math.sqrt(0.5)
    centers = torch.tensor([[0.9, 0.1, 0.2], [0.4, 0.2, 0.1], [0.6, 0.1, 0.2], [0.3, 0.2, 0.1]], requires_grad=True)
    boxes = OrientedBoundingBox(
        centers,
        (1.0, 0.25, 0.5),
        torch.tensor([[0, 0, 0, 1], [0, 0, s, s], [0.5, 0.5, 0.5, 0.5], [s, s, 0, 0]]),
    )
    target = AxisAlignedBoundingBox((-0.3, -0.4, -0.2), (0.3, 0.4, 0.2))
    world_extents = torch.tensor([[1, 0.25, 0.5], [0.25, 1, 0.5], [0.5, 1, 0.25], [0.25, 1, 0.5]])
    upper = torch.minimum(centers + world_extents, target.max_point + 0.1)
    lower = torch.maximum(centers - world_extents, target.min_point - 0.1)
    expected = torch.relu(upper - lower).prod(dim=-1)
    actual = boxes.overlap_volume_upper_bound(target, margin=0.1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), centers)[0], torch.autograd.grad(expected.sum(), centers)[0]
    )
    torch.testing.assert_close(boxes.overlap_volume_upper_bound(target), target.overlap_volume_upper_bound(boxes))


def test_overlap_bound_matches_corner_projections_and_gradients():
    """Tilted batches equal the smaller face-axis volume computed from independently projected corners."""
    s = math.sqrt(0.5)
    R_W_A = torch.tensor([[2 / 15, -2 / 3, 11 / 15], [14 / 15, 1 / 3, 2 / 15], [-1 / 3, 2 / 3, 2 / 3]])
    R_W_B = torch.tensor([[s, -s, 0], [s, s, 0], [0, 0, 1]])
    centers_a = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.4, 0.1], [0.2, 0.1, 0.0]], requires_grad=True)
    centers_b = torch.tensor([[0.3, 0.1, 0.2], [0.7, -0.3, 0.4], [4.0, 0.0, 0.0]])
    half_a = torch.tensor([0.7, 0.3, 0.4])
    half_b = torch.tensor([0.3, 0.5, 0.2])
    first = OrientedBoundingBox(centers_a, half_a, tuple(component / math.sqrt(30) for component in (1, 2, 3, 4)))
    second = OrientedBoundingBox(centers_b, half_b, _yaw(math.pi / 4))

    corners_a = (half_a * _CORNER_SIGNS) @ R_W_A.T + centers_a.unsqueeze(-2)
    corners_b = (half_b * _CORNER_SIGNS) @ R_W_B.T + centers_b.unsqueeze(-2)
    volumes = []
    for axes in (R_W_A, R_W_B):
        projections_a, projections_b = corners_a @ axes, corners_b @ axes
        upper = torch.minimum(projections_a.amax(dim=-2), projections_b.amax(dim=-2) + 0.05)
        lower = torch.maximum(projections_a.amin(dim=-2), projections_b.amin(dim=-2) - 0.05)
        volumes.append(torch.relu(upper - lower).prod(dim=-1))
    expected = torch.minimum(*volumes)
    actual = first.overlap_volume_upper_bound(second, margin=0.05)
    assert (expected[:2] > 0).all() and expected[2] == 0
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), centers_a)[0], torch.autograd.grad(expected.sum(), centers_a)[0]
    )


def test_overlap_bound_margin_inflates_only_the_other_box():
    """Margin inflates other's projections, turning touching into overlap; invalid margins are rejected."""
    outer = OrientedBoundingBox((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    inner = OrientedBoundingBox((0.5, 0.1, 0.2), (0.25, 0.5, 0.75))
    torch.testing.assert_close(
        outer.overlap_volume_upper_bound(inner, margin=0.125), torch.tensor([0.75 * 1.25 * 1.75])
    )
    torch.testing.assert_close(inner.overlap_volume_upper_bound(outer, margin=0.125), torch.tensor([0.75]))
    for margin in (-0.1, math.nan, math.inf):
        with pytest.raises(AssertionError, match="Margin"):
            outer.overlap_volume_upper_bound(inner, margin=margin)

    first = OrientedBoundingBox((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    second = OrientedBoundingBox((2.125, 0.0, 0.0), (1.0, 1.0, 1.0))
    torch.testing.assert_close(first.overlap_volume_upper_bound(second, margin=0.125), torch.zeros(1))
    torch.testing.assert_close(first.overlap_volume_upper_bound(second, margin=0.25), torch.tensor([0.5]))


def test_overlap_bound_gradient_pushes_hardest_along_the_shallowest_overlap():
    """Each face axis is pushed by volume over overlap length; contained axes and boxes get no push."""
    angle = math.pi / 6
    R_W_A = torch.tensor([[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0, 0, 1]])
    obstacle = OrientedBoundingBox((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), _yaw(angle))
    # In the obstacle frame, overlaps are 0.2 along x, 1.5 along y, and z is contained (length 1.0).
    center = (R_W_A @ torch.tensor([1.8, 0.5, 0.1])).requires_grad_()
    subject = OrientedBoundingBox(center, (1.0, 1.0, 0.5), _yaw(angle))
    volume = obstacle.overlap_volume_upper_bound(subject)
    torch.testing.assert_close(volume, torch.tensor([0.2 * 1.5 * 1.0]))
    torch.testing.assert_close(torch.autograd.grad(volume.sum(), center)[0], R_W_A @ torch.tensor([-1.5, -0.2, 0.0]))

    contained_center = torch.tensor([0.2, 0.1, 0.0], requires_grad=True)
    contained = OrientedBoundingBox(contained_center, (0.25, 0.25, 0.25), _yaw(angle))
    contained_volume = obstacle.overlap_volume_upper_bound(contained)
    torch.testing.assert_close(contained_volume, torch.tensor([0.125]))
    torch.testing.assert_close(torch.autograd.grad(contained_volume.sum(), contained_center)[0], torch.zeros(3))


def test_overlap_bound_is_positive_exactly_when_yaw_only_boxes_overlap():
    """Face axes complete the separating-axis test for boxes rotated only about Z."""
    generator = torch.Generator().manual_seed(0)
    count = 256
    yaws = torch.rand(2, count, generator=generator) * 2 * math.pi
    rotations = torch.zeros(2, count, 4)
    rotations[..., 2] = torch.sin(yaws / 2)
    rotations[..., 3] = torch.cos(yaws / 2)
    centers = torch.rand(2, count, 3, generator=generator) * 2 - 1
    half_extents = torch.rand(2, count, 3, generator=generator) * 0.6 + 0.05
    first = OrientedBoundingBox(centers[0], half_extents[0], rotations[0])
    second = OrientedBoundingBox(centers[1], half_extents[1], rotations[1])
    for margin in (0.0, 0.05):
        overlaps = first.overlaps(second, margin=margin)
        assert 0 < overlaps.sum() < count
        assert torch.equal(first.overlap_volume_upper_bound(second, margin=margin) > 0, overlaps)


def test_overlap_bound_of_touching_aabbs_is_exactly_zero():
    """AABB pairs read min/max bounds, so touching faces give exactly zero despite float32 centers."""
    generator = torch.Generator().manual_seed(0)
    first_min = torch.rand(1000, 3, generator=generator)
    first_max = first_min + torch.rand(1000, 3, generator=generator) + 0.01
    second_min = torch.rand(1000, 3, generator=generator)
    second_min[:, 0] = first_max[:, 0]
    second_max = second_min + torch.rand(1000, 3, generator=generator) + 0.01
    first = AxisAlignedBoundingBox(first_min, first_max)
    second = AxisAlignedBoundingBox(second_min, second_max)
    assert not first.overlaps(second).any()
    assert torch.equal(first.overlap_volume_upper_bound(second), torch.zeros(1000))
    assert torch.equal(second.overlap_volume_upper_bound(first), torch.zeros(1000))
    torch.testing.assert_close(
        first.overlap_volume_upper_bound(second, margin=0.1),
        first.to_obb().overlap_volume_upper_bound(second.to_obb(), margin=0.1),
    )


def _yaw(angle: float) -> tuple[float, float, float, float]:
    """Return the xyzw quaternion for a rotation of angle radians about Z."""
    return (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))
