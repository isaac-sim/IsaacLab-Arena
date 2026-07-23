# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for AxisAlignedBoundingBox with always-tensor API."""

import math
import torch

import pytest

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


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


def test_union_encloses_all_boxes():
    """Union spans the component-wise min/max of disjoint boxes."""
    a = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 1.0))
    b = AxisAlignedBoundingBox(min_point=(2.0, -1.0, 0.0), max_point=(3.0, 0.0, 2.0))
    result = AxisAlignedBoundingBox.union([a, b])
    torch.testing.assert_close(result.min_point, torch.tensor([[0.0, -1.0, 0.0]]))
    torch.testing.assert_close(result.max_point, torch.tensor([[3.0, 1.0, 2.0]]))


def test_union_broadcasts_single_box_over_batch():
    """An N=1 box broadcasts against a batched box, yielding a per-env union."""
    batched = AxisAlignedBoundingBox(
        min_point=torch.tensor([[0.0, 0.0, 0.0], [5.0, 5.0, 0.0]]),
        max_point=torch.tensor([[1.0, 1.0, 1.0], [6.0, 6.0, 1.0]]),
    )
    single = AxisAlignedBoundingBox(min_point=(-1.0, -1.0, -1.0), max_point=(0.0, 0.0, 0.0))
    result = AxisAlignedBoundingBox.union([batched, single])
    assert result.num_envs == 2
    # The single box broadcasts into both rows, so check each independently.
    torch.testing.assert_close(result.min_point[0], torch.tensor([-1.0, -1.0, -1.0]))
    torch.testing.assert_close(result.max_point[0], torch.tensor([1.0, 1.0, 1.0]))
    torch.testing.assert_close(result.min_point[1], torch.tensor([-1.0, -1.0, -1.0]))
    torch.testing.assert_close(result.max_point[1], torch.tensor([6.0, 6.0, 1.0]))


def test_union_rejects_mismatched_batch_sizes():
    """Boxes with differing num_envs (neither N=1) are ambiguous and assert."""
    two = AxisAlignedBoundingBox(
        min_point=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        max_point=torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    )
    three = AxisAlignedBoundingBox(
        min_point=torch.zeros(3, 3),
        max_point=torch.ones(3, 3),
    )
    with pytest.raises(AssertionError):
        AxisAlignedBoundingBox.union([two, three])


def test_union_rejects_empty_sequence():
    """An empty sequence has no bounds to union and must assert, not raise a bare ValueError."""
    with pytest.raises(AssertionError):
        AxisAlignedBoundingBox.union([])


def test_union_combines_two_per_env_boxes():
    """Two genuinely per-env N=2 boxes union independently per row (no broadcasting)."""
    a = AxisAlignedBoundingBox(
        min_point=torch.tensor([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]),
        max_point=torch.tensor([[1.0, 1.0, 1.0], [6.0, 6.0, 6.0]]),
    )
    b = AxisAlignedBoundingBox(
        min_point=torch.tensor([[-1.0, 0.0, 0.0], [5.0, 4.0, 5.0]]),
        max_point=torch.tensor([[1.0, 2.0, 1.0], [7.0, 6.0, 6.0]]),
    )
    result = AxisAlignedBoundingBox.union([a, b])
    assert result.num_envs == 2
    torch.testing.assert_close(result.min_point, torch.tensor([[-1.0, 0.0, 0.0], [5.0, 4.0, 5.0]]))
    torch.testing.assert_close(result.max_point, torch.tensor([[1.0, 2.0, 1.0], [7.0, 6.0, 6.0]]))


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
