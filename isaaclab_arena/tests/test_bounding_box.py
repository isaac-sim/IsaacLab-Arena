# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for unified AxisAlignedBoundingBox (single and batched)."""

import torch

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


# =============================================================================
# Construction tests
# =============================================================================


def test_aabb_from_tuples():
    """Test that constructing from tuples yields (1, 3) tensors."""
    aabb = AxisAlignedBoundingBox(min_point=(1.0, 2.0, 0.0), max_point=(3.0, 4.0, 0.5))
    assert aabb.min_point.shape == (1, 3)
    assert aabb.max_point.shape == (1, 3)
    assert aabb.num_envs == 1
    torch.testing.assert_close(aabb.min_point, torch.tensor([[1.0, 2.0, 0.0]]))
    torch.testing.assert_close(aabb.max_point, torch.tensor([[3.0, 4.0, 0.5]]))


def test_aabb_from_1d_tensors():
    """Test that constructing from 1D tensors yields (1, 3) tensors."""
    aabb = AxisAlignedBoundingBox(min_point=torch.tensor([0.0, 0.0, 0.0]), max_point=torch.tensor([1.0, 1.0, 1.0]))
    assert aabb.min_point.shape == (1, 3)
    assert aabb.num_envs == 1


def test_aabb_from_2d_tensors():
    """Test that constructing from 2D (N, 3) tensors keeps shape."""
    aabb = AxisAlignedBoundingBox(
        min_point=torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]]),
        max_point=torch.tensor([[1.0, 1.0, 1.0], [3.0, 4.0, 0.5]]),
    )
    assert aabb.min_point.shape == (2, 3)
    assert aabb.num_envs == 2


# =============================================================================
# Property tests
# =============================================================================


def test_aabb_size():
    """Test that size has shape (N, 3) and equals max - min."""
    aabb = AxisAlignedBoundingBox(
        min_point=torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]]),
        max_point=torch.tensor([[1.0, 1.0, 1.0], [3.0, 4.0, 0.5]]),
    )
    size = aabb.size
    assert size.shape == (2, 3)
    torch.testing.assert_close(size[0], torch.tensor([1.0, 1.0, 1.0]))
    torch.testing.assert_close(size[1], torch.tensor([2.0, 2.0, 0.5]))


def test_aabb_center():
    """Test that center has shape (N, 3) and equals (min + max) / 2."""
    aabb = AxisAlignedBoundingBox(
        min_point=torch.tensor([[0.0, 0.0, 0.0]]),
        max_point=torch.tensor([[2.0, 4.0, 0.2]]),
    )
    center = aabb.center
    assert center.shape == (1, 3)
    torch.testing.assert_close(center[0], torch.tensor([1.0, 2.0, 0.1]))


def test_aabb_top_bottom_surface_z():
    """Test that top_surface_z and bottom_surface_z have shape (N,) and match max/min z."""
    aabb = AxisAlignedBoundingBox(
        min_point=torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.5]]),
        max_point=torch.tensor([[1.0, 1.0, 0.4], [1.0, 1.0, 0.8]]),
    )
    assert aabb.top_surface_z.shape == (2,)
    assert aabb.bottom_surface_z.shape == (2,)
    torch.testing.assert_close(aabb.top_surface_z, torch.tensor([0.4, 0.8]))
    torch.testing.assert_close(aabb.bottom_surface_z, torch.tensor([0.1, 0.5]))


# =============================================================================
# Method tests
# =============================================================================


def test_aabb_translated():
    """Test that translated() applies offset and returns new AABB."""
    aabb = AxisAlignedBoundingBox(
        min_point=torch.tensor([[0.0, 0.0, 0.0]]),
        max_point=torch.tensor([[1.0, 1.0, 1.0]]),
    )
    moved = aabb.translated((1.0, 2.0, 0.5))
    torch.testing.assert_close(moved.min_point[0], torch.tensor([1.0, 2.0, 0.5]))
    torch.testing.assert_close(moved.max_point[0], torch.tensor([2.0, 3.0, 1.5]))


def test_aabb_overlaps_n1_vs_n1():
    """Test that overlaps() between two N=1 AABBs returns (1,) bool tensor."""
    a = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 1.0))
    b = AxisAlignedBoundingBox(min_point=(0.5, 0.5, 0.0), max_point=(1.5, 1.5, 0.5))
    result = a.overlaps(b, margin=0.0)
    assert result.shape == (1,)
    assert result[0].item() is True


def test_aabb_overlaps_batched_vs_single():
    """Test that overlaps() with N=2 vs N=1 returns (2,) bool; one env overlaps, one does not."""
    batched = AxisAlignedBoundingBox(
        min_point=torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]),
        max_point=torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 1.0]]),
    )
    other = AxisAlignedBoundingBox(min_point=(0.5, 0.5, 0.0), max_point=(1.5, 1.5, 0.5))
    result = batched.overlaps(other, margin=0.0)
    assert result.shape == (2,)
    assert result[0].item() is True
    assert result[1].item() is False


def test_aabb_get_corners_at():
    """Test that get_corners_at() returns shape (N, 8, 3); with pos, corners are offset."""
    aabb = AxisAlignedBoundingBox(
        min_point=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        max_point=torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    )
    corners = aabb.get_corners_at(pos=None)
    assert corners.shape == (2, 8, 3)
    corners_with_pos = aabb.get_corners_at(pos=torch.tensor([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]))
    assert corners_with_pos.shape == (2, 8, 3)
    torch.testing.assert_close(corners_with_pos[0, 0], corners[0, 0] + torch.tensor([10.0, 0.0, 0.0]))


def test_aabb_rotated_90_around_z():
    """Test that 90-degree Z rotation swaps X/Y correctly."""
    aabb = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(2.0, 1.0, 0.5))
    rotated = aabb.rotated_90_around_z(1)
    torch.testing.assert_close(rotated.min_point, torch.tensor([[-1.0, 0.0, 0.0]]))
    torch.testing.assert_close(rotated.max_point, torch.tensor([[0.0, 2.0, 0.5]]))


def test_aabb_scaled():
    """Test that scaled() multiplies min/max by scale factors."""
    aabb = AxisAlignedBoundingBox(min_point=(1.0, 2.0, 0.0), max_point=(3.0, 4.0, 1.0))
    scaled = aabb.scaled((2.0, 0.5, 3.0))
    torch.testing.assert_close(scaled.min_point, torch.tensor([[2.0, 1.0, 0.0]]))
    torch.testing.assert_close(scaled.max_point, torch.tensor([[6.0, 2.0, 3.0]]))


def test_aabb_centered():
    """Test that centered() shifts the bbox so center is at origin."""
    aabb = AxisAlignedBoundingBox(min_point=(2.0, 4.0, 0.0), max_point=(4.0, 6.0, 2.0))
    c = aabb.centered()
    torch.testing.assert_close(c.center, torch.tensor([[0.0, 0.0, 0.0]]))
    torch.testing.assert_close(c.size, aabb.size)
