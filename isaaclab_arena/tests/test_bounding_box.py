# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for BatchedAxisAlignedBoundingBox and world_bbox_to_min_max_tensors."""

import torch

from isaaclab_arena.utils.bounding_box import (
    AxisAlignedBoundingBox,
    BatchedAxisAlignedBoundingBox,
    world_bbox_to_min_max_tensors,
)


def _create_batched_aabb(min_corner: torch.Tensor, max_corner: torch.Tensor) -> BatchedAxisAlignedBoundingBox:
    """Create a batched AABB (min_corner/max_corner shape (N, 3))."""
    return BatchedAxisAlignedBoundingBox(min_corner=min_corner, max_corner=max_corner)


# =============================================================================
# BatchedAxisAlignedBoundingBox tests
# =============================================================================


def test_batched_aabb_size():
    """Test that batched size has shape (N, 3) and equals max - min."""
    batched = _create_batched_aabb(
        min_corner=torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]]),
        max_corner=torch.tensor([[1.0, 1.0, 1.0], [3.0, 4.0, 0.5]]),
    )
    size = batched.size
    assert size.shape == (2, 3)
    torch.testing.assert_close(size[0], torch.tensor([1.0, 1.0, 1.0]))
    torch.testing.assert_close(size[1], torch.tensor([2.0, 2.0, 0.5]))


def test_batched_aabb_center():
    """Test that batched center has shape (N, 3) and equals (min + max) / 2."""
    batched = _create_batched_aabb(
        min_corner=torch.tensor([[0.0, 0.0, 0.0]]),
        max_corner=torch.tensor([[2.0, 4.0, 0.2]]),
    )
    center = batched.center
    assert center.shape == (1, 3)
    torch.testing.assert_close(center[0], torch.tensor([1.0, 2.0, 0.1]))


def test_batched_aabb_top_bottom_surface_z():
    """Test that batched top_surface_z and bottom_surface_z have shape (N,) and match max/min z."""
    batched = _create_batched_aabb(
        min_corner=torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.5]]),
        max_corner=torch.tensor([[1.0, 1.0, 0.4], [1.0, 1.0, 0.8]]),
    )
    assert batched.top_surface_z.shape == (2,)
    assert batched.bottom_surface_z.shape == (2,)
    torch.testing.assert_close(batched.top_surface_z, torch.tensor([0.4, 0.8]))
    torch.testing.assert_close(batched.bottom_surface_z, torch.tensor([0.1, 0.5]))


def test_batched_aabb_translated():
    """Test that batched translated() applies offset and returns new batched bbox."""
    batched = _create_batched_aabb(
        min_corner=torch.tensor([[0.0, 0.0, 0.0]]),
        max_corner=torch.tensor([[1.0, 1.0, 1.0]]),
    )
    moved = batched.translated((1.0, 2.0, 0.5))
    torch.testing.assert_close(moved.min_corner[0], torch.tensor([1.0, 2.0, 0.5]))
    torch.testing.assert_close(moved.max_corner[0], torch.tensor([2.0, 3.0, 1.5]))


def test_batched_aabb_overlaps_single_bbox():
    """Test that batched overlaps() with AxisAlignedBoundingBox returns (N,) bool; one env overlaps, one does not."""
    batched = _create_batched_aabb(
        min_corner=torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]),
        max_corner=torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 1.0]]),
    )
    other = AxisAlignedBoundingBox(min_point=(0.5, 0.5, 0.0), max_point=(1.5, 1.5, 0.5))
    result = batched.overlaps(other, margin=0.0)
    assert result.shape == (2,)
    assert result[0].item() is True
    assert result[1].item() is False


def test_batched_aabb_get_corners_at():
    """Test that batched get_corners_at() returns shape (N, 8, 3); with pos, corners are offset."""
    batched = _create_batched_aabb(
        min_corner=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        max_corner=torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    )
    corners = batched.get_corners_at(pos=None)
    assert corners.shape == (2, 8, 3)
    # With pos, corners are offset per env
    corners_with_pos = batched.get_corners_at(pos=torch.tensor([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]))
    assert corners_with_pos.shape == (2, 8, 3)
    torch.testing.assert_close(corners_with_pos[0, 0], corners[0, 0] + torch.tensor([10.0, 0.0, 0.0]))


# =============================================================================
# world_bbox_to_min_max_tensors tests
# =============================================================================


def test_world_bbox_to_min_max_tensors_single():
    """Test that world_bbox_to_min_max_tensors with AxisAlignedBoundingBox returns (1, 3) tensors."""
    single = AxisAlignedBoundingBox(min_point=(1.0, 2.0, 0.0), max_point=(3.0, 4.0, 0.5))
    min_c, max_c = world_bbox_to_min_max_tensors(single, torch.device("cpu"), torch.float32)
    assert min_c.shape == (1, 3)
    assert max_c.shape == (1, 3)
    torch.testing.assert_close(min_c, torch.tensor([[1.0, 2.0, 0.0]]))
    torch.testing.assert_close(max_c, torch.tensor([[3.0, 4.0, 0.5]]))


def test_world_bbox_to_min_max_tensors_batched():
    """Test that world_bbox_to_min_max_tensors with BatchedAxisAlignedBoundingBox returns (N, 3) tensors."""
    batched = _create_batched_aabb(
        min_corner=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        max_corner=torch.tensor([[1.0, 1.0, 0.2], [2.0, 2.0, 0.3]]),
    )
    min_c, max_c = world_bbox_to_min_max_tensors(batched, torch.device("cpu"), torch.float32)
    assert min_c.shape == (2, 3)
    assert max_c.shape == (2, 3)
    torch.testing.assert_close(min_c, batched.min_corner)
    torch.testing.assert_close(max_c, batched.max_corner)
