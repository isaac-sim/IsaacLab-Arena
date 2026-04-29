# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for AxisAlignedBoundingBox with always-tensor API."""

import torch

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
