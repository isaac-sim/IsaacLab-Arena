# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for AxisAlignedBoundingBox with batch dimension support."""

import torch

import pytest

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

# =============================================================================
# Single-env backward-compatibility tests (properties return tuples / floats)
# =============================================================================


def test_bounding_box_single_env_properties():
    """Single env: properties return tuples/floats matching the original dataclass API."""
    aabb = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(2.0, 4.0, 0.2))
    assert aabb.num_envs == 1
    assert isinstance(aabb.min_point, tuple)
    assert aabb.min_point == (0.0, 0.0, 0.0)
    assert aabb.max_point == (2.0, 4.0, pytest.approx(0.2, abs=1e-6))
    assert aabb.size == (2.0, 4.0, pytest.approx(0.2, abs=1e-6))
    assert aabb.center == (1.0, 2.0, pytest.approx(0.1, abs=1e-6))
    assert aabb.top_surface_z == pytest.approx(0.2, abs=1e-6)
    assert isinstance(aabb.top_surface_z, float)


def test_bounding_box_single_env_transforms():
    """Single env: translated, scaled, centered, rotated return AABBs with correct values."""
    aabb = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(2.0, 1.0, 0.5))
    moved = aabb.translated((1.0, 2.0, 0.5))
    assert moved.min_point == (1.0, 2.0, 0.5)
    assert moved.max_point == (3.0, 3.0, 1.0)

    scaled = aabb.scaled((2.0, 0.5, 3.0))
    assert scaled.max_point == (4.0, 0.5, 1.5)

    centered = AxisAlignedBoundingBox(min_point=(2.0, 4.0, 0.0), max_point=(4.0, 6.0, 2.0)).centered()
    assert centered.center == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)

    rotated = aabb.rotated_90_around_z(1)
    assert rotated.min_point == pytest.approx((-1.0, 0.0, 0.0), abs=1e-6)
    assert rotated.max_point == pytest.approx((0.0, 2.0, 0.5), abs=1e-6)


def test_bounding_box_single_env_overlaps():
    """Single env: overlaps() returns a Python bool; margin widens the check."""
    a = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 1.0))
    b = AxisAlignedBoundingBox(min_point=(0.5, 0.5, 0.0), max_point=(1.5, 1.5, 0.5))
    assert a.overlaps(b) is True

    c = AxisAlignedBoundingBox(min_point=(1.05, 0.0, 0.0), max_point=(2.0, 1.0, 1.0))
    assert a.overlaps(c) is False
    assert a.overlaps(c, margin=0.1) is True


def test_bounding_box_single_env_get_corners_at():
    """Single env: get_corners_at() returns (8, 3) tensor offset by pos."""
    aabb = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 1.0))
    corners = aabb.get_corners_at()
    assert corners.shape == (8, 3)
    corners_offset = aabb.get_corners_at(pos=torch.tensor([10.0, 0.0, 0.0]))
    torch.testing.assert_close(corners_offset[0], corners[0] + torch.tensor([10.0, 0.0, 0.0]))


# =============================================================================
# Multi-env batch tests (properties return tensors)
# =============================================================================


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
