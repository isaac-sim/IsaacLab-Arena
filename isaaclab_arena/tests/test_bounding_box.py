# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal tests for BatchedAxisAlignedBoundingBox and world_bbox_to_min_max_tensors."""

import torch

from isaaclab_arena.utils.bounding_box import (
    AxisAlignedBoundingBox,
    BatchedAxisAlignedBoundingBox,
    world_bbox_to_min_max_tensors,
)


def test_batched_aabb_properties_and_overlaps():
    """BatchedAxisAlignedBoundingBox: shape (N,3) for size/center; overlaps returns (N,) bool."""
    min_c = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
    max_c = torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 1.0]])
    batched = BatchedAxisAlignedBoundingBox(min_corner=min_c, max_corner=max_c)
    assert batched.size.shape == (2, 3)
    assert batched.center.shape == (2, 3)
    other = AxisAlignedBoundingBox(min_point=(0.5, 0.5, 0.0), max_point=(1.5, 1.5, 0.5))
    result = batched.overlaps(other, margin=0.0)
    assert result.shape == (2,)
    assert result[0].item() is True
    assert result[1].item() is False


def test_world_bbox_to_min_max_tensors():
    """world_bbox_to_min_max_tensors: AxisAligned -> (1,3), Batched -> (N,3)."""
    single = AxisAlignedBoundingBox(min_point=(1.0, 2.0, 0.0), max_point=(3.0, 4.0, 0.5))
    min_c, max_c = world_bbox_to_min_max_tensors(single, torch.device("cpu"), torch.float32)
    assert min_c.shape == (1, 3)
    assert max_c.shape == (1, 3)
    torch.testing.assert_close(min_c, torch.tensor([[1.0, 2.0, 0.0]]))

    batched = BatchedAxisAlignedBoundingBox(
        min_corner=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        max_corner=torch.tensor([[1.0, 1.0, 0.2], [2.0, 2.0, 0.3]]),
    )
    min_c, max_c = world_bbox_to_min_max_tensors(batched, torch.device("cpu"), torch.float32)
    assert min_c.shape == (2, 3)
    assert max_c.shape == (2, 3)
