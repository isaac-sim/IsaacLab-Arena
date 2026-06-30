# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the resize and intrinsic-scaling helpers. No Isaac Sim required."""

import torch

import pytest

from isaaclab_arena_datagen.utils import image_ops


def test_resize_color_changes_shape_keeps_uint8():
    rgb = torch.randint(0, 256, (4, 6, 3), dtype=torch.uint8)
    out = image_ops.resize_color(rgb, (3, 2))
    assert out.shape == (2, 3, 3)
    assert out.dtype == torch.uint8


def test_resize_identity_returns_input_unchanged():
    depth = torch.rand(4, 6)
    assert image_ops.resize_depth(depth, (6, 4)) is depth


def test_resize_label_map_preserves_ids_with_nearest():
    ids = torch.tensor([[1, 1, 2, 2, 3, 3]] * 4, dtype=torch.int32)
    out = image_ops.resize_label_map(ids, (3, 2))
    assert out.shape == (2, 3)
    assert out.dtype == torch.int32
    assert set(out.unique().tolist()).issubset({1, 2, 3})


def test_resize_flow2d_scales_vectors_by_resolution_ratio():
    flow = torch.empty(4, 6, 2)
    flow[..., 0] = 10.0  # dx
    flow[..., 1] = 20.0  # dy
    out = image_ops.resize_flow2d(flow, (3, 2))  # width 6->3, height 4->2
    assert out.shape == (2, 3, 2)
    assert torch.allclose(out[..., 0], torch.full((2, 3), 5.0))
    assert torch.allclose(out[..., 1], torch.full((2, 3), 10.0))


def test_scale_intrinsics_scales_focal_and_principal_point():
    K = torch.tensor([[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]])
    out = image_ops.scale_intrinsics(K, (640, 480), (320, 240))
    assert torch.allclose(out[0], torch.tensor([50.0, 0.0, 160.0]))
    assert torch.allclose(out[1], torch.tensor([0.0, 50.0, 120.0]))


def test_assert_within_render_rejects_upscaling():
    image_ops.assert_within_render((512, 384), (640, 480), "color")  # ok
    with pytest.raises(AssertionError):
        image_ops.assert_within_render((800, 384), (640, 480), "color")
