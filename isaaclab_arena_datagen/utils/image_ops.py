# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Resize and intrinsic-scaling helpers for storing camera data below render resolution.

All resolutions are ``(width, height)`` tuples to match the JSON config convention.
Tensors keep their input device and the spatial layout used by the writer (``(H, W, ...)``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def assert_within_render(target_wh: tuple[int, int], render_wh: tuple[int, int], name: str) -> None:
    """Assert a stored resolution does not exceed the render resolution."""
    tw, th = target_wh
    rw, rh = render_wh
    assert tw <= rw and th <= rh, f"{name} resolution {target_wh} exceeds render resolution {render_wh}"


def _is_identity(hw: torch.Tensor, target_wh: tuple[int, int]) -> bool:
    """Return True when the tensor already has the target (width, height)."""
    tw, th = target_wh
    return hw.shape[0] == th and hw.shape[1] == tw


def resize_color(rgb_hw3: torch.Tensor, target_wh: tuple[int, int]) -> torch.Tensor:
    """Downscale an (H, W, 3) uint8 image to target (width, height) with area averaging."""
    if _is_identity(rgb_hw3, target_wh):
        return rgb_hw3
    tw, th = target_wh
    chw = rgb_hw3.permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(chw, size=(th, tw), mode="area")
    return resized.round().clamp_(0, 255).to(torch.uint8).squeeze(0).permute(1, 2, 0)


def resize_depth(depth_hw: torch.Tensor, target_wh: tuple[int, int]) -> torch.Tensor:
    """Downscale an (H, W) depth map to target (width, height) using nearest sampling.

    Nearest avoids blending foreground and background depths into invalid in-between values.
    """
    if _is_identity(depth_hw, target_wh):
        return depth_hw
    tw, th = target_wh
    resized = F.interpolate(depth_hw.unsqueeze(0).unsqueeze(0).float(), size=(th, tw), mode="nearest")
    return resized.squeeze(0).squeeze(0)


def resize_label_map(ids_hw: torch.Tensor, target_wh: tuple[int, int]) -> torch.Tensor:
    """Downscale an (H, W) integer ID map to target (width, height) using nearest sampling.

    Nearest is required so labels stay exact and no spurious in-between IDs are produced.
    """
    if _is_identity(ids_hw, target_wh):
        return ids_hw
    tw, th = target_wh
    resized = F.interpolate(ids_hw.unsqueeze(0).unsqueeze(0).float(), size=(th, tw), mode="nearest")
    return resized.squeeze(0).squeeze(0).to(ids_hw.dtype)


def resize_flow2d(flow_hw2: torch.Tensor, target_wh: tuple[int, int]) -> torch.Tensor:
    """Downscale an (H, W, 2) pixel-flow field and rescale its vectors to the new resolution."""
    if _is_identity(flow_hw2, target_wh):
        return flow_hw2
    src_h, src_w = flow_hw2.shape[0], flow_hw2.shape[1]
    tw, th = target_wh
    chw = flow_hw2.permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(chw, size=(th, tw), mode="bilinear", align_corners=False)
    resized = resized.squeeze(0).permute(1, 2, 0).clone()
    resized[..., 0] *= tw / src_w
    resized[..., 1] *= th / src_h
    return resized


def scale_intrinsics(K_33: torch.Tensor, src_wh: tuple[int, int], target_wh: tuple[int, int]) -> torch.Tensor:
    """Scale a 3x3 intrinsic matrix from src (width, height) to target (width, height)."""
    sw, sh = src_wh
    tw, th = target_wh
    sx, sy = tw / sw, th / sh
    scaled = K_33.clone()
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled
