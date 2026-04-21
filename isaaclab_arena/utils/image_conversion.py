# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Image-conversion helpers shared by client and server pipelines.

The remote-policy ``obs_spatial_hints`` handshake (see
``DESIGN_CLIENT_OBS_SHAPE_HANDSHAKE_AND_RESIZE.md``) requires both the client
(packing) and the server (defensive fallback) to apply the *same* resize
implementation, so this module owns the canonical implementation.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch


def _to_numpy(frames: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(frames, torch.Tensor):
        return frames.detach().cpu().numpy()
    if isinstance(frames, np.ndarray):
        return frames
    raise ValueError(f"Invalid frame type: {type(frames)!r}")


def resize_frames_with_padding(
    frames: torch.Tensor | np.ndarray,
    target_image_size: tuple[int, int] | tuple[int, int, int],
    bgr_conversion: bool = False,
    pad_img: bool = True,
) -> np.ndarray:
    """Resize a batched frame tensor with optional symmetric vertical padding.

    Args:
        frames: ``(N, H, W, C)`` torch tensor or numpy array.
        target_image_size: ``(H, W)`` or ``(H, W, C)`` target dimensions.
        bgr_conversion: If True, convert BGR to RGB via ``cv2.cvtColor`` first.
        pad_img: If True, zero-pad top/bottom so the source becomes square-ish
            before the final resize.

    Returns:
        Numpy array shaped ``(N, target_H, target_W, C)``.
    """
    frames = _to_numpy(frames)

    if bgr_conversion:
        frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)

    if pad_img:
        top_padding = (frames.shape[2] - frames.shape[1]) // 2
        bottom_padding = top_padding

        frames = np.pad(
            frames,
            pad_width=((0, 0), (top_padding, bottom_padding), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    target_hw = tuple(target_image_size[:2])
    if frames.shape[1:3] != target_hw:
        target_size_cv2 = (target_hw[1], target_hw[0])  # cv2 wants (W, H)
        frames = np.stack([cv2.resize(f, target_size_cv2) for f in frames])

    return frames


def apply_obs_spatial_hint(
    value: Any,
    hint: dict[str, Any] | None,
) -> Any:
    """Apply a single ``obs_spatial_hints`` entry to one observation value.

    Behaviour:
      * ``hint is None`` or value is not an ``(N, H, W, C)`` array/tensor →
        return ``value`` unchanged.
      * spatial already matches ``hint["size"]`` → return ``value`` unchanged
        (preserves the original device / dtype, including CUDA tensors that
        the dedicated tensor transport relies on).
      * otherwise → run :func:`resize_frames_with_padding` and, if the input
        was a torch tensor, restore the same device so downstream packing
        keeps the tensor on its original GPU.
    """
    if hint is None:
        return value

    target_size = hint.get("size")
    if not isinstance(target_size, (list, tuple)) or len(target_size) != 2:
        return value
    target_hw = (int(target_size[0]), int(target_size[1]))
    pad = bool(hint.get("pad", False))

    if isinstance(value, torch.Tensor):
        if value.dim() < 3:
            return value
        current_hw = (int(value.shape[1]), int(value.shape[2])) if value.dim() >= 3 else None
    elif isinstance(value, np.ndarray):
        if value.ndim < 3:
            return value
        current_hw = (int(value.shape[1]), int(value.shape[2]))
    else:
        return value

    if current_hw == target_hw:
        return value

    was_tensor = isinstance(value, torch.Tensor)
    orig_device = value.device if was_tensor else None

    resized_np = resize_frames_with_padding(
        value,
        target_image_size=target_hw,
        bgr_conversion=False,
        pad_img=pad,
    )

    if was_tensor:
        return torch.as_tensor(resized_np, device=orig_device)
    return resized_np
