# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

import cv2

TARGET_H: int = 180
TARGET_W: int = 320


def resize_with_pad(img: np.ndarray, height: int = TARGET_H, width: int = TARGET_W) -> np.ndarray:
    """Resize a uint8 HWC image to (height, width, 3) with letterbox padding.

    Preserves aspect ratio by scaling uniformly and centering the result on a black canvas.
    Handles both upscaling and downscaling.

    Args:
        img: Input uint8 numpy array of shape (H, W, 3).
        height: Target height in pixels.
        width: Target width in pixels.

    Returns:
        uint8 ndarray of shape (height, width, 3).
    """
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas
