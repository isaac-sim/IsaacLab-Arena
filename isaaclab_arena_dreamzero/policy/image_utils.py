# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from openpi_client import image_tools

TARGET_H: int = 180
TARGET_W: int = 320


def resize_with_pad(img: np.ndarray, height: int = TARGET_H, width: int = TARGET_W) -> np.ndarray:
    """Resize a uint8 HWC image to (height, width, 3) with letterbox padding.

    Thin wrapper around ``openpi_client.image_tools.resize_with_pad`` — the same
    letterbox resize isaaclab_arena_openpi's DROID adapter uses — so DreamZero
    doesn't reimplement it or pull in a separate cv2 dependency.

    Args:
        img: Input uint8 numpy array of shape (H, W, 3).
        height: Target height in pixels.
        width: Target width in pixels.

    Returns:
        uint8 ndarray of shape (height, width, 3).
    """
    return image_tools.resize_with_pad(img, height, width)
