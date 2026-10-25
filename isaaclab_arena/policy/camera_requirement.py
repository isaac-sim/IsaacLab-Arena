# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraRequirement:
    """Bounding box for the image a policy consumes from one camera.

    The camera is rendered at the largest size that fits inside the box while keeping its native
    aspect ratio, so only the pixel count changes and the lens model is untouched.
    """

    height: int
    """Height of the box the policy consumes, in pixels."""

    width: int
    """Width of the box the policy consumes, in pixels."""

    def render_size(self, native_height: int, native_width: int) -> tuple[int, int]:
        """Return the (height, width) to render a camera of the given native size at.

        The result keeps the native aspect ratio and is never larger than the native size.
        """
        scale = min(1.0, self.height / native_height, self.width / native_width)
        return max(1, round(native_height * scale)), max(1, round(native_width * scale))
