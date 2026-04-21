# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Backwards-compatible re-export.

The canonical implementation now lives in
``isaaclab_arena.utils.image_conversion`` so the remote-policy client can
share the same resize path with this GR00T-specific package without a
reverse layering dependency.
"""

from isaaclab_arena.utils.image_conversion import (
    apply_obs_spatial_hint,
    resize_frames_with_padding,
)

__all__ = ["apply_obs_spatial_hint", "resize_frames_with_padding"]
