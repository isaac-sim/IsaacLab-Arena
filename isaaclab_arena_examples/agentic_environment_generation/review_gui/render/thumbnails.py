# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


def format_aabb_dimensions_m(dims: tuple[float, float, float]) -> str:
    """Format axis-aligned bounding box size as ``x × y × z m``."""
    x, y, z = dims
    return f"{x:.3f} × {y:.3f} × {z:.3f} m"
