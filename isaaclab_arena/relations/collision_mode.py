# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class CollisionMode(Enum):
    """Selects which collision detection method the solver uses for no-overlap constraints."""

    BBOX = "bbox"
    """Axis-aligned bounding box overlap volume (fast, conservative)."""

    MESH = "mesh"
    """Sphere-to-SDF queries against actual mesh geometry (accurate, slower)."""
