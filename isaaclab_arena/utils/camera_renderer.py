# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Camera renderer identifiers shared across Arena layers."""

from enum import Enum


class CameraRenderer(str, Enum):
    """Renderers that draw the embodiment cameras in Arena."""

    ISAAC_RTX = "isaac_rtx"
    """Isaac Sim's RTX renderer, through Kit. Renders tiled and untiled cameras with either physics backend."""

    NEWTON_WARP = "newton_warp"
    """Newton's Warp ray tracer, which renders Newton's model of the scene on the GPU. Requires tiled cameras and,
    in Arena, the Newton physics backend: with PhysX, cameras on robot links stay at their start pose."""

    def __str__(self) -> str:
        return self.value
