# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Camera renderer identifiers shared across Arena layers."""

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.renderers import RendererCfg


class CameraRenderer(str, Enum):
    """Renderers that draw the embodiment cameras in Arena."""

    ISAAC_RTX = "isaac_rtx"
    """Isaac Sim's RTX renderer, through Kit. Renders tiled and untiled cameras with either physics backend."""

    NEWTON_WARP = "newton_warp"
    """Newton's Warp ray tracer, which renders Newton's model of the scene on the GPU. Requires tiled cameras and,
    in Arena, the Newton physics backend: with PhysX, cameras on robot links stay at their start pose."""

    OVRTX = "ovrtx"
    """Omniverse RTX without Kit, through Isaac Lab's OVRTX renderer. Requires the Newton physics backend and tiled
    cameras."""

    def __str__(self) -> str:
        return self.value

    def renderer_cfg(self) -> "RendererCfg | None":
        """Return a new Isaac Lab configuration of this renderer, or None for Isaac RTX, the cameras' default."""
        if self is CameraRenderer.NEWTON_WARP:
            from isaaclab_newton.renderers import NewtonWarpRendererCfg

            return NewtonWarpRendererCfg()
        if self is CameraRenderer.OVRTX:
            from isaaclab_ov.renderers import OVRTXRendererCfg

            return OVRTXRendererCfg()
        return None
