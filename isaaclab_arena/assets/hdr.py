# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

# The texture format types supported by Isaac Sim's DomeLightCfg.
TextureFormat = Literal["automatic", "latlong", "mirroredBall", "angular", "cubeMapVerticalCross"]


class HDR:
    """An HDR/EXR environment map texture for dome lights.

    Example usage::

        dome_light = asset_registry.get_asset_by_name("light")()
        dome_light.add_hdr("home_office_robolab")
    """

    def __init__(
        self,
        name: str,
        texture_file: str,
        tags: list[str] | None = None,
        description: str = "",
        texture_format: TextureFormat = "latlong",
    ):
        self.name = name
        self.texture_file = texture_file
        self.tags = tags or []
        self.description = description
        self.texture_format = texture_format

    def __repr__(self) -> str:
        return f"HDR(name={self.name!r}, texture_file={self.texture_file!r})"
