# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""HDR environment map asset for dome lights."""

from typing import Literal

from isaaclab_arena.assets.asset import Asset

# The texture format types supported by Isaac Sim's DomeLightCfg.
TextureFormat = Literal["automatic", "latlong", "mirroredBall", "angular", "cubeMapVerticalCross"]


class HDRImage(Asset):
    """An HDR/EXR environment map texture for dome lights.

    Example usage::

        hdr = hdr_registry.get_hdr_by_name("home_office_robolab")()
        dome_light = asset_registry.get_asset_by_name("light")()
        dome_light.add_hdr(hdr)
    """

    def __init__(
        self,
        name: str,
        texture_file: str,
        tags: list[str] | None = None,
        description: str = "",
        texture_format: TextureFormat = "latlong",
    ):
        """Create an HDR image asset.

        Args:
            name: Name of the asset.
            texture_file: Path or URL of the HDR or EXR texture file.
            tags: Optional labels attached to the asset.
            description: Human-readable description of the image.
            texture_format: Projection format of the texture, as understood by the Isaac Sim dome light.
        """
        super().__init__(name=name, tags=tags)
        self.texture_file = texture_file
        self.description = description
        self.texture_format = texture_format

    def __repr__(self) -> str:
        return f"HDRImage(name={self.name!r}, texture_file={self.texture_file!r})"
