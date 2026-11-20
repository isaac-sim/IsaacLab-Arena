# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase


@configclass
class EmptyActionsCfg:
    """Empty actions config for environments without an embodiment."""

    pass


@register_asset
class NoEmbodiment(EmbodimentBase):
    """Null object for environments without an embodiment."""

    registry_name = "no_embodiment"

    def __init__(self, enable_cameras: bool = False):
        super().__init__(enable_cameras=enable_cameras)
        self.action_config = EmptyActionsCfg()
