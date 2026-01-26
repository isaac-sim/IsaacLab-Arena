# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase


@configclass
class EmptyActionsCfg:
    """Empty actions config for environments without an embodiment."""

    pass


class NoEmbodiment(EmbodimentBase):
    """Null object for environments without an embodiment."""

    name = "no_embodiment"

    def __init__(self):
        super().__init__()
        self.action_config = EmptyActionsCfg()
