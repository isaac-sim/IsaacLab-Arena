# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Null embodiment for scenes without a robot."""

from isaaclab.utils import configclass

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase


@configclass
class EmptyActionsCfg:
    """Empty actions config for environments without a robot.

    This allows the ActionManager to be instantiated without any action terms.
    """

    pass


@register_asset
class NullEmbodiment(EmbodimentBase):
    """A minimal embodiment for scenes without a robot.

    Use this when you want to run a simulation environment without any
    controllable robot, for example:
    - Visualization-only scenes
    - Object placement testing
    - Scene composition debugging

    Example:
        >>> from isaaclab_arena.embodiments.null_embodiment import NullEmbodiment
        >>> null_embodiment = NullEmbodiment()
        >>> env = IsaacLabArenaEnvironment(
        ...     name="my_scene",
        ...     embodiment=null_embodiment,
        ...     scene=scene,
        ...     task=DummyTask(),
        ... )
    """

    name = "null"
    tags = ["embodiment", "null"]

    def __init__(self):
        super().__init__()
        # Provide an empty action config so ActionManager can be instantiated
        self.action_config = EmptyActionsCfg()
