# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Miscellaneous reference scenes.

These four legacy scenes pre-date the structured one-object / two-object
splits and do not share a common base class.  They are kept as the
``BASE_SCENES`` set used for quick experiments and debugging.  All four carry
:attr:`SceneCategory.REFERENCE` metadata and write to
``<output_root>/miscellaneous/<base_name>/dataset.h5``.
"""

from isaaclab_arena_datagen.environments.miscellaneous.ball_and_box_environment import BallAndBoxEnvironment
from isaaclab_arena_datagen.environments.miscellaneous.ball_box_robot_environment import BallBoxRobotEnvironment
from isaaclab_arena_datagen.environments.miscellaneous.single_ball_environment import SingleBallEnvironment
from isaaclab_arena_datagen.environments.miscellaneous.single_cracker_box_environment import SingleCrackerBoxEnvironment

__all__ = [
    "BallAndBoxEnvironment",
    "BallBoxRobotEnvironment",
    "SingleBallEnvironment",
    "SingleCrackerBoxEnvironment",
]
