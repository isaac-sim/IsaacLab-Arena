# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.orchestrator.orchestrator_base import OrchestratorBase
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.task_base import TaskBase
    from isaaclab_arena.teleop_devices.teleop_device_base import TeleopDeviceBase


@configclass
class IsaacLabArenaEnvironment:
    """Describes an environment in IsaacLab Arena."""

    name: str = MISSING
    """The name of the environment."""

    embodiment: EmbodimentBase = MISSING
    """The embodiment to use in the environment."""

    scene: Scene = MISSING
    """The scene to use in the environment."""

    task: TaskBase = MISSING
    """The task to use in the environment."""

    teleop_device: TeleopDeviceBase | None = None
    """The teleop device to use in the environment."""

    orchestrator: OrchestratorBase | None = None
    """The orchestrator to use in the environment."""
