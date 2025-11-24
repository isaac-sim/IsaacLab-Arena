# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.task_base import TaskBase


class OrchestratorBase(ABC):
    """Base class for orchestrators."""

    @abstractmethod
    def orchestrate(self, embodiment: EmbodimentBase, scene: Scene, task: TaskBase) -> None:
        """Orchestrate the environment member interaction."""
        pass
