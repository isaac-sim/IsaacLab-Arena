# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
