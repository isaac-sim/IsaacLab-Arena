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
from typing import Any

from isaac_arena.metrics.metric_base import MetricBase


class TaskBase(ABC):

    @abstractmethod
    def get_scene_cfg(self) -> Any:
        raise NotImplementedError("Function not implemented yet.")

    @abstractmethod
    def get_termination_cfg(self) -> Any:
        raise NotImplementedError("Function not implemented yet.")

    @abstractmethod
    def get_events_cfg(self) -> Any:
        raise NotImplementedError("Function not implemented yet.")

    @abstractmethod
    def get_prompt(self) -> str:
        raise NotImplementedError("Function not implemented yet.")

    @abstractmethod
    def get_mimic_env_cfg(self, embodiment_name: str) -> Any:
        raise NotImplementedError("Function not implemented yet.")

    @abstractmethod
    def get_metrics(self) -> list[MetricBase]:
        raise NotImplementedError("Function not implemented yet.")
