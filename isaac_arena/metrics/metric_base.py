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

import numpy as np
from abc import ABC, abstractmethod

from isaaclab.managers.recorder_manager import RecorderTermCfg


class MetricBase(ABC):

    name: str
    recorder_term_name: str

    @abstractmethod
    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        raise NotImplementedError("Function not implemented yet.")

    @abstractmethod
    def compute_metric_from_recording(self, recorded_metric_data: list[np.ndarray]) -> float:
        raise NotImplementedError("Function not implemented yet.")
