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
import torch

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.metrics.metric_base import MetricBase


class SuccessRecorder(RecorderTerm):
    """Records whether an episode was successful just before the environment is reset."""

    name = "success"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # We track the first reset for each environment
        self.first_reset = True

    def record_pre_reset(self, env_ids):
        # The first time that the environment is reset, we don't want to record the success,
        # because nothing has happened yet.
        if self.first_reset:
            # We expect that on the first reset ALL the environments are reset.
            assert len(env_ids) == self._env.num_envs
            self.first_reset = False
            # Record nothing.
            return None, None
        assert hasattr(self._env, "termination_manager")
        assert "success" in self._env.termination_manager.active_terms
        success_results = torch.zeros(len(env_ids), dtype=bool, device=self._env.device)
        success_results |= self._env.termination_manager.get_term("success")[env_ids]
        return self.name, success_results


@configclass
class SuccessRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = SuccessRecorder


class SuccessRateMetric(MetricBase):
    """Computes the success rate.

    The success rate is the number of episodes in which the environment was successful, divided
    by the total number of episodes.
    """

    name = "success_rate"
    recorder_term_name = SuccessRecorder.name

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        """Return the recorder term configuration for the success rate metric."""
        return SuccessRecorderCfg()

    def compute_metric_from_recording(self, recorded_metric_data: list[np.ndarray]) -> float:
        """Gets the average success rate from a list of recorded success flags.

        Args:
            recorded_metric_data(list[np.ndarray]): The recorded success flags per simulated episode.

        Returns:
            The success rate(float). Value between 0 and 1. The proportion of episodes in
                which the environment was successful.
        """
        num_demos = len(recorded_metric_data)
        if num_demos == 0:
            return 0.0
        all_demos_success_flags = np.concatenate(recorded_metric_data)
        assert all_demos_success_flags.ndim == 1
        assert all_demos_success_flags.shape[0] == num_demos
        success_rate = np.mean(all_demos_success_flags)
        return success_rate
