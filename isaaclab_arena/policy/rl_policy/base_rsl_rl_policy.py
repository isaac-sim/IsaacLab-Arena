# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

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

from dataclasses import field
from typing import Any

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RLPolicyCfg(RslRlOnPolicyRunnerCfg):

    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 200
    experiment_name = "franka_lift"
    obs_groups = field(
        default_factory=lambda: {
            "policy": ["policy"],
            "critic": ["policy"],
        }
    )
    policy: RslRlPpoActorCriticCfg = field(default_factory=lambda: RslRlPpoActorCriticCfg())
    algorithm: RslRlPpoAlgorithmCfg = field(default_factory=lambda: RslRlPpoAlgorithmCfg())

    @classmethod
    def update_cfg(cls, policy_cfg: dict[str, Any], algorithm_cfg: dict[str, Any], obs_groups: dict[str, list[str]]):
        cfg = cls()
        cfg.policy = RslRlPpoActorCriticCfg(**policy_cfg)
        cfg.algorithm = RslRlPpoAlgorithmCfg(**algorithm_cfg)
        cfg.obs_groups = obs_groups
        return cfg
