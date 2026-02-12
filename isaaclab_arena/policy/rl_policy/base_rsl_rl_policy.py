# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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

import argparse
import json
from dataclasses import field
from typing import Any

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RLPolicyCfg(RslRlOnPolicyRunnerCfg):

    num_steps_per_env: int = 24
    max_iterations: int = 4000
    save_interval: int = 200
    experiment_name: str = "generic_experiment"
    obs_groups = field(
        default_factory=lambda: {
            "policy": ["policy"],
            "critic": ["policy"],
        }
    )
    # policy: RslRlPpoActorCriticCfg = field(default_factory=RslRlPpoActorCriticCfg)
    # algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=0.0001,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    @classmethod
    def update_cfg(
        cls,
        policy_cfg: dict[str, Any],
        algorithm_cfg: dict[str, Any],
        obs_groups: dict[str, list[str]],
        num_steps_per_env: int,
        max_iterations: int,
        save_interval: int,
        experiment_name: str,
    ):
        cfg = cls()
        cfg.policy = RslRlPpoActorCriticCfg(**policy_cfg)
        cfg.algorithm = RslRlPpoAlgorithmCfg(**algorithm_cfg)
        cfg.obs_groups = obs_groups
        cfg.num_steps_per_env = num_steps_per_env
        cfg.max_iterations = max_iterations
        cfg.save_interval = save_interval
        cfg.experiment_name = experiment_name
        return cfg


def get_agent_cfg(args_cli: argparse.Namespace) -> Any:
    """Get the environment and agent configuration from the command line arguments."""

    # Read a json file containing the agent configuration
    with open(args_cli.agent_cfg_path) as f:
        agent_cfg_dict = json.load(f)

    policy_cfg = agent_cfg_dict["policy_cfg"]
    algorithm_cfg = agent_cfg_dict["algorithm_cfg"]
    obs_groups = agent_cfg_dict["obs_groups"]
    # Load all other arguments if they are in args_cli as policy arguments
    num_steps_per_env = args_cli.num_steps_per_env
    max_iterations = args_cli.max_iterations
    save_interval = args_cli.save_interval
    experiment_name = args_cli.experiment_name

    agent_cfg = RLPolicyCfg.update_cfg(
        policy_cfg, algorithm_cfg, obs_groups, num_steps_per_env, max_iterations, save_interval, experiment_name
    )

    return agent_cfg
