# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RLPolicyCfg(RslRlOnPolicyRunnerCfg):
    """Default RSL-RL runner configuration for Arena environments.

    Used as the ``rsl_rl_cfg_entry_point`` when registering environments with gym,
    allowing IsaacLab's ``train.py`` to load it via ``@hydra_task_config``.
    """

    num_steps_per_env: int = 24
    max_iterations: int = 4000
    save_interval: int = 200
    experiment_name: str = "generic_experiment"
    obs_groups = field(
        default_factory=lambda: {
            "actor": ["policy", "task_obs"],
            "critic": ["policy", "task_obs"],
        }
    )
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
