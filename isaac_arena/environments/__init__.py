# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

from . import arena_env

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Arena-Kitchen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": arena_env.ArrangeKitchenObjectEnvCfg,
    },
    disable_env_checker=True,
)
