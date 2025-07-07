# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

from . import agents, arrange_ik_rel_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Arrange-Kitchen-Object-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": arrange_ik_rel_env_cfg.ArrangeKitchenObjectEnvCfg,
    },
    disable_env_checker=True,
)
