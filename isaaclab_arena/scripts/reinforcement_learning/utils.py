# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from typing import Any

from isaaclab_arena.policy.rl_policy.base_rsl_rl_policy import RLPolicyCfg


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
