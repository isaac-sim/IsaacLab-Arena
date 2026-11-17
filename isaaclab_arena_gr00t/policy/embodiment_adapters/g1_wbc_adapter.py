# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T embodiment adapter for G1 locomanipulation with the whole-body controller."""

from __future__ import annotations

import numpy as np
import torch
from typing import Any

from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.policy_constants import (
    NUM_BASE_HEIGHT_CMD,
    NUM_NAVIGATE_CMD,
    NUM_TORSO_ORIENTATION_RPY_CMD,
)
from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyCfg
from isaaclab_arena_gr00t.policy.embodiment_adapters.gr00t_embodiment_adapter import Gr00tEmbodimentAdapter
from isaaclab_arena_gr00t.utils.io_utils import load_robot_joints_config_from_yaml, to_numpy
from isaaclab_arena_gr00t.utils.joints_conversion import (
    remap_policy_joints_to_sim_joints_np,
    remap_sim_joints_to_policy_joints_from_np,
    sim_joint_pos_to_hold_action,
)


class Gr00tG1WbcAdapter(Gr00tEmbodimentAdapter):
    """Maps G1's GR00T joint groups to sim joints and appends whole-body-controller commands to each action.

    Each env action is the sim-ordered joints followed by navigate, base height, and torso RPY commands.
    GR00T state values are the raw ``robot_joint_pos`` observation, so the checkpoint must use sim joint units.
    """

    def __init__(self, policy_config: Gr00tClosedloopPolicyCfg):
        assert policy_config.policy_joints_config_path is not None, "policy_joints_config_path is required"
        assert policy_config.action_joints_config_path is not None, "action_joints_config_path is required"
        assert policy_config.state_joints_config_path is not None, "state_joints_config_path is required"
        self.embodiment_tag = policy_config.embodiment_tag
        self.policy_joints_config = load_robot_joints_config_from_yaml(policy_config.policy_joints_config_path)
        self.action_joints_config = load_robot_joints_config_from_yaml(policy_config.action_joints_config_path)
        self.state_joints_config = load_robot_joints_config_from_yaml(policy_config.state_joints_config_path)
        self.action_dim = (
            len(self.action_joints_config) + NUM_NAVIGATE_CMD + NUM_BASE_HEIGHT_CMD + NUM_TORSO_ORIENTATION_RPY_CMD
        )

    def extract_state_np(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        joint_pos_sim_np = to_numpy(observation["policy"]["robot_joint_pos"])
        return remap_sim_joints_to_policy_joints_from_np(
            joint_pos_sim_np, self.state_joints_config, self.policy_joints_config
        )

    def build_action_np(self, policy_action: dict[str, np.ndarray]) -> np.ndarray:
        joints_sim_np = remap_policy_joints_to_sim_joints_np(
            policy_action, self.policy_joints_config, self.action_joints_config, embodiment_tag=self.embodiment_tag
        )
        # NOTE(xinjieyao, 2025-09-29): GR00T output does not include
        # torso_orientation_rpy_command; use zeros.
        nav = np.asarray(policy_action["navigate_command"], dtype=np.float64)
        base_h = np.asarray(policy_action["base_height_command"], dtype=np.float64)
        torso_rpy = np.zeros_like(nav, dtype=np.float64)
        return np.concatenate([joints_sim_np, nav, base_h, torso_rpy], axis=2)

    def build_hold_action(self, observation: dict[str, Any]) -> torch.Tensor:
        """Hold the current joint positions; the whole-body-controller command slots hold zero."""
        return sim_joint_pos_to_hold_action(
            observation["policy"]["robot_joint_pos"],
            self.state_joints_config,
            self.action_joints_config,
            self.action_dim,
        )
