# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared GR00T embodiment adapter base for robots whose GR00T state and actions are sim joint positions."""

from __future__ import annotations

import numpy as np
import torch
from typing import Any

from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyCfg
from isaaclab_arena_gr00t.policy.embodiment_adapters.gr00t_embodiment_adapter import Gr00tEmbodimentAdapter
from isaaclab_arena_gr00t.utils.io_utils import load_robot_joints_config_from_yaml, to_numpy
from isaaclab_arena_gr00t.utils.joints_conversion import (
    remap_policy_joints_to_sim_joints_np,
    remap_sim_joints_to_policy_joints_from_np,
)


class Gr00tJointSpaceAdapter(Gr00tEmbodimentAdapter):
    """Maps GR00T joint groups to and from sim joint positions by joint name.

    The policy config's joint YAMLs list the GR00T joint groups and the sim state and action joint orders.
    GR00T state values are the raw ``robot_joint_pos`` observation, so the checkpoint must use sim joint units.
    """

    joint_pos_obs_group = "policy"
    """Observation group holding the sim joint positions."""

    joint_pos_obs_name = "robot_joint_pos"
    """Observation term holding the sim joint positions, shape (num_envs, num_sim_joints)."""

    def __init__(self, policy_config: Gr00tClosedloopPolicyCfg):
        assert policy_config.policy_joints_config_path is not None, "policy_joints_config_path is required"
        assert policy_config.action_joints_config_path is not None, "action_joints_config_path is required"
        assert policy_config.state_joints_config_path is not None, "state_joints_config_path is required"
        self.embodiment_tag = policy_config.embodiment_tag
        self.policy_joints_config = load_robot_joints_config_from_yaml(policy_config.policy_joints_config_path)
        self.action_joints_config = load_robot_joints_config_from_yaml(policy_config.action_joints_config_path)
        self.state_joints_config = load_robot_joints_config_from_yaml(policy_config.state_joints_config_path)
        self.action_dim = len(self.action_joints_config)

    def extract_state_np(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        joint_pos_sim_np = to_numpy(self._get_joint_pos_sim(observation))
        return remap_sim_joints_to_policy_joints_from_np(
            joint_pos_sim_np, self.state_joints_config, self.policy_joints_config
        )

    def build_action_np(self, policy_action: dict[str, np.ndarray]) -> np.ndarray:
        return remap_policy_joints_to_sim_joints_np(
            policy_action,
            self.policy_joints_config,
            self.action_joints_config,
            embodiment_tag=self.embodiment_tag,
        )

    def build_hold_action(self, observation: dict[str, Any]) -> torch.Tensor:
        """Copy current sim joint positions into the action slots that share a joint name with the state config.

        Action slots without a matching state joint hold zero.
        """
        joint_pos_sim = self._get_joint_pos_sim(observation)
        hold_action = torch.zeros(
            (joint_pos_sim.shape[0], self.action_dim), dtype=joint_pos_sim.dtype, device=joint_pos_sim.device
        )
        for joint_name, action_idx in self.action_joints_config.items():
            state_idx = self.state_joints_config.get(joint_name)
            if state_idx is not None:
                hold_action[:, action_idx] = joint_pos_sim[:, state_idx]
        return hold_action

    def _get_joint_pos_sim(self, observation: dict[str, Any]) -> torch.Tensor:
        assert self.joint_pos_obs_group in observation, f"{self.joint_pos_obs_group} is not in observation"
        group = observation[self.joint_pos_obs_group]
        assert self.joint_pos_obs_name in group, f"{self.joint_pos_obs_name} is not in {self.joint_pos_obs_group}"
        return group[self.joint_pos_obs_name]
