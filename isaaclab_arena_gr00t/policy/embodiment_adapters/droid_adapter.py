# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T embodiment adapter for the DROID Franka arm with a Robotiq gripper."""

from __future__ import annotations

import numpy as np
import torch
from typing import Any

from isaaclab_arena_gr00t.policy.embodiment_adapters.gr00t_embodiment_adapter import Gr00tEmbodimentAdapter
from isaaclab_arena_gr00t.utils.io_utils import to_numpy


class Gr00tDroidAdapter(Gr00tEmbodimentAdapter):
    """Adapter for GR00T-N1.6-DROID, whose gripper state and action range from 0 (open) to 1 (closed).

    State comes from the DROID embodiment's ``joint_pos`` and ``gripper_pos`` observations. ``gripper_pos``
    divides the finger angle by the backend's closed angle, matching the 0-to-1 command the gripper action
    term accepts. Each env action is the 7 arm joint targets followed by the gripper command.
    """

    num_arm_joints = 7
    """Franka arm joints in GR00T's ``joint_position`` state and action."""

    action_dim = num_arm_joints + 1
    """Arm joint targets followed by one gripper command."""
    policy_obs_group = "policy"
    """Observation group holding the DROID embodiment's proprioceptive terms."""

    def extract_state_np(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        proprio = observation[self.policy_obs_group]
        return {
            "joint_position": to_numpy(proprio["joint_pos"]),
            "gripper_position": to_numpy(proprio["gripper_pos"]),
        }

    def build_action_np(self, policy_action: dict[str, np.ndarray]) -> np.ndarray:
        joint_position = np.asarray(policy_action["joint_position"], dtype=np.float64)
        gripper_position = np.asarray(policy_action["gripper_position"], dtype=np.float64)
        assert (
            joint_position.shape[-1] == self.num_arm_joints
        ), f"Expected {self.num_arm_joints} arm joints in joint_position, got shape {joint_position.shape}"
        assert gripper_position.shape[-1] == 1, f"Expected one gripper value, got shape {gripper_position.shape}"
        return np.concatenate([joint_position, gripper_position], axis=-1)

    def build_hold_action(self, observation: dict[str, Any]) -> torch.Tensor:
        """Hold the measured arm joints and the last gripper command.

        The gripper action term thresholds a 0-to-1 command, so holding the measured finger angle would
        read a closed Newton gripper (0.461 rad) as an open command.
        """
        proprio = observation[self.policy_obs_group]
        last_action = proprio["actions"]
        assert (
            last_action.shape[-1] == self.action_dim
        ), f"Expected last action width {self.action_dim}, got shape {last_action.shape}"
        last_gripper_command = last_action[:, self.num_arm_joints :]
        return torch.cat([proprio["joint_pos"], last_gripper_command.to(proprio["joint_pos"].dtype)], dim=-1)
