# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T embodiment adapter for G1 locomanipulation with the whole-body controller."""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Any

from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.policy_constants import (
    NUM_BASE_HEIGHT_CMD,
    NUM_NAVIGATE_CMD,
    NUM_TORSO_ORIENTATION_RPY_CMD,
)
from isaaclab_arena_gr00t.policy.embodiment_adapters.gr00t_embodiment_adapter import Gr00tEmbodimentAdapter
from isaaclab_arena_gr00t.utils.io_utils import load_robot_joints_config_from_yaml, to_numpy
from isaaclab_arena_gr00t.utils.joints_conversion import (
    remap_policy_joints_to_sim_joints_np,
    remap_sim_joints_to_policy_joints_from_np,
    sim_joint_pos_to_hold_action,
)

_G1_JOINTS_DIR = Path(__file__).resolve().parents[2] / "embodiments" / "g1"


class Gr00tG1WbcAdapter(Gr00tEmbodimentAdapter):
    """Maps G1's GR00T joint groups to sim joints and appends whole-body-controller commands to each action.

    Each env action is the sim-ordered joints followed by navigate, base height, and torso RPY commands.
    GR00T state values are the raw ``robot_joint_pos`` observation, so the checkpoint must use sim joint units.
    """

    embodiment_tag = "NEW_EMBODIMENT"

    policy_joints_config_path = _G1_JOINTS_DIR / "gr00t_43dof_joint_space.yaml"
    """GR00T joint groups, each listing its sim joint names in GR00T order."""

    sim_joints_config_path = _G1_JOINTS_DIR / "43dof_joint_space.yaml"
    """Sim joint order shared by the ``robot_joint_pos`` observation and the joint part of the env action."""

    def __init__(self):
        self.policy_joints_config = load_robot_joints_config_from_yaml(self.policy_joints_config_path)
        self.sim_joints_config = load_robot_joints_config_from_yaml(self.sim_joints_config_path)
        self.action_dim = (
            len(self.sim_joints_config) + NUM_NAVIGATE_CMD + NUM_BASE_HEIGHT_CMD + NUM_TORSO_ORIENTATION_RPY_CMD
        )

    def extract_state_np(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        joint_pos_sim_np = to_numpy(observation["policy"]["robot_joint_pos"])
        return remap_sim_joints_to_policy_joints_from_np(
            joint_pos_sim_np, self.sim_joints_config, self.policy_joints_config
        )

    def build_action_np(self, policy_action: dict[str, np.ndarray]) -> np.ndarray:
        joints_sim_np = remap_policy_joints_to_sim_joints_np(
            policy_action, self.policy_joints_config, self.sim_joints_config, embodiment_tag=self.embodiment_tag
        )
        # NOTE(xinjieyao, 2025-09-29): GR00T output does not include
        # torso_orientation_rpy_command; use zeros.
        nav = np.asarray(policy_action["navigate_command"], dtype=np.float64)
        base_h = np.asarray(policy_action["base_height_command"], dtype=np.float64)
        torso_rpy = np.zeros((*nav.shape[:-1], NUM_TORSO_ORIENTATION_RPY_CMD), dtype=np.float64)
        return np.concatenate([joints_sim_np, nav, base_h, torso_rpy], axis=2)

    def build_hold_action(self, observation: dict[str, Any]) -> torch.Tensor:
        """Hold the current joint positions; the whole-body-controller command slots hold zero."""
        return sim_joint_pos_to_hold_action(
            observation["policy"]["robot_joint_pos"],
            self.sim_joints_config,
            self.sim_joints_config,
            self.action_dim,
        )
