# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T embodiment adapter for GR1 tabletop manipulation with arm and hand joint control."""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Any

from isaaclab_arena_gr00t.policy.embodiment_adapters.gr00t_embodiment_adapter import Gr00tEmbodimentAdapter
from isaaclab_arena_gr00t.utils.io_utils import load_robot_joints_config_from_yaml, to_numpy
from isaaclab_arena_gr00t.utils.joints_conversion import (
    remap_policy_joints_to_sim_joints_np,
    remap_sim_joints_to_policy_joints_from_np,
    sim_joint_pos_to_hold_action,
)

_GR1_JOINTS_DIR = Path(__file__).resolve().parents[2] / "embodiments" / "gr1"


class Gr00tGr1ArmsAdapter(Gr00tEmbodimentAdapter):
    """Maps GR1's GR00T arm and hand joint groups to and from sim joint positions by joint name.

    GR00T state values are the raw ``robot_joint_pos`` observation, so the checkpoint must use sim joint units.
    """

    embodiment_tag = "GR1"

    policy_joints_config_path = _GR1_JOINTS_DIR / "gr00t_26dof_joint_space.yaml"
    """GR00T joint groups, each listing its sim joint names in GR00T order."""

    state_joints_config_path = _GR1_JOINTS_DIR / "54dof_joint_space.yaml"
    """Sim joint order of the ``robot_joint_pos`` observation."""

    action_joints_config_path = _GR1_JOINTS_DIR / "36dof_joint_space.yaml"
    """Sim joint order of the env action."""

    def __init__(self):
        self.policy_joints_config = load_robot_joints_config_from_yaml(self.policy_joints_config_path)
        self.state_joints_config = load_robot_joints_config_from_yaml(self.state_joints_config_path)
        self.action_joints_config = load_robot_joints_config_from_yaml(self.action_joints_config_path)
        self.action_dim = len(self.action_joints_config)

    def extract_state_np(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        joint_pos_sim_np = to_numpy(observation["policy"]["robot_joint_pos"])
        return remap_sim_joints_to_policy_joints_from_np(
            joint_pos_sim_np, self.state_joints_config, self.policy_joints_config
        )

    def build_action_np(self, policy_action: dict[str, np.ndarray]) -> np.ndarray:
        return remap_policy_joints_to_sim_joints_np(
            policy_action, self.policy_joints_config, self.action_joints_config, embodiment_tag=self.embodiment_tag
        )

    def build_hold_action(self, observation: dict[str, Any]) -> torch.Tensor:
        return sim_joint_pos_to_hold_action(
            observation["policy"]["robot_joint_pos"],
            self.state_joints_config,
            self.action_joints_config,
            self.action_dim,
        )
