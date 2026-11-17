# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T embodiment adapter for G1 locomanipulation with the whole-body controller."""

from __future__ import annotations

import numpy as np

from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.policy_constants import (
    NUM_BASE_HEIGHT_CMD,
    NUM_NAVIGATE_CMD,
    NUM_TORSO_ORIENTATION_RPY_CMD,
)
from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyCfg
from isaaclab_arena_gr00t.policy.embodiment_adapters.joint_space_adapter import Gr00tJointSpaceAdapter


class Gr00tG1WbcAdapter(Gr00tJointSpaceAdapter):
    """Joint-space adapter for G1 locomanipulation that appends whole-body-controller commands to joint actions.

    Each env action is the sim-ordered joints followed by navigate, base height, and torso RPY commands.
    """

    def __init__(self, policy_config: Gr00tClosedloopPolicyCfg):
        super().__init__(policy_config)
        self.action_dim += NUM_NAVIGATE_CMD + NUM_BASE_HEIGHT_CMD + NUM_TORSO_ORIENTATION_RPY_CMD

    def build_action_np(self, policy_action: dict[str, np.ndarray]) -> np.ndarray:
        joints_sim_np = super().build_action_np(policy_action)
        # NOTE(xinjieyao, 2025-09-29): GR00T output does not include
        # torso_orientation_rpy_command; use zeros.
        nav = np.asarray(policy_action["navigate_command"], dtype=np.float64)
        base_h = np.asarray(policy_action["base_height_command"], dtype=np.float64)
        torso_rpy = np.zeros_like(nav, dtype=np.float64)
        return np.concatenate([joints_sim_np, nav, base_h, torso_rpy], axis=2)
