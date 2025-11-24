# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_joint_action_cfg import G1DecoupledWBCJointActionCfg
from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_pink_action import G1DecoupledWBCPinkAction


@configclass
class G1DecoupledWBCPinkActionCfg(G1DecoupledWBCJointActionCfg):
    class_type: type[ActionTerm] = G1DecoupledWBCPinkAction
    """Specifies the action term class type for G1 WBC with upper body PINK IK controller."""

    # Navigation Segment: Use P-controller
    use_p_control: bool = False

    # Navigation Segment: P-controller parameters
    distance_error_threshold: float = 0.1
    heading_diff_threshold: float = 0.2
    kp_angular_turning_only: float = 0.4
    kp_linear_x: float = 2.0
    kp_linear_y: float = 2.0
    kp_angular: float = 0.05
    min_vel: float = -0.4
    max_vel: float = 0.4

    # Navigation Segment: Target x, y, heading, and turning_in_place flag subgoals
    navigation_subgoals: list[tuple[list[float], bool]] | None = None

    # Navigation Segment: Turning first
    turning_first: bool = False

    # Navigation Segment: Max navigation steps
    max_navigation_steps: int = 700
