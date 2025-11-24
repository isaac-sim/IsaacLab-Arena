# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_joint_action import G1DecoupledWBCJointAction


@configclass
class G1DecoupledWBCJointActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = G1DecoupledWBCJointAction
    """Specifies the action term class type for G1 WBC with upper body direct joint position control."""

    preserve_order: bool = False
    joint_names: list[str] = MISSING

    wbc_version: str = "homie_v2"
