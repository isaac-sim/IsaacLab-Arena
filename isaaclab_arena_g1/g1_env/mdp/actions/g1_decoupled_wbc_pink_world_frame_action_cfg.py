# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from .g1_decoupled_wbc_pink_action_cfg import G1DecoupledWBCPinkActionCfg
from .g1_decoupled_wbc_pink_world_frame_action import G1DecoupledWBCPinkWorldFrameAction


@configclass
class G1DecoupledWBCPinkWorldFrameActionCfg(G1DecoupledWBCPinkActionCfg):
    """Configuration for G1 WBC Pink action with world-to-base frame transformation.
    
    This configuration enables automatic transformation of wrist poses from world frame
    to robot base frame, which is necessary for VR controller inputs (e.g., Quest).
    """

    class_type: type[ActionTerm] = G1DecoupledWBCPinkWorldFrameAction
    """Specifies the action term class type for G1 WBC Pink with world frame transformation."""

    transform_to_base_frame: bool = True
    """Whether to transform wrist poses from world frame to robot base frame.
    
    Set this to True when using VR controllers or other devices that output world-space poses.
    Set to False if the input poses are already in the robot's base frame.
    """
