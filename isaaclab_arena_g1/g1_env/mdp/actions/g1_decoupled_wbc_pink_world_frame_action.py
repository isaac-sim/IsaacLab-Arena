# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul

from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_pink_action import G1DecoupledWBCPinkAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .g1_decoupled_wbc_pink_world_frame_action_cfg import G1DecoupledWBCPinkWorldFrameActionCfg


class G1DecoupledWBCPinkWorldFrameAction(G1DecoupledWBCPinkAction):
    """Action term for G1 WBC Pink that transforms world-frame wrist poses to base-frame.
    
    This action term extends the base G1DecoupledWBCPinkAction to handle VR controller inputs
    that provide wrist poses in world coordinates. It automatically transforms these to the
    robot's base frame before passing them to the IK controller.
    
    This is necessary because VR controllers (like Quest) output absolute world positions,
    but the robot's IK expects positions relative to its base. Without this transformation,
    when the robot moves, the hands would try to stay at their original world position and
    drift away from the body.
    """

    cfg: G1DecoupledWBCPinkWorldFrameActionCfg

    def __init__(self, cfg: G1DecoupledWBCPinkWorldFrameActionCfg, env: ManagerBasedEnv):
        """Initialize the action term.

        Args:
            cfg: The configuration for this action term.
            env: The environment in which the action term will be applied.
        """
        super().__init__(cfg, env)

    def process_actions(self, actions: torch.Tensor):
        """Process the input actions with world-to-base frame transformation.

        This method first transforms wrist poses from world frame to robot base frame
        (if enabled in config), then calls the parent class's process_actions.

        Args:
            actions: The input actions tensor.

            Expected action layout (23 elements):
            [0:1] left_hand_state (0=open, 1=close)
            [1:2] right_hand_state (0=open, 1=close)
            [2:5] left_wrist_pos (x,y,z in world frame)
            [5:9] left_wrist_quat (w,x,y,z in world frame)
            [9:12] right_wrist_pos (x,y,z in world frame)
            [12:16] right_wrist_quat (w,x,y,z in world frame)
            [16:19] navigate_cmd (x, y, angular_z)
            [19:20] base_height
            [20:23] torso_orientation_rpy
        """
        if self.cfg.transform_to_base_frame:
            actions = self._transform_wrist_poses_to_base_frame(actions)

        # Call parent class to do the actual processing
        super().process_actions(actions)

    def _transform_wrist_poses_to_base_frame(self, actions: torch.Tensor) -> torch.Tensor:
        """Transform wrist positions and orientations from world frame to robot base frame.

        The VR controllers give world-space positions, but the IK expects positions
        relative to the robot's base. Without this, when the robot moves, the hands
        will try to stay at their original world position and drift away from the body.

        Args:
            actions: Input actions with wrist poses in world frame.

        Returns:
            Actions with wrist poses transformed to robot base frame.
        """
        # Get robot's current base pose (position and orientation in world frame)
        robot_base_pos = self._asset.data.root_link_pos_w[0, :3]  # Shape: (3,)
        robot_base_quat = self._asset.data.root_link_quat_w[0]  # Shape: (4,) in (w,x,y,z) format

        # Clone actions to avoid modifying the input
        actions = actions.clone()

        # Batch process both wrists together for efficiency
        # Stack positions: shape (2, 3) for left and right
        wrist_pos_world = torch.stack([actions[0, 2:5], actions[0, 9:12]], dim=0)

        # Step 1: Translate to remove base position offset
        wrist_pos_translated = wrist_pos_world - robot_base_pos

        # Step 2: Rotate into robot's local frame (batch operation)
        wrist_pos_base = quat_apply_inverse(robot_base_quat.unsqueeze(0), wrist_pos_translated)

        # Transform orientations to robot base frame (batch operation)
        # Stack quaternions: shape (2, 4) for left and right
        wrist_quat_world = torch.stack([actions[0, 5:9], actions[0, 12:16]], dim=0)
        robot_base_quat_inv = quat_inv(robot_base_quat.unsqueeze(0))  # Shape: (1, 4)
        robot_base_quat_inv = robot_base_quat_inv.expand(2, -1)  # Expand to (2, 4) for batching
        wrist_quat_base = quat_mul(robot_base_quat_inv, wrist_quat_world)

        # Update action with base-relative poses
        actions[0, 2:5] = wrist_pos_base[0]
        actions[0, 5:9] = wrist_quat_base[0]
        actions[0, 9:12] = wrist_pos_base[1]
        actions[0, 12:16] = wrist_quat_base[1]

        return actions
