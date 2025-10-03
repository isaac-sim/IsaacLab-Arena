# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from scipy.spatial.transform import Rotation as R
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation

from isaac_arena.arena_control.g1_whole_body_controller.wbc_policy.g1_wbc_upperbody_ik.g1_wbc_upperbody_controller import (
    G1WBCUpperbodyController,
)
from isaac_arena.arena_control.g1_whole_body_controller.wbc_policy.policy.action_constants import (
    LEFT_HAND_STATE_DIM,
    LEFT_HAND_STATE_IDX,
    LEFT_WRIST_LINK_NAME,
    LEFT_WRIST_POS_DIM,
    LEFT_WRIST_POS_END_IDX,
    LEFT_WRIST_POS_START_IDX,
    LEFT_WRIST_QUAT_DIM,
    LEFT_WRIST_QUAT_END_IDX,
    LEFT_WRIST_QUAT_START_IDX,
    NAVIGATE_THRESHOLD,
    RIGHT_HAND_STATE_DIM,
    RIGHT_HAND_STATE_IDX,
    RIGHT_WRIST_LINK_NAME,
    RIGHT_WRIST_POS_DIM,
    RIGHT_WRIST_POS_END_IDX,
    RIGHT_WRIST_POS_START_IDX,
    RIGHT_WRIST_QUAT_DIM,
    RIGHT_WRIST_QUAT_END_IDX,
    RIGHT_WRIST_QUAT_START_IDX,
)
from isaac_arena.arena_control.g1_whole_body_controller.wbc_policy.run_policy import (
    postprocess_actions,
    prepare_observations,
)
from isaac_arena.arena_control.g1_whole_body_controller.wbc_policy.utils.p_controller import PController
from isaac_arena.embodiments.g1.mdp.actions.g1_decoupled_wbc_joint_action import G1DecoupledWBCJointAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaac_arena.embodiments.g1.mdp.actions.g1_decoupled_wbc_pink_action_cfg import G1DecoupledWBCPinkActionCfg


class G1DecoupledWBCPinkAction(G1DecoupledWBCJointAction):
    """Action term for the G1 decoupled WBC policy. Upper body PINK IK control, lower body RL-based policy."""

    cfg: G1DecoupledWBCPinkActionCfg

    _asset: Articulation
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: G1DecoupledWBCPinkActionCfg, env: ManagerBasedEnv):
        """Initialize the action term.

        Args:
            cfg: The configuration for this action term.
            env: The environment in which the action term will be applied.
        """
        super().__init__(cfg, env)

        assert self.num_envs == 1, "PINK controller currently only supports single environment"

        self.navigation_p_controller = PController(
            distance_error_threshold=self.cfg.distance_error_threshold,
            heading_diff_threshold=self.cfg.heading_diff_threshold,
            kp_angular_turning_only=self.cfg.kp_angular_turning_only,
            kp_linear_x=self.cfg.kp_linear_x,
            kp_linear_y=self.cfg.kp_linear_y,
            kp_angular=self.cfg.kp_angular,
            min_vel=self.cfg.min_vel,
            max_vel=self.cfg.max_vel,
            num_envs=self.num_envs,
            inplace_turning_flag=self.cfg.turning_first,
        )

        # Mimic navigation P-controller variables
        self._is_navigating = False
        self._navigation_goal_reached = False
        self._navigation_step_counter = 0
        self._num_navigation_subgoals_reached = -1
        self._navigate_cmd = torch.zeros([self.num_envs, 3], device=self.device)
        self._torso_orientation_rpy_cmd = torch.zeros([self.num_envs, 3], device=self.device)

        # Create the PINK IK controller
        self.upperbody_controller = G1WBCUpperbodyController(
            robot_model=self.robot_model,
            body_active_joint_groups=["arms"],
        )

    # Properties.
    # """
    @property
    def is_navigating(self) -> bool:
        """Get the is navigating flag."""
        return self._is_navigating

    @property
    def navigation_goal_reached(self) -> bool:
        """Get the navigation goal reached tensor."""
        return self._navigation_goal_reached

    @property
    def left_wrist_pos_dim(self) -> int:
        """Dimension of left wrist position command."""
        return LEFT_WRIST_POS_DIM

    @property
    def left_wrist_quat_dim(self) -> int:
        """Dimension of left wrist quaternion command."""
        return LEFT_WRIST_QUAT_DIM

    @property
    def right_wrist_pos_dim(self) -> int:
        """Dimension of right wrist position command."""
        return RIGHT_WRIST_POS_DIM

    @property
    def right_wrist_quat_dim(self) -> int:
        """Dimension of right wrist quaternion command."""
        return RIGHT_WRIST_QUAT_DIM

    @property
    def left_hand_state_dim(self) -> int:
        """Dimension of left hand state command."""
        return LEFT_HAND_STATE_DIM

    @property
    def right_hand_state_dim(self) -> int:
        """Dimension of right hand state command."""
        return RIGHT_HAND_STATE_DIM

    @property
    def action_dim(self) -> int:
        """Dimension of the action space."""
        return (
            self.left_hand_state_dim
            + self.right_hand_state_dim
            + self.left_wrist_pos_dim
            + self.left_wrist_quat_dim
            + self.right_wrist_pos_dim
            + self.right_wrist_quat_dim
            + self.navigate_cmd_dim
            + self.base_height_cmd_dim
            + self.torso_orientation_rpy_cmd_dim
        )

    @property
    def raw_actions(self) -> torch.Tensor:
        """Get the raw actions tensor."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Get the processed actions tensor."""
        return self._processed_actions

    @property
    def navigate_cmd(self):
        return self._navigate_cmd

    def compute_upperbody_joint_positions(
        self, body_data: dict[str, np.ndarray], left_hand_state: torch.Tensor, right_hand_state: torch.Tensor
    ) -> np.ndarray:
        """Run the PINK IK controller to compute the target joint positions for the upper body."""
        if self.upperbody_controller.in_warmup:
            for _ in range(50):
                target_robot_joints = self.upperbody_controller.inverse_kinematics(
                    body_data, left_hand_state, right_hand_state
                )
            self.upperbody_controller.in_warmup = False
        else:
            target_robot_joints = self.upperbody_controller.inverse_kinematics(
                body_data, left_hand_state, right_hand_state
            )
        return target_robot_joints

    # """
    # Operations.
    # """
    def process_actions(self, actions: torch.Tensor):
        """Process the input actions and set targets for each task.

        Args:
            actions: The input actions tensor.

            action tensor layout:
            action = [left_hand_state: dim=1, 0 for open, 1 for close,
                      right_hand_state: dim=1, 0 for open, 1 for close,
                      left_arm_pos: dim=3, xyz position,
                      left_arm_quat: dim=4, wxyz quaternion,
                      right_arm_pos: dim=3, xyz position,
                      right_arm_quat: dim=4, wxyz quaternion,
                      navigate_cmd: dim=3, xyz velocity,
                      base_height_cmd: dim=1, height,
                      torso_orientation_rpy_cmd: dim=3, rpy]
        """

        # Store the raw actions
        self._raw_actions[:] = actions[:, : self.action_dim]

        # Make a copy of actions before modifying so that raw actions are not modified
        actions_clone = actions.clone()

        """
        **************************************************
        Upper body PINK controller
        **************************************************
        """
        # Extract upper body left/right arm pos/quat from actions
        left_arm_pos = actions_clone[:, LEFT_WRIST_POS_START_IDX:LEFT_WRIST_POS_END_IDX].squeeze(0).cpu()
        left_arm_quat = actions_clone[:, LEFT_WRIST_QUAT_START_IDX:LEFT_WRIST_QUAT_END_IDX].squeeze(0).cpu()
        right_arm_pos = actions_clone[:, RIGHT_WRIST_POS_START_IDX:RIGHT_WRIST_POS_END_IDX].squeeze(0).cpu()
        right_arm_quat = actions_clone[:, RIGHT_WRIST_QUAT_START_IDX:RIGHT_WRIST_QUAT_END_IDX].squeeze(0).cpu()

        # Convert from pos/quat to 4x4 transform matrix
        # Scipy requires quat xyzw, IsaacLab uses wxyz so a conversion is needed
        left_arm_quat = np.roll(left_arm_quat, -1)
        right_arm_quat = np.roll(right_arm_quat, -1)
        left_rotmat = R.from_quat(left_arm_quat).as_matrix()
        right_rotmat = R.from_quat(right_arm_quat).as_matrix()

        left_arm_pose = np.eye(4)
        left_arm_pose[:3, :3] = left_rotmat
        left_arm_pose[:3, 3] = left_arm_pos

        right_arm_pose = np.eye(4)
        right_arm_pose[:3, :3] = right_rotmat
        right_arm_pose[:3, 3] = right_arm_pos

        # Extract left/right hand state from actions
        left_hand_state = actions_clone[:, LEFT_HAND_STATE_IDX].squeeze(0).cpu()
        right_hand_state = actions_clone[:, RIGHT_HAND_STATE_IDX].squeeze(0).cpu()

        # Assemble data format for running IK
        body_data = {LEFT_WRIST_LINK_NAME: left_arm_pose, RIGHT_WRIST_LINK_NAME: right_arm_pose}

        # Run IK
        target_robot_joints = self.compute_upperbody_joint_positions(body_data, left_hand_state, right_hand_state)

        # Reformat the joint position tensor to the correct order for G1 upper body
        target_upper_body_joints = target_robot_joints[self.robot_model.get_joint_group_indices("upper_body")]

        """
        **************************************************
        WBC closedloop
        **************************************************
        """
        # Extract navigate_cmd  base_height_cmd, and torso_orientation_rpy_cmd from actions
        navigate_cmd = self.get_navigation_cmd_from_actions(actions_clone)
        base_height_cmd = self.get_base_height_cmd_from_actions(actions_clone)
        torso_orientation_rpy_cmd = self.get_torso_orientation_rpy_cmd_from_actions(actions_clone)

        if self.cfg.use_p_control:
            if not self._is_navigating and self._navigation_goal_reached:
                self._navigation_goal_reached = False

            # Set flag for mimic to indicate that the robot has entered a navigation segment
            if not self._is_navigating and (np.abs(navigate_cmd) > NAVIGATE_THRESHOLD).any():
                self._is_navigating = True
                self._navigation_step_counter = 0
                self.navigation_p_controller.set_navigation_step_counter(self._navigation_step_counter)

            # Start applying navigation P-controller if conditions are met
            if self._is_navigating:
                assert self.cfg.navigation_subgoals is not None
                assert len(self.cfg.navigation_subgoals) > 0
                self._navigation_step_counter = self.navigation_p_controller.navigation_step_counter

                # No more subgoals to navigate to, stop navigation
                if (
                    self._num_navigation_subgoals_reached == len(self.cfg.navigation_subgoals) - 1
                ) or self._navigation_step_counter > self.cfg.max_navigation_steps:
                    computed_lin_vel_x, computed_lin_vel_y, computed_ang_vel = 0, 0, 0
                    self._is_navigating = False
                    self._navigation_goal_reached = True
                else:
                    target_xy_heading = self.cfg.navigation_subgoals[self._num_navigation_subgoals_reached + 1][0]
                    self.navigation_p_controller.set_inplace_turning_flag(
                        self.cfg.navigation_subgoals[self._num_navigation_subgoals_reached + 1][1]
                    )

                    target_xy = torch.tensor(target_xy_heading[:2])
                    target_heading = torch.tensor(target_xy_heading[2])
                    current_xy = self._asset.data.root_link_pos_w
                    current_heading = self._asset.data.heading_w

                    check_xy_reached = self.navigation_p_controller.check_xy_within_threshold(target_xy, current_xy)
                    check_heading_reached = self.navigation_p_controller.check_heading_within_threshold(
                        target_heading, current_heading
                    )

                    if check_xy_reached and check_heading_reached:
                        self._num_navigation_subgoals_reached += 1
                        computed_lin_vel_x, computed_lin_vel_y, computed_ang_vel = 0, 0, 0

                        self._is_navigating = False
                        self._navigation_goal_reached = True

                    # only turing in place, but may be deviated from the command xy position
                    elif check_heading_reached and self.navigation_p_controller.inplace_turning_flag:
                        computed_lin_vel_x, computed_lin_vel_y, computed_ang_vel = 0, 0, 0

                        self._num_navigation_subgoals_reached += 1
                        self._is_navigating = False
                        self._navigation_goal_reached = True

                    else:

                        computed_lin_vel_x, computed_lin_vel_y, computed_ang_vel = (
                            self.navigation_p_controller.run_p_controller(
                                target_heading=target_heading,
                                current_heading=current_heading,
                                target_xy=target_xy,
                                current_xy=current_xy,
                            )
                        )
                        # get single value out from the tensor
                        if isinstance(computed_lin_vel_x, torch.Tensor):
                            computed_lin_vel_x = computed_lin_vel_x.item()
                        if isinstance(computed_lin_vel_y, torch.Tensor):
                            computed_lin_vel_y = computed_lin_vel_y.item()
                        if isinstance(computed_ang_vel, torch.Tensor):
                            computed_ang_vel = computed_ang_vel.item()

                navigate_cmd[:, 0] = computed_lin_vel_x
                navigate_cmd[:, 1] = computed_lin_vel_y
                navigate_cmd[:, 2] = computed_ang_vel

        self._navigate_cmd = navigate_cmd.clone()

        self.set_wbc_goal(navigate_cmd, base_height_cmd, torso_orientation_rpy_cmd)
        self.wbc_policy.set_goal(self._wbc_goal)

        """
        **************************************************
        Prepare WBC policy input
        **************************************************
        """
        wbc_obs = prepare_observations(self.num_envs, self._asset.data, self.wbc_g1_joints_order)
        self.wbc_policy.set_observation(wbc_obs)

        wbc_action = self.wbc_policy.get_action(target_upper_body_joints)
        self._processed_actions = postprocess_actions(
            wbc_action, self._asset.data, self.wbc_g1_joints_order, self.device
        )

    def apply_actions(self):
        """Apply the computed joint positions based on the WBC solution."""
        self._asset.set_joint_position_target(self._processed_actions, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term for specified environments.
        Args:
            env_ids: A list of environment IDs to reset. If None, all environments are reset.
        """
        self._raw_actions[env_ids] = torch.zeros(self.action_dim, device=self.device)
