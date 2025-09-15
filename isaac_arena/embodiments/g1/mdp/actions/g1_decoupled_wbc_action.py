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

import copy
import numpy as np
import os
import time
import torch
import yaml
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

from isaac_arena.embodiments.g1.wbc_policy.config.configs import HomieV2Config
from isaac_arena.embodiments.g1.wbc_policy.policy.wbc_policy_factory import get_wbc_policy
from isaac_arena.embodiments.g1.wbc_policy.run_policy import (
    convert_sim_joint_to_wbc_joint,
    postprocess_actions,
    prepare_observations,
)
from isaac_arena.embodiments.g1.wbc_policy.utils.g1 import instantiate_g1_robot_model

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaac_arena.embodiments.g1.mdp.actions.g1_decoupled_wbc_action_cfg import G1DecoupledWBCActionCfg


class G1DecoupledWBCAction(ActionTerm):
    """Action term for the G1 decoupled WBC policy. Upper body direct joint position control, lower body RL-based policy."""

    cfg: G1DecoupledWBCActionCfg

    _asset: Articulation
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: G1DecoupledWBCActionCfg, env: ManagerBasedEnv):
        """Initialize the action term.

        Args:
            cfg: The configuration for this action term.
            env: The environment in which the action term will be applied.
        """
        super().__init__(cfg, env)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        self._processed_actions = torch.zeros([self.num_envs, self._num_joints], device=self.device)

        self._wbc_version = self.cfg.wbc_version

        if self._wbc_version == "homie_v2":
            wbc_config = HomieV2Config()
            self._nav_cmd_dim = 3
            self._base_height_cmd_dim = 1
            self._torso_orientation_rpy_cmd_dim = 3
        else:
            raise ValueError(f"Invalid WBC version: {self._wbc_version}")

        waist_location = "lower_and_upper_body" if wbc_config.enable_waist else "lower_body"
        self.robot_model = instantiate_g1_robot_model(waist_location=waist_location)

        self.wbc_policy = get_wbc_policy("g1", self.robot_model, wbc_config)

        self._wbc_goal = {
            # "target_upper_body_pose": np.tile(self.current_upper_body_pose, (self.num_envs, 1)),
            # lin_vel_cmd_x, lin_vel_cmd_y, ang_vel_cmd
            "navigate_cmd": np.tile(np.array([[0.0, 0.0, 0.0]]), (self.num_envs, 1)),
            # base_height_cmd: 0.75 as pelvis height
            "base_height_command": np.tile(np.array([0.75]), (self.num_envs, 1)),
            # toggle_policy_action: 0 for disable policy, 1 for enable policy
            "toggle_policy_action": np.tile(np.array([0]), (self.num_envs, 1)),
            # roll pitch yaw command
            "torso_orientation_rpy_cmd": np.tile(np.array([[0.0, 0.0, 0.0]]), (self.num_envs, 1)),
        }
        wbc_g1_joints_order_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "wbc_policy/config/loco_manip_g1_joints_order_43dof.yaml",
        )
        try:
            with open(wbc_g1_joints_order_path) as f:
                self.wbc_g1_joints_order = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {wbc_g1_joints_order_path}")

    # Properties.
    # """
    @property
    def num_joints(self) -> int:
        """Get the number of joints."""
        return self._num_joints

    @property
    def navigation_goal_reached(self) -> bool:
        """Get the navigation goal reached tensor."""
        return self._navigation_goal_reached

    @property
    def navigate_cmd_dim(self) -> int:
        """Dimension of navigation command."""
        return self._nav_cmd_dim

    @property
    def base_height_cmd_dim(self) -> int:
        """Dimension of base height command."""
        return self._base_height_cmd_dim

    @property
    def torso_orientation_rpy_cmd_dim(self) -> int:
        """Dimension of torso orientation command."""
        return self._torso_orientation_rpy_cmd_dim

    @property
    def action_dim(self) -> int:
        """Dimension of the action space (based on number of tasks and pose dimension)."""
        return 43 + 3 + 1 + 3

    @property
    def raw_actions(self) -> torch.Tensor:
        """Get the raw actions tensor."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Get the processed actions tensor."""
        return self._processed_actions

    @property
    def get_wbc_version(self):
        return self._wbc_version

    @property
    def get_wbc_policy(self):
        return self.wbc_policy

    @property
    def get_wbc_goal(self):
        return self._wbc_goal

    def set_wbc_goal(self, navigate_cmd, base_height_cmd, torso_orientation_rpy_cmd=None):
        self._wbc_goal["navigate_cmd"] = navigate_cmd.cpu().numpy().repeat(self.num_envs, axis=0)
        self._wbc_goal["base_height_command"] = base_height_cmd.cpu().numpy().repeat(self.num_envs, axis=0)
        if self._wbc_version == "homie_v2" and torso_orientation_rpy_cmd is not None:
            self._wbc_goal["torso_orientation_rpy_cmd"] = (
                torso_orientation_rpy_cmd.cpu().numpy().repeat(self.num_envs, axis=0)
            )

    # """
    # Operations.
    # """

    def process_actions(self, actions: torch.Tensor):
        """Process the input actions and set targets for each task.

        Args:
            actions: The input actions tensor.
        """

        # Store the raw actions
        self._raw_actions[:] = actions[:, : self.action_dim]

        # Make a copy of actions before modifying so that raw actions are not modified
        actions_clone = actions.clone()

        """
        **************************************************
        WBC closedloop
        **************************************************
        """
        # extract navigate_cmd  base_height_cmd, and torso_orientation_rpy_cmd from actions
        navigate_cmd = actions_clone[:, -7:-4]
        base_height_cmd = actions_clone[:, -4:-3]
        torso_orientation_rpy_cmd = actions_clone[:, -3:]

        self.set_wbc_goal(navigate_cmd, base_height_cmd, torso_orientation_rpy_cmd)
        self.wbc_policy.set_goal(self._wbc_goal)

        """
        **************************************************
        Prepare WBC policy input
        **************************************************
        """
        wbc_obs = prepare_observations(self.num_envs, self._asset.data, self.wbc_g1_joints_order)
        sim_target_full_body_joints = actions_clone[:, : self._num_joints]
        wbc_target_full_body_joints = convert_sim_joint_to_wbc_joint(
            sim_target_full_body_joints, self._asset.data.joint_names, self.wbc_g1_joints_order
        )
        wbc_target_upper_body_joints = wbc_target_full_body_joints[
            :, self.robot_model.get_joint_group_indices("upper_body")
        ]

        self.wbc_policy.set_observation(wbc_obs)

        # TODO: add batched dimension
        wbc_action = self.wbc_policy.get_action(wbc_target_upper_body_joints)
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
