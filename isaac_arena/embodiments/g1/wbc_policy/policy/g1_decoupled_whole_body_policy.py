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

from typing import Optional

import numpy as np

from isaac_arena.embodiments.g1.wbc_policy.policy.base import WBCPolicy


class G1DecoupledWholeBodyPolicy(WBCPolicy):
    """
    This class implements a whole-body policy for the G1 robot by combining an upper-body
    interpolation policy and a lower-body RL-based policy.
    It is designed to work with the G1 robot's specific configuration and control requirements.
    """

    def __init__(
        self,
        robot_model,
        lower_body_policy: WBCPolicy,
        upper_body_policy: WBCPolicy,
        num_envs=1,
    ):
        self.robot_model = robot_model
        self.lower_body_policy = lower_body_policy
        self.upper_body_policy = upper_body_policy
        self.num_envs = num_envs

    def set_observation(self, observation):
        # NOTE(xinjie.yao): Upper body policy is pass-through, so we don't need to set the observation
        self.lower_body_policy.set_observation(observation)

    def set_goal(self, goal):
        """
        Set the goal for both upper and lower body policies.

        Args:
            goal: Command from the planners
            goal["navigate_cmd"]: Target navigation velocities for the lower body policy
            goal["base_height_command"]: Target base height for the lower body policy
        """
        # upper_body_goal = {}
        lower_body_goal = {}
        # # Upper body goal keys
        # upper_body_keys = [
        #     "target_upper_body_pose",
        # ]
        # for key in upper_body_keys:
        #     if key in goal:
        #         upper_body_goal[key] = goal[key]

        # Lower body goal keys
        lower_body_keys = [
            "navigate_cmd",
            "base_height_command",
            # only used in homie_v2
            "torso_orientation_rpy_cmd",
        ]
        for key in lower_body_keys:
            if key in goal:
                lower_body_goal[key] = goal[key]

        # self.upper_body_policy.set_goal(upper_body_goal)
        self.lower_body_policy.set_goal(lower_body_goal)

    def get_action(self, upper_body_target_pose_mujoco: Optional[np.ndarray] = None):
        # Get indices for groups
        lower_body_indices = self.robot_model.get_joint_group_indices("lower_body")
        upper_body_indices = self.robot_model.get_joint_group_indices("upper_body")

        # Initialize full configuration with zeros
        q = np.zeros([self.num_envs, self.robot_model.num_dofs])
        if upper_body_target_pose_mujoco is not None:
            upper_body_action = self.upper_body_policy.get_action(upper_body_target_pose_mujoco)
            assert upper_body_action["q"].shape[-1] == len(upper_body_indices), f"Upper body action has {upper_body_action['q'].shape[-1]} dofs, but upper body has {len(upper_body_indices)} dofs"
            q[:, upper_body_indices] = upper_body_action["q"]

        lower_body_action = self.lower_body_policy.get_action()
        q[:, lower_body_indices] = lower_body_action["body_action"][:, :len(lower_body_indices)]

        return {"q": q}
