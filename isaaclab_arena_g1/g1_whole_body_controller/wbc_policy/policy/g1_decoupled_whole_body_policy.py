# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from isaaclab_arena_g1.g1_env.robot_model import RobotModel
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.base import WBCPolicy


class G1DecoupledWholeBodyPolicy(WBCPolicy):
    """
    This class implements a whole-body policy for the G1 robot by combining an upper-body
    interpolation policy and a lower-body RL-based policy.
    It is designed to work with the G1 robot's specific configuration and control requirements.
    """

    def __init__(
        self,
        robot_model: RobotModel,
        lower_body_policy: WBCPolicy,
        upper_body_policy: WBCPolicy,
        num_envs=1,
    ):
        self.robot_model = robot_model
        self.lower_body_policy = lower_body_policy
        self.upper_body_policy = upper_body_policy
        self.num_envs = num_envs

    def set_observation(self, observation: dict[str, np.ndarray]):
        # NOTE(xinjie.yao): Upper body policy is pass-through, so we don't need to set the observation
        self.lower_body_policy.set_observation(observation)

    def set_goal(self, goal: dict[str, np.ndarray]):
        """
        Set the goal for both upper and lower body policies.

        Args:
            goal: Command from the planners
            goal["navigate_cmd"]: Target base navigation velocities for the lower body policy
            goal["base_height_command"]: Target base height for the lower body policy
            goal["torso_orientation_rpy_cmd"]: Target torso orientation for the lower body policy (optional)
        """
        lower_body_goal = {}

        # Lower body goal keys
        lower_body_keys = [
            "navigate_cmd",
            "base_height_command",
            "torso_orientation_rpy_cmd",
        ]
        for key in lower_body_keys:
            if key in goal:
                lower_body_goal[key] = goal[key]

        self.lower_body_policy.set_goal(lower_body_goal)

    def get_action(self, upper_body_target_pose: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Get the action for the whole body policy.

        Args:
            upper_body_target_pose: The target pose for the upper body policy (optional)

        Returns:
            The action for the whole body policy in dictionary format
        """
        # Get indices for groups
        lower_body_indices = self.robot_model.get_joint_group_indices("lower_body")
        upper_body_indices = self.robot_model.get_joint_group_indices("upper_body")

        # Initialize full configuration with zeros
        q = np.zeros([self.num_envs, self.robot_model.num_dofs])
        if upper_body_target_pose is not None:
            upper_body_action = self.upper_body_policy.get_action(upper_body_target_pose)
            assert upper_body_action["q"].shape[-1] == len(upper_body_indices), (
                f"Upper body action has {upper_body_action['q'].shape[-1]} dofs, but upper body has"
                f" {len(upper_body_indices)} dofs"
            )
            q[:, upper_body_indices] = upper_body_action["q"]

        lower_body_action = self.lower_body_policy.get_action()
        q[:, lower_body_indices] = lower_body_action["body_action"][:, : len(lower_body_indices)]

        return {"q": q}
