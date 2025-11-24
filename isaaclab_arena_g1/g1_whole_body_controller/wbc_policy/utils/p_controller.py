# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

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

import torch

import isaaclab.utils.math as math_utils


# TODO(xyao, 9/25/2025): Add p-controller test case
class PController:
    """P-controller for turning/navigation."""

    def __init__(
        self,
        distance_error_threshold: float = 0.1,
        heading_diff_threshold: float = 0.1,
        kp_angular_turning_only: float = 0.4,
        kp_linear_x: float = 0.2,
        kp_linear_y: float = 0.2,
        kp_angular: float = 0.05,
        min_vel: float = -1,
        max_vel: float = 1,
        num_envs: int = 1,
        inplace_turning_flag: bool = False,
    ):
        self._distance_error_threshold = distance_error_threshold
        self._heading_diff_threshold = heading_diff_threshold
        self._kp_angular_turning_only = kp_angular_turning_only
        self._kp_linear_x = kp_linear_x
        self._kp_linear_y = kp_linear_y
        self._kp_angular = kp_angular
        self._min_vel = min_vel
        self._max_vel = max_vel
        self._num_envs = num_envs
        self._inplace_turning_flag = inplace_turning_flag
        self._navigation_step_counter = 0

    @property
    def inplace_turning_flag(self) -> bool:
        return self._inplace_turning_flag

    @property
    def navigation_step_counter(self) -> int:
        return self._navigation_step_counter

    def set_navigation_step_counter(self, navigation_step_counter: int) -> None:
        self._navigation_step_counter = navigation_step_counter

    def set_inplace_turning_flag(self, inplace_turning_flag: bool) -> None:
        self._inplace_turning_flag = inplace_turning_flag

    def get_pos_diff(
        self, target_xy: torch.Tensor, current_xy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the position difference between the target and current position."""
        # NOTE(xinjieyao, 2025-09-24): only developed for single env as Mimic runs in single env mode
        assert (
            current_xy.shape[0] == self._num_envs
        ), f"Current position shape must be {self._num_envs}, got {current_xy.shape[0]}"
        dx = target_xy[0] - current_xy[:, 0]
        dy = target_xy[1] - current_xy[:, 1]
        distance_error = torch.sqrt(dx**2 + dy**2)
        return dx, dy, distance_error

    def get_heading_diff(self, target_heading: torch.Tensor, current_heading: torch.Tensor) -> torch.Tensor:
        """Get the heading difference between the target and current heading."""
        # NOTE(xinjieyao, 2025-09-24): only developed for single env as Mimic runs in single env mode
        assert (
            current_heading.shape[0] == self._num_envs
        ), f"Current heading shape must be {self._num_envs}, got {current_heading.shape[0]}"
        heading_error = math_utils.wrap_to_pi(target_heading - math_utils.wrap_to_pi(current_heading))
        return heading_error

    def check_xy_within_threshold(self, target_xy: torch.Tensor, current_xy: torch.Tensor) -> bool:
        dx, dy, distance_error = self.get_pos_diff(target_xy, current_xy)
        return distance_error < self._distance_error_threshold

    def check_heading_within_threshold(self, target_heading: torch.Tensor, current_heading: torch.Tensor) -> bool:
        heading_error = self.get_heading_diff(target_heading, current_heading)
        return torch.abs(heading_error) < self._heading_diff_threshold

    def turning_p_controller(self, target_heading: torch.Tensor, current_heading: torch.Tensor) -> torch.Tensor:
        """P-controller for (theoretically) in-place turning."""
        heading_error = math_utils.wrap_to_pi(target_heading - current_heading)
        angular_velocity = self._kp_angular_turning_only * heading_error
        ang_vel = max(min(angular_velocity[0], self._max_vel), self._min_vel)
        return ang_vel

    def navigation_p_controller(
        self, target_xy: torch.Tensor, current_xy: torch.Tensor, current_theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """P-controller for x & y & yaw control."""
        # apply a P controller such that the robot can stay along the command velocity
        dx, dy, distance_error = self.get_pos_diff(target_xy, current_xy)
        wrapped_theta = math_utils.wrap_to_pi(current_theta)
        # translate wdx, dy in world frame to roobt local frame
        dx_local = dx * torch.cos(current_theta) + dy * torch.sin(current_theta)
        dy_local = -dx * torch.sin(current_theta) + dy * torch.cos(current_theta)

        # Desired angle to target (still useful for angular control)
        desired_angle = torch.atan2(dy, dx)

        # Angular error (difference between desired and current orientation)
        # Normalize angle error to be between -pi and pi
        angle_error = math_utils.wrap_to_pi(desired_angle - wrapped_theta)

        # --- Linear Velocity Control (P-only for x and y components) ---
        # vx and vy are directly proportional to the x and y errors
        vx = self._kp_linear_x * dx_local
        vy = self._kp_linear_y * dy_local

        # --- Angular Velocity Control (P-only) ---
        angular_velocity = self._kp_angular * angle_error

        # NOTE(xinjieyao, 2025-09-24): only developed for single env as Mimic runs in single env mode
        lin_vel_x = max(min(vx[0], self._max_vel), self._min_vel)
        lin_vel_y = max(min(vy[0], self._max_vel), self._min_vel)
        ang_vel = max(min(angular_velocity[0], self._max_vel), self._min_vel)
        return lin_vel_x, lin_vel_y, ang_vel

    def run_p_controller(
        self,
        target_heading: torch.Tensor,
        target_xy: torch.Tensor,
        current_heading: torch.Tensor,
        current_xy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the P-controller for navigatiion and/or turning."""

        lin_vel_x = 0.0
        lin_vel_y = 0.0
        ang_vel = 0.0

        xy_reached = self.check_xy_within_threshold(target_xy, current_xy)
        heading_reached = self.check_heading_within_threshold(target_heading, current_heading)
        if not xy_reached and not self.inplace_turning_flag:
            lin_vel_x, lin_vel_y, ang_vel = self.navigation_p_controller(target_xy, current_xy, current_heading)
        # Assuming the robot is already at the target xy, then only turn to the target heading, maybe causing infinite loops (xy <-> turning)
        elif not heading_reached:
            ang_vel = self.turning_p_controller(target_heading, current_heading)
        # if both are reached, stop the robot
        else:
            return 0, 0, 0

        self._navigation_step_counter += 1

        if lin_vel_x > 0.005 and lin_vel_x < 0.1:
            lin_vel_x = 0.1
        if lin_vel_y > 0.005 and lin_vel_y < 0.1:
            lin_vel_y = 0.1
        if ang_vel > 0.005 and ang_vel < 0.1:
            ang_vel = 0.1

        return lin_vel_x, lin_vel_y, ang_vel
