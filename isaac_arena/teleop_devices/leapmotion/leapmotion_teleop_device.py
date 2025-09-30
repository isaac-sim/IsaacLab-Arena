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

import numpy as np
import torch
from collections.abc import Callable
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

from isaaclab.devices.device_base import DeviceBase, DeviceCfg

from isaac_arena.embodiments.g1.robot_model import ReducedRobotModel, RobotModel
from isaac_arena.embodiments.g1.wbc_policy.utils.g1 import instantiate_g1_robot_model
from isaac_arena.teleop_devices.leapmotion.leapmotion_streamer import LeapMotionStreamer
from isaac_arena.teleop_devices.leapmotion.preprocesors.fingers import FingersPreProcessor
from isaac_arena.teleop_devices.leapmotion.preprocesors.wrists import WristsPreProcessor


class LeapmotionStreamer:
    """
    Robot-agnostic teleop retargeting inverse kinematics code.
    """

    def __init__(
        self,
        robot_model: RobotModel,
        left_hand_ik_solver,
        right_hand_ik_solver,
        body_control_device: str,
        hand_control_device: str | None = None,
        body_active_joint_groups: list[str] | None = None,
    ):
        # initialize the body
        if body_active_joint_groups is not None:
            self.body = ReducedRobotModel.from_active_groups(robot_model, body_active_joint_groups)
            self.full_robot = self.body.full_robot
            self.using_reduced_robot_model = True
        else:
            self.body = robot_model
            self.full_robot = self.body
            self.using_reduced_robot_model = False

        # initialize pre_proccessors
        self.body_control_device = body_control_device
        if body_control_device:
            self.body_pre_processor = WristsPreProcessor()
            self.body_pre_processor.register(self.body)
        else:
            self.body_pre_processor = None

        # initialize hand pre-processors and post-processors
        self.hand_control_device = hand_control_device
        if hand_control_device:
            self.left_hand_pre_processor = FingersPreProcessor(side="left")
            self.right_hand_pre_processor = FingersPreProcessor(side="right")
        else:
            self.left_hand_pre_processor = None
            self.right_hand_pre_processor = None

        self.body_streamer = LeapMotionStreamer()
        self.body_streamer.start_streaming()

        self.raw_data = {}

    def get_streamer_raw_data(self):
        body_data_ = self.body_streamer.get()
        self.raw_data.update(body_data_)
        return self.raw_data

    def calibrate(self):
        """Calibrate the pre-processors."""
        assert self.body_streamer, "Real device is enabled, but no streamer is initialized."
        raw_data = self.get_streamer_raw_data()

        if self.body_pre_processor:
            self.body_pre_processor.calibrate(raw_data)
        if self.left_hand_pre_processor:
            self.left_hand_pre_processor.calibrate(raw_data)
        if self.right_hand_pre_processor:
            self.right_hand_pre_processor.calibrate(raw_data)

    def pre_process(self, raw_data):
        """Pre-process the raw data."""
        assert (
            self.body_pre_processor or self.left_hand_pre_processor or self.right_hand_pre_processor
        ), "Pre-processors are not initialized."
        if self.body_pre_processor:
            body_data = self.body_pre_processor(raw_data)
            if self.left_hand_pre_processor and self.right_hand_pre_processor:
                left_hand_data = self.left_hand_pre_processor(raw_data)
                right_hand_data = self.right_hand_pre_processor(raw_data)
                return body_data, left_hand_data, right_hand_data
            else:
                return body_data, None, None
        else:  # only hands
            if self.left_hand_pre_processor and self.right_hand_pre_processor:
                left_hand_data = self.left_hand_pre_processor(raw_data)
                right_hand_data = self.right_hand_pre_processor(raw_data)
                return None, left_hand_data, right_hand_data

    def get_g1_gripper_state(self, finger_data, dist_threshold=0.05):
        fingertips = finger_data["position"]

        # Extract X, Y, Z positions of fingertips from the transformation matrices
        positions = np.array([finger[:3, 3] for finger in fingertips])
        positions = np.reshape(positions, (-1, 3))  # Ensure 2D array with shape (N, 3)

        # Compute the distance between the thumb and index finger
        thumb_pos = positions[4, :]
        index_pos = positions[4 + 5, :]
        dist = np.linalg.norm(thumb_pos - index_pos)
        hand_close = dist < dist_threshold

        return int(hand_close)

    def get_leapmotion_action(self):
        raw_action = self.get_streamer_raw_data()
        body_data, left_hand_data, right_hand_data = self.pre_process(raw_action)

        # Original variables
        left_wrist_action = np.array(body_data["left_wrist_yaw_link"])
        right_wrist_action = np.array(body_data["right_wrist_yaw_link"])

        # Extract position and quaternion from left wrist 4x4 pose matrix
        left_wrist_pos = torch.from_numpy(left_wrist_action[:3, 3])
        left_wrist_rot_matrix = left_wrist_action[:3, :3]
        left_wrist_quat = R.from_matrix(left_wrist_rot_matrix).as_quat()

        # Extract position and quaternion from right wrist 4x4 pose matrix
        right_wrist_pos = torch.from_numpy(right_wrist_action[:3, 3])
        right_wrist_rot_matrix = right_wrist_action[:3, :3]
        right_wrist_quat = R.from_matrix(right_wrist_rot_matrix).as_quat()

        # Convert quaternions from (x, y, z, w) to IsaacLab (w, x, y, z) format
        left_wrist_quat_wxyz = torch.from_numpy(np.roll(left_wrist_quat, 1))
        right_wrist_quat_wxyz = torch.from_numpy(np.roll(right_wrist_quat, 1))

        # Get G1 hand state (0 for open, 1 for close)
        left_hand_state = torch.from_numpy(np.array([self.get_g1_gripper_state(left_hand_data)]))
        right_hand_state = torch.from_numpy(np.array([self.get_g1_gripper_state(right_hand_data)]))

        # Assemble into upper body env action
        upperbody_action = torch.cat([
            left_hand_state,
            right_hand_state,
            left_wrist_pos,
            left_wrist_quat_wxyz,
            right_wrist_pos,
            right_wrist_quat_wxyz,
        ])

        return upperbody_action


@dataclass
class LeapmotionCfg(DeviceCfg):
    """Configuration for the Leapmotion teleop device."""

    body_control_device: str = "leapmotion"


class Leapmotion(DeviceBase):

    def __init__(self, cfg: LeapmotionCfg):
        self.robot_model = instantiate_g1_robot_model()
        self.device_streamer = LeapmotionStreamer(
            robot_model=self.robot_model,
            left_hand_ik_solver=None,
            right_hand_ik_solver=None,
            body_control_device=cfg.body_control_device,
            hand_control_device=cfg.body_control_device,
            body_active_joint_groups=["arms"],
        )
        self.device_streamer.calibrate()

        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Leapmotion Controller for G1: {self.__class__.__name__}\n"
        return msg

    def reset(self):
        self.device_streamer.calibrate()

    def add_callback(self, key: str, func: Callable):
        """Add additional functions."""
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        return self.device_streamer.get_leapmotion_action()
