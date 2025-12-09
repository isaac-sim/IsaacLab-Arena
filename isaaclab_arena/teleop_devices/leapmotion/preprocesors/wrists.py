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
from copy import deepcopy

from isaaclab_arena.teleop_devices.leapmotion.preprocesors.pre_processor import PreProcessor

WRIST_POSITION_OFFSET = np.matrix(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.01, -0.04, 0.03, 1.0]]
).T
RIGHT_HAND_ROTATION = np.matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

# TODO @fcastanedaga: get this rotation from robot model
G1_HAND_ROTATION = np.matrix([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])


class WristsPreProcessor(PreProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pose = {}  # poses to calibrate the robot
        self.ee_name_list = []  # name of the end-effector "link_head_pitch", "link_LArm7", "link_RArm7"
        self.init_ee_pose = {}  # initial end-effector pose of the robot
        self.T_init_inv = {}  # initial transformation matrix
        self.R = {}  # relative transformation matrix
        self.rotation_correction = {}

        self.latest_data = None

    def calibrate(self, data):
        left_elbow_joint_name = self.robot.supplemental_info.joint_name_mapping["elbow_pitch"]["left"]
        right_elbow_joint_name = self.robot.supplemental_info.joint_name_mapping["elbow_pitch"]["right"]
        if "left_wrist" in data:
            self.ee_name_list.append(self.robot.supplemental_info.hand_frame_names["left"])
            self.pose[left_elbow_joint_name] = self.robot.supplemental_info.elbow_calibration_joint_angles["left"]
        if "right_wrist" in data:
            self.ee_name_list.append(self.robot.supplemental_info.hand_frame_names["right"])
            self.pose[right_elbow_joint_name] = self.robot.supplemental_info.elbow_calibration_joint_angles["right"]

        if self.pose:
            q = deepcopy(self.robot.q_default)
            # set pose
            for joint, degree in self.pose.items():
                joint_idx = self.robot.joint_to_dof_index[joint]
                q[joint_idx] = np.deg2rad(degree)
            self.robot.cache_forward_kinematics(q)
            target_ee_poses = [self.robot.frame_placement(ee_name).np for ee_name in self.ee_name_list]
            self.robot.reset_forward_kinematics()
        else:
            target_ee_poses = [self.robot.frame_placement(ee_name).np for ee_name in self.ee_name_list]

        for ee_name in self.ee_name_list:
            self.init_ee_pose[ee_name] = deepcopy(target_ee_poses[self.ee_name_list.index(ee_name)])
            self.T_init_inv[ee_name] = (
                np.linalg.inv(deepcopy(data["left_wrist"] @ WRIST_POSITION_OFFSET))
                if ee_name == self.robot.supplemental_info.hand_frame_names["left"]
                else np.linalg.inv(deepcopy(data["right_wrist"] @ WRIST_POSITION_OFFSET))
            )

    def __call__(self, data) -> dict:
        processed_data = {}
        for ee_name in self.ee_name_list:
            # Select wrist data based on ee_name
            T_cur = (
                data["left_wrist"]
                if ee_name == self.robot.supplemental_info.hand_frame_names["left"]
                else data["right_wrist"]
            )

            target_ee_pose = np.zeros((4, 4))
            T_cur_local = self.T_init_inv[ee_name] @ T_cur @ WRIST_POSITION_OFFSET
            if ee_name == self.robot.supplemental_info.hand_frame_names["left"]:
                target_ee_pose[:3, :3] = (
                    self.init_ee_pose[ee_name][:3, :3]
                    @ np.linalg.inv(G1_HAND_ROTATION)
                    @ T_cur_local[:3, :3]
                    @ G1_HAND_ROTATION
                )
                target_ee_pose[0, 3] = self.init_ee_pose[ee_name][0, 3] - T_cur_local[2, 3]
                target_ee_pose[1, 3] = self.init_ee_pose[ee_name][1, 3] + T_cur_local[1, 3]
                target_ee_pose[2, 3] = self.init_ee_pose[ee_name][2, 3] + T_cur_local[0, 3]
            else:
                target_ee_pose[:3, :3] = (
                    self.init_ee_pose[ee_name][:3, :3]
                    @ np.linalg.inv(G1_HAND_ROTATION)
                    @ np.linalg.inv(RIGHT_HAND_ROTATION)
                    @ T_cur_local[:3, :3]
                    @ RIGHT_HAND_ROTATION
                    @ G1_HAND_ROTATION
                )
                target_ee_pose[0, 3] = self.init_ee_pose[ee_name][0, 3] - T_cur_local[2, 3]
                target_ee_pose[1, 3] = self.init_ee_pose[ee_name][1, 3] - T_cur_local[1, 3]
                target_ee_pose[2, 3] = self.init_ee_pose[ee_name][2, 3] - T_cur_local[0, 3]

            processed_data[ee_name] = target_ee_pose

        self.latest_data = processed_data
        return processed_data
