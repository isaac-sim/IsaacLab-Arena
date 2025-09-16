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
from dataclasses import dataclass

# NOTE(xinjie.yao, 9.11.2025): consider inheritating from a base class `RobotSupplementalInfo`
@dataclass
class G1SupplementalInfo:
    """
    Supplemental information for the G1 robot.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Define all actuated joints
        self.body_actuated_joints = [
            # Left leg
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            # Right leg
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            # Waist
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            # Left arm
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            # Right arm
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        self.left_hand_actuated_joints = [
            # Left hand
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
        ]

        self.right_hand_actuated_joints = [
            # Right hand
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
        ]

        # Define joint limits from URDF
        self.joint_limits = {
            # Left leg
            "left_hip_pitch_joint": [-2.5307, 2.8798],
            "left_hip_roll_joint": [-0.5236, 2.9671],
            "left_hip_yaw_joint": [-2.7576, 2.7576],
            "left_knee_joint": [-0.087267, 2.8798],
            "left_ankle_pitch_joint": [-0.87267, 0.5236],
            "left_ankle_roll_joint": [-0.2618, 0.2618],
            # Right leg
            "right_hip_pitch_joint": [-2.5307, 2.8798],
            "right_hip_roll_joint": [-2.9671, 0.5236],
            "right_hip_yaw_joint": [-2.7576, 2.7576],
            "right_knee_joint": [-0.087267, 2.8798],
            "right_ankle_pitch_joint": [-0.87267, 0.5236],
            "right_ankle_roll_joint": [-0.2618, 0.2618],
            # Waist
            "waist_yaw_joint": [-2.618, 2.618],
            "waist_roll_joint": [-0.52, 0.52],
            "waist_pitch_joint": [-0.52, 0.52],
            # Left arm
            "left_shoulder_pitch_joint": [-3.0892, 2.6704],
            "left_shoulder_roll_joint": [-1.5882, 2.2515],
            "left_shoulder_yaw_joint": [-2.618, 2.618],
            "left_elbow_joint": [-1.0472, 2.0944],
            "left_wrist_roll_joint": [-1.972222054, 1.972222054],
            "left_wrist_pitch_joint": [-1.614429558, 1.614429558],
            "left_wrist_yaw_joint": [-1.614429558, 1.614429558],
            # Right arm
            "right_shoulder_pitch_joint": [-3.0892, 2.6704],
            "right_shoulder_roll_joint": [-2.2515, 1.5882],
            "right_shoulder_yaw_joint": [-2.618, 2.618],
            "right_elbow_joint": [-1.0472, 2.0944],
            "right_wrist_roll_joint": [-1.972222054, 1.972222054],
            "right_wrist_pitch_joint": [-1.614429558, 1.614429558],
            "right_wrist_yaw_joint": [-1.614429558, 1.614429558],
            # Left hand
            "left_hand_thumb_0_joint": [-1.04719755, 1.04719755],
            "left_hand_thumb_1_joint": [-0.72431163, 1.04719755],
            "left_hand_thumb_2_joint": [0, 1.74532925],
            "left_hand_index_0_joint": [-1.57079632, 0],
            "left_hand_index_1_joint": [-1.74532925, 0],
            "left_hand_middle_0_joint": [-1.57079632, 0],
            "left_hand_middle_1_joint": [-1.74532925, 0],
            # Right hand
            "right_hand_thumb_0_joint": [-1.04719755, 1.04719755],
            "right_hand_thumb_1_joint": [-0.72431163, 1.04719755],
            "right_hand_thumb_2_joint": [0, 1.74532925],
            "right_hand_index_0_joint": [-1.57079632, 0],
            "right_hand_index_1_joint": [-1.74532925, 0],
            "right_hand_middle_0_joint": [-1.57079632, 0],
            "right_hand_middle_1_joint": [-1.74532925, 0],
        }

        # Define joint groups
        self.joint_groups = {
            # Body groups
            "waist": {
                "joints": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
                "groups": [],
            },
            # Leg groups
            "left_leg": {
                "joints": [
                    "left_hip_pitch_joint",
                    "left_hip_roll_joint",
                    "left_hip_yaw_joint",
                    "left_knee_joint",
                    "left_ankle_pitch_joint",
                    "left_ankle_roll_joint",
                ],
                "groups": [],
            },
            "right_leg": {
                "joints": [
                    "right_hip_pitch_joint",
                    "right_hip_roll_joint",
                    "right_hip_yaw_joint",
                    "right_knee_joint",
                    "right_ankle_pitch_joint",
                    "right_ankle_roll_joint",
                ],
                "groups": [],
            },
            "legs": {"joints": [], "groups": ["left_leg", "right_leg"]},
            # Arm groups
            "left_arm": {
                "joints": [
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                ],
                "groups": [],
            },
            "right_arm": {
                "joints": [
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ],
                "groups": [],
            },
            "arms": {"joints": [], "groups": ["left_arm", "right_arm"]},
            # Hand groups
            "left_hand": {
                "joints": [
                    "left_hand_index_0_joint",
                    "left_hand_index_1_joint",
                    "left_hand_middle_0_joint",
                    "left_hand_middle_1_joint",
                    "left_hand_thumb_0_joint",
                    "left_hand_thumb_1_joint",
                    "left_hand_thumb_2_joint",
                ],
                "groups": [],
            },
            "right_hand": {
                "joints": [
                    "right_hand_index_0_joint",
                    "right_hand_index_1_joint",
                    "right_hand_middle_0_joint",
                    "right_hand_middle_1_joint",
                    "right_hand_thumb_0_joint",
                    "right_hand_thumb_1_joint",
                    "right_hand_thumb_2_joint",
                ],
                "groups": [],
            },
            "hands": {"joints": [], "groups": ["left_hand", "right_hand"]},
            # Full body groups
            "lower_body": {"joints": [], "groups": ["waist", "legs"]},
            "upper_body_no_hands": {"joints": [], "groups": ["arms"]},
            "body": {"joints": [], "groups": ["lower_body", "upper_body_no_hands"]},
            "upper_body": {"joints": [], "groups": ["upper_body_no_hands", "hands"]},
        }

        # Define joint name mapping from generic types to robot-specific names
        self.joint_name_mapping = {
            # Waist joints
            "waist_pitch": "waist_pitch_joint",
            "waist_roll": "waist_roll_joint",
            "waist_yaw": "waist_yaw_joint",
            # Shoulder joints
            "shoulder_pitch": {
                "left": "left_shoulder_pitch_joint",
                "right": "right_shoulder_pitch_joint",
            },
            "shoulder_roll": {
                "left": "left_shoulder_roll_joint",
                "right": "right_shoulder_roll_joint",
            },
            "shoulder_yaw": {
                "left": "left_shoulder_yaw_joint",
                "right": "right_shoulder_yaw_joint",
            },
            # Elbow joints
            "elbow_pitch": {"left": "left_elbow_joint", "right": "right_elbow_joint"},
            # Wrist joints
            "wrist_pitch": {"left": "left_wrist_pitch_joint", "right": "right_wrist_pitch_joint"},
            "wrist_roll": {"left": "left_wrist_roll_joint", "right": "right_wrist_roll_joint"},
            "wrist_yaw": {"left": "left_wrist_yaw_joint", "right": "right_wrist_yaw_joint"},
        }

        self.root_frame_name = "pelvis"
        self.hand_frame_names = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}
        self.elbow_calibration_joint_angles = {"left": 0.0, "right": 0.0}
        self.hand_rotation_correction = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        self.default_joint_q = {}


@dataclass
class G1SupplementalInfoWaistUpperBody(G1SupplementalInfo):
    """
    G1 supplemental information with waist as part of upper body instead of lower body.
    This version moves the waist joints from lower_body to upper_body_no_hands.
    """

    def __init__(self):
        # Initialize with the base G1 configuration
        super().__init__()

        # Modify joint groups to move waist from lower_body to upper_body_no_hands
        modified_joint_groups = self.joint_groups.copy()

        # Remove waist from lower_body (keep only legs)
        modified_joint_groups["lower_body"] = {"joints": [], "groups": ["legs"]}

        # Add waist to upper_body_no_hands (along with arms)
        modified_joint_groups["upper_body_no_hands"] = {"joints": [], "groups": ["arms", "waist"]}

        # Update the joint_groups attribute
        self.joint_groups = modified_joint_groups


@dataclass
class G1SupplementalInfoWaistLowerAndUpperBody(G1SupplementalInfo):
    """
    G1 supplemental information with waist as part of both upper and lower body.
    This version includes the waist joints in both upper_body_no_hands and lower_body.
    """

    def __init__(self):
        # Initialize with the base G1 configuration
        super().__init__()

        # Modify joint groups to include waist in both upper_body_no_hands and lower_body
        modified_joint_groups = self.joint_groups.copy()

        # Add waist to upper_body_no_hands (along with arms)
        modified_joint_groups["upper_body_no_hands"] = {"joints": [], "groups": ["arms", "waist"]}

        # Update the joint_groups attribute
        self.joint_groups = modified_joint_groups
