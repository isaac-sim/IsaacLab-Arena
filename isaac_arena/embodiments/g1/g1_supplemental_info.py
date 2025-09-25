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

from dataclasses import dataclass, field

import isaac_arena.embodiments.g1.g1_constants as g1_constants


# NOTE(xinjie.yao, 9.11.2025): consider inheritating from a base class `RobotSupplementalInfo`
@dataclass
class G1SupplementalInfo:
    """
    Supplemental information for the G1 robot.
    """

    # Define all actuated joints
    # NOTE(xinjie.yao, 9.22.2025): dataclass doesn't support mutable objects as default value, to
    # prevent modification that would share among all instances.
    # Using the default factory to create a new instance of the list each dataclass instance
    body_actuated_joints: list[str] = field(
        default_factory=lambda: [
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
    )

    left_hand_actuated_joints: list[str] = field(
        default_factory=lambda: [
            # Left hand
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
        ]
    )

    right_hand_actuated_joints: list[str] = field(
        default_factory=lambda: [
            # Right hand
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
        ]
    )

    # Define joint limits from URDF
    joint_limits: dict[str, list[float]] = field(
        default_factory=lambda: {
            # Left leg
            "left_hip_pitch_joint": g1_constants.G1_LEFT_HIP_PITCH_LIMITS,
            "left_hip_roll_joint": g1_constants.G1_LEFT_HIP_ROLL_LIMITS,
            "left_hip_yaw_joint": g1_constants.G1_LEFT_HIP_YAW_LIMITS,
            "left_knee_joint": g1_constants.G1_LEFT_KNEE_LIMITS,
            "left_ankle_pitch_joint": g1_constants.G1_LEFT_ANKLE_PITCH_LIMITS,
            "left_ankle_roll_joint": g1_constants.G1_LEFT_ANKLE_ROLL_LIMITS,
            # Right leg
            "right_hip_pitch_joint": g1_constants.G1_RIGHT_HIP_PITCH_LIMITS,
            "right_hip_roll_joint": g1_constants.G1_RIGHT_HIP_ROLL_LIMITS,
            "right_hip_yaw_joint": g1_constants.G1_RIGHT_HIP_YAW_LIMITS,
            "right_knee_joint": g1_constants.G1_RIGHT_KNEE_LIMITS,
            "right_ankle_pitch_joint": g1_constants.G1_RIGHT_ANKLE_PITCH_LIMITS,
            "right_ankle_roll_joint": g1_constants.G1_RIGHT_ANKLE_ROLL_LIMITS,
            # Waist
            "waist_yaw_joint": g1_constants.G1_WAIST_YAW_LIMITS,
            "waist_roll_joint": g1_constants.G1_WAIST_ROLL_LIMITS,
            "waist_pitch_joint": g1_constants.G1_WAIST_PITCH_LIMITS,
            # Left arm
            "left_shoulder_pitch_joint": g1_constants.G1_LEFT_SHOULDER_PITCH_LIMITS,
            "left_shoulder_roll_joint": g1_constants.G1_LEFT_SHOULDER_ROLL_LIMITS,
            "left_shoulder_yaw_joint": g1_constants.G1_LEFT_SHOULDER_YAW_LIMITS,
            "left_elbow_joint": g1_constants.G1_LEFT_ELBOW_LIMITS,
            "left_wrist_roll_joint": g1_constants.G1_LEFT_WRIST_ROLL_LIMITS,
            "left_wrist_pitch_joint": g1_constants.G1_LEFT_WRIST_PITCH_LIMITS,
            "left_wrist_yaw_joint": g1_constants.G1_LEFT_WRIST_YAW_LIMITS,
            # Right arm
            "right_shoulder_pitch_joint": g1_constants.G1_RIGHT_SHOULDER_PITCH_LIMITS,
            "right_shoulder_roll_joint": g1_constants.G1_RIGHT_SHOULDER_ROLL_LIMITS,
            "right_shoulder_yaw_joint": g1_constants.G1_RIGHT_SHOULDER_YAW_LIMITS,
            "right_elbow_joint": g1_constants.G1_RIGHT_ELBOW_LIMITS,
            "right_wrist_roll_joint": g1_constants.G1_RIGHT_WRIST_ROLL_LIMITS,
            "right_wrist_pitch_joint": g1_constants.G1_RIGHT_WRIST_PITCH_LIMITS,
            "right_wrist_yaw_joint": g1_constants.G1_RIGHT_WRIST_YAW_LIMITS,
            # Left hand
            "left_hand_thumb_0_joint": g1_constants.G1_LEFT_HAND_THUMB_0_LIMITS,
            "left_hand_thumb_1_joint": g1_constants.G1_LEFT_HAND_THUMB_1_LIMITS,
            "left_hand_thumb_2_joint": g1_constants.G1_LEFT_HAND_THUMB_2_LIMITS,
            "left_hand_index_0_joint": g1_constants.G1_LEFT_HAND_INDEX_0_LIMITS,
            "left_hand_index_1_joint": g1_constants.G1_LEFT_HAND_INDEX_1_LIMITS,
            "left_hand_middle_0_joint": g1_constants.G1_LEFT_HAND_MIDDLE_0_LIMITS,
            "left_hand_middle_1_joint": g1_constants.G1_LEFT_HAND_MIDDLE_1_LIMITS,
            # Right hand
            "right_hand_thumb_0_joint": g1_constants.G1_RIGHT_HAND_THUMB_0_LIMITS,
            "right_hand_thumb_1_joint": g1_constants.G1_RIGHT_HAND_THUMB_1_LIMITS,
            "right_hand_thumb_2_joint": g1_constants.G1_RIGHT_HAND_THUMB_2_LIMITS,
            "right_hand_index_0_joint": g1_constants.G1_RIGHT_HAND_INDEX_0_LIMITS,
            "right_hand_index_1_joint": g1_constants.G1_RIGHT_HAND_INDEX_1_LIMITS,
            "right_hand_middle_0_joint": g1_constants.G1_RIGHT_HAND_MIDDLE_0_LIMITS,
            "right_hand_middle_1_joint": g1_constants.G1_RIGHT_HAND_MIDDLE_1_LIMITS,
        }
    )

    # Define joint groups
    joint_groups: dict[str, dict[str, list[str]]] = field(
        default_factory=lambda: {
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
    )

    # Define joint name mapping from generic types to robot-specific names
    joint_name_mapping: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
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
    )

    root_frame_name: str = "pelvis"
    hand_frame_names: dict[str, str] = field(
        default_factory=lambda: {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}
    )
    default_joint_q: dict[str, float] = field(default_factory=lambda: {})


@dataclass
class G1SupplementalInfoWaistUpperBody(G1SupplementalInfo):
    """
    G1 supplemental information with waist as part of upper body instead of lower body.
    This version moves the waist joints from lower_body to upper_body_no_hands.
    """

    def __post_init__(self):
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

    def __post_init__(self):
        # Modify joint groups to include waist in both upper_body_no_hands and lower_body
        modified_joint_groups = self.joint_groups.copy()

        # Add waist to upper_body_no_hands (along with arms)
        modified_joint_groups["upper_body_no_hands"] = {"joints": [], "groups": ["arms", "waist"]}

        # Update the joint_groups attribute
        self.joint_groups = modified_joint_groups
