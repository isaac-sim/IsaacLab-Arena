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

from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np

@dataclass
class RobotSupplementalInfo:
    """
    Base class for robot-specific information that is not easily extractable from URDF.
    This includes information about actuated joints, joint hierarchies, etc.
    """

    # List of body actuated joint names (excluding hands)
    body_actuated_joints: List[str]

    # List of left hand actuated joint names
    left_hand_actuated_joints: List[str]

    # List of right hand actuated joint names
    right_hand_actuated_joints: List[str]

    # Dictionary of joint groups, where each group is a dictionary with:
    # - "joints": list of joint names
    # - "groups": list of subgroup names (optional)
    # Example: {
    #   "right_arm": {
    #       "joints": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_joint"],
    #       "groups": []
    #   },
    #   "left_arm": {
    #       "joints": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_joint"],
    #       "groups": []
    #   },
    #   "upper_body": {
    #       "joints": ["torso_pitch_joint", "torso_yaw_joint", "torso_roll_joint"],
    #       "groups": ["right_arm", "left_arm"]
    #   }
    # }
    joint_groups: Dict[str, Dict[str, List[str]]]

    # Name of the root frame
    root_frame_name: str

    # Dictionary of hand frame names
    # Example: {
    #   "left": "left_hand_frame",
    #   "right": "right_hand_frame"
    # }
    hand_frame_names: Dict[str, str]

    # Dictionary of joint limits
    # Example: {
    #   "left_shoulder_pitch_joint": [-np.pi / 2, np.pi / 2],
    #   "right_shoulder_pitch_joint": [-np.pi / 2, np.pi / 2]
    # }
    joint_limits: Dict[str, List[float]]

    # Dictionary of elbow calibration joint angles
    # Example: {
    #   "left": -90.0,
    #   "right": -90.0
    # }

    joint_name_mapping: Dict[str, Union[str, Dict[str, str]]]
    # Dictionary of joint name mapping from generic types to robot-specific names
    # Example: {
    #   "waist_pitch": "waist_pitch_joint",
    #   "shoulder_pitch": {
    #       "left": "left_shoulder_pitch_joint",
    #       "right": "right_shoulder_pitch_joint"
    #   },
    #   "elbow_pitch": {
    #       "left": "left_elbow_pitch_joint",
    #       "right": "right_elbow_pitch_joint"
    #   }
    # }