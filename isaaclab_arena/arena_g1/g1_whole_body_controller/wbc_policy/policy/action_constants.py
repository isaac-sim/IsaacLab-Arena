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

"""Constants for the WBC PINK action."""

# Action dimensions
LEFT_WRIST_POS_DIM = 3
LEFT_WRIST_QUAT_DIM = 4
RIGHT_WRIST_POS_DIM = 3
RIGHT_WRIST_QUAT_DIM = 4
LEFT_HAND_STATE_DIM = 1
RIGHT_HAND_STATE_DIM = 1

# Action indices
LEFT_HAND_STATE_IDX = 0
RIGHT_HAND_STATE_IDX = 1
LEFT_WRIST_POS_START_IDX = 2
LEFT_WRIST_POS_END_IDX = 5
LEFT_WRIST_QUAT_START_IDX = 5
LEFT_WRIST_QUAT_END_IDX = 9
RIGHT_WRIST_POS_START_IDX = 9
RIGHT_WRIST_POS_END_IDX = 12
RIGHT_WRIST_QUAT_START_IDX = 12
RIGHT_WRIST_QUAT_END_IDX = 16
NAVIGATE_CMD_START_IDX = 16
NAVIGATE_CMD_END_IDX = 19
BASE_HEIGHT_CMD_START_IDX = 19
BASE_HEIGHT_CMD_END_IDX = 20
TORSO_ORIENTATION_RPY_CMD_START_IDX = 20
TORSO_ORIENTATION_RPY_CMD_END_IDX = 23

# Navigation p-controller params
NAVIGATE_THRESHOLD = 1e-4

# Robot model link names
LEFT_WRIST_LINK_NAME = "left_wrist_yaw_link"
RIGHT_WRIST_LINK_NAME = "right_wrist_yaw_link"
