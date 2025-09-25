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

"""Constants for G1 robot joint limits (from URDF)."""


# G1 Joint Limit Values (from URDF)
# Format: [min_limit, max_limit]

# Left leg limits
G1_LEFT_HIP_PITCH_LIMITS: list[float] = [-2.5307, 2.8798]
G1_LEFT_HIP_ROLL_LIMITS: list[float] = [-0.5236, 2.9671]
G1_LEFT_HIP_YAW_LIMITS: list[float] = [-2.7576, 2.7576]
G1_LEFT_KNEE_LIMITS: list[float] = [-0.087267, 2.8798]
G1_LEFT_ANKLE_PITCH_LIMITS: list[float] = [-0.87267, 0.5236]
G1_LEFT_ANKLE_ROLL_LIMITS: list[float] = [-0.2618, 0.2618]

# Right leg limits
G1_RIGHT_HIP_PITCH_LIMITS: list[float] = [-2.5307, 2.8798]
G1_RIGHT_HIP_ROLL_LIMITS: list[float] = [-2.9671, 0.5236]
G1_RIGHT_HIP_YAW_LIMITS: list[float] = [-2.7576, 2.7576]
G1_RIGHT_KNEE_LIMITS: list[float] = [-0.087267, 2.8798]
G1_RIGHT_ANKLE_PITCH_LIMITS: list[float] = [-0.87267, 0.5236]
G1_RIGHT_ANKLE_ROLL_LIMITS: list[float] = [-0.2618, 0.2618]

# Waist limits
G1_WAIST_YAW_LIMITS: list[float] = [-2.618, 2.618]
G1_WAIST_ROLL_LIMITS: list[float] = [-0.52, 0.52]
G1_WAIST_PITCH_LIMITS: list[float] = [-0.52, 0.52]

# Left arm limits
G1_LEFT_SHOULDER_PITCH_LIMITS: list[float] = [-3.0892, 2.6704]
G1_LEFT_SHOULDER_ROLL_LIMITS: list[float] = [-1.5882, 2.2515]
G1_LEFT_SHOULDER_YAW_LIMITS: list[float] = [-2.618, 2.618]
G1_LEFT_ELBOW_LIMITS: list[float] = [-1.0472, 2.0944]
G1_LEFT_WRIST_ROLL_LIMITS: list[float] = [-1.972222054, 1.972222054]
G1_LEFT_WRIST_PITCH_LIMITS: list[float] = [-1.614429558, 1.614429558]
G1_LEFT_WRIST_YAW_LIMITS: list[float] = [-1.614429558, 1.614429558]

# Right arm limits
G1_RIGHT_SHOULDER_PITCH_LIMITS: list[float] = [-3.0892, 2.6704]
G1_RIGHT_SHOULDER_ROLL_LIMITS: list[float] = [-2.2515, 1.5882]
G1_RIGHT_SHOULDER_YAW_LIMITS: list[float] = [-2.618, 2.618]
G1_RIGHT_ELBOW_LIMITS: list[float] = [-1.0472, 2.0944]
G1_RIGHT_WRIST_ROLL_LIMITS: list[float] = [-1.972222054, 1.972222054]
G1_RIGHT_WRIST_PITCH_LIMITS: list[float] = [-1.614429558, 1.614429558]
G1_RIGHT_WRIST_YAW_LIMITS: list[float] = [-1.614429558, 1.614429558]

# Left hand limits
G1_LEFT_HAND_THUMB_0_LIMITS: list[float] = [-1.04719755, 1.04719755]
G1_LEFT_HAND_THUMB_1_LIMITS: list[float] = [-0.72431163, 1.04719755]
G1_LEFT_HAND_THUMB_2_LIMITS: list[float] = [0, 1.74532925]
G1_LEFT_HAND_INDEX_0_LIMITS: list[float] = [-1.57079632, 0]
G1_LEFT_HAND_INDEX_1_LIMITS: list[float] = [-1.74532925, 0]
G1_LEFT_HAND_MIDDLE_0_LIMITS: list[float] = [-1.57079632, 0]
G1_LEFT_HAND_MIDDLE_1_LIMITS: list[float] = [-1.74532925, 0]

# Right hand limits
G1_RIGHT_HAND_THUMB_0_LIMITS: list[float] = [-1.04719755, 1.04719755]
G1_RIGHT_HAND_THUMB_1_LIMITS: list[float] = [-0.72431163, 1.04719755]
G1_RIGHT_HAND_THUMB_2_LIMITS: list[float] = [0, 1.74532925]
G1_RIGHT_HAND_INDEX_0_LIMITS: list[float] = [-1.57079632, 0]
G1_RIGHT_HAND_INDEX_1_LIMITS: list[float] = [-1.74532925, 0]
G1_RIGHT_HAND_MIDDLE_0_LIMITS: list[float] = [-1.57079632, 0]
G1_RIGHT_HAND_MIDDLE_1_LIMITS: list[float] = [-1.74532925, 0]
