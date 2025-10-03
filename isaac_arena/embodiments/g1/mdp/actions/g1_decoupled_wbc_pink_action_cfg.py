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


from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from isaac_arena.embodiments.g1.mdp.actions.g1_decoupled_wbc_joint_action_cfg import G1DecoupledWBCJointActionCfg
from isaac_arena.embodiments.g1.mdp.actions.g1_decoupled_wbc_pink_action import G1DecoupledWBCPinkAction


@configclass
class G1DecoupledWBCPinkActionCfg(G1DecoupledWBCJointActionCfg):
    class_type: type[ActionTerm] = G1DecoupledWBCPinkAction
    """Specifies the action term class type for G1 WBC with upper body PINK IK controller."""

    # Navigation Segment: Use P-controller
    use_p_control: bool = False

    # Navigation Segment: P-controller parameters
    distance_error_threshold: float = 0.1
    heading_diff_threshold: float = 0.2
    kp_angular_turning_only: float = 0.4
    kp_linear_x: float = 2.0
    kp_linear_y: float = 2.0
    kp_angular: float = 0.05
    min_vel: float = -0.4
    max_vel: float = 0.4

    # Navigation Segment: Target x, y, heading, and turning_in_place flag subgoals
    navigation_subgoals: list[tuple[list[float], bool]] | None = None

    # Navigation Segment: Turning first
    turning_first: bool = False

    # Navigation Segment: Max navigation steps
    max_navigation_steps: int = 700
