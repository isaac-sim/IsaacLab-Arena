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

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from isaac_arena.embodiments.g1.mdp.actions.g1_decoupled_wbc_pink_action import G1DecoupledWBCPinkAction


@configclass
class G1DecoupledWBCPinkActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = G1DecoupledWBCPinkAction
    """Specifies the action term class type for G1 WBC with upper body PINK IK controller."""

    preserve_order: bool = False
    joint_names: list[str] = MISSING

    wbc_version: str = "homie_v2"
    # Navigation Segment: P-controller parameters
    distance_error_threshold: float = 0.05
    heading_diff_threshold: float = 0.15
    kp_angular_turning_only: float = 0.4
    kp_linear_x: float = 2.0
    kp_linear_y: float = 2.0
    kp_angular: float = 0.05
    min_vel: float = -0.4
    max_vel: float = 0.4
    # Navigation Segment: Target xy and heading subgoals
    # navigation_target_xy_heading: list[tuple[list[float], bool]] = MISSING
    navigation_target_xy_heading: list[tuple[list[float], bool]] =[([0.18, 0.0, 0.0], False),
                                                                    ([0.18, 0.0, -1.74], True),
                                                                    ([-0.0955, -1.0670, -1.74], False),
                                                                    ([-0.0955, -1.0670, -1.74], False)]
    # Navigation Segment: Use P-controller
    use_p_control: bool = True
    # Navigation Segment: Turning first
    turning_first: bool = False
    # Navigation Segment: Max navigation steps
    max_navigation_steps: int = 700
