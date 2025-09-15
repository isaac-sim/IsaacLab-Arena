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

import time

import numpy as np

from isaac_arena.embodiments.g1.wbc_policy.policy.g1_homie_policy import G1HomiePolicyV2
from isaac_arena.embodiments.g1.wbc_policy.policy.identity_policy import IdentityPolicy
from isaac_arena.embodiments.g1.wbc_policy.policy.g1_decoupled_whole_body_policy import G1DecoupledWholeBodyPolicy


def get_wbc_policy(
    robot_type,
    robot_model,
    wbc_config
):
    # current_upper_body_pose = robot_model.get_initial_upper_body_pose()

    if robot_type == "g1":
        # Only one mode for upper body -- passing thru, no interpolation
        upper_body_policy = IdentityPolicy()

        lower_body_policy_type = wbc_config.wbc_version
        assert wbc_config.policy_config_path is not None
        if lower_body_policy_type == "homie_v2":
            lower_body_policy = G1HomiePolicyV2(
                robot_model=robot_model,
                config=wbc_config.policy_config_path,
                model_path=wbc_config.wbc_model_path,
            )
        else:
            raise ValueError(f"Invalid lower body policy type: {lower_body_policy_type}, Supported lower body policy types: homie_v2")

        wbc_policy = G1DecoupledWholeBodyPolicy(
            robot_model=robot_model,
            upper_body_policy=upper_body_policy,
            lower_body_policy=lower_body_policy,
        )
    else:
        raise ValueError(f"Invalid robot type: {robot_type}. Supported robot types: g1")
    return wbc_policy
