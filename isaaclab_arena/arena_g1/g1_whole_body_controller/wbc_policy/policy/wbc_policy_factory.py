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


from isaaclab_arena.arena_g1.g1_env.robot_model import RobotModel
from isaaclab_arena.arena_g1.g1_whole_body_controller.wbc_policy.config.configs import BaseConfig
from isaaclab_arena.arena_g1.g1_whole_body_controller.wbc_policy.policy.base import WBCPolicy
from isaaclab_arena.arena_g1.g1_whole_body_controller.wbc_policy.policy.g1_decoupled_whole_body_policy import (
    G1DecoupledWholeBodyPolicy,
)
from isaaclab_arena.arena_g1.g1_whole_body_controller.wbc_policy.policy.g1_homie_policy import G1HomiePolicyV2
from isaaclab_arena.arena_g1.g1_whole_body_controller.wbc_policy.policy.identity_policy import IdentityPolicy


def get_wbc_policy(robot_type: str, robot_model: RobotModel, wbc_config: BaseConfig, num_envs: int = 1) -> WBCPolicy:
    """Get the WBC policy for the given robot type and configuration.

    Args:
        robot_type: The type of robot to get the WBC policy for. Only "g1" is supported.
        robot_model: The robot model to use for the WBC policy
        wbc_config: The configuration for the WBC policy
        num_envs: The number of environments to use in IsaacLab

    Returns:
        The WBC policy for the given robot type and configuration
    """
    assert num_envs > 0, f"num_envs must be greater than 0, got {num_envs}"
    if robot_type == "g1":
        # Only one mode for upper body -- passing thru, no interpolation
        upper_body_policy = IdentityPolicy()

        lower_body_policy_type = wbc_config.wbc_version
        assert wbc_config.policy_config_path is not None
        if lower_body_policy_type == "homie_v2":
            lower_body_policy = G1HomiePolicyV2(
                robot_model=robot_model,
                config_path=wbc_config.policy_config_path,
                model_path=wbc_config.wbc_model_path,
                num_envs=num_envs,
            )
        else:
            raise ValueError(
                f"Invalid lower body policy type: {lower_body_policy_type}, Supported lower body policy types: homie_v2"
            )

        wbc_policy = G1DecoupledWholeBodyPolicy(
            robot_model=robot_model,
            upper_body_policy=upper_body_policy,
            lower_body_policy=lower_body_policy,
            num_envs=num_envs,
        )
    else:
        raise ValueError(f"Invalid robot type: {robot_type}. Supported robot types: g1")
    return wbc_policy
