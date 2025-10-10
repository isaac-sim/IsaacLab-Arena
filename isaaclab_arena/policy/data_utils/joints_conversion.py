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
import torch

from isaaclab_arena.policy.data_utils.robot_joints import JointsAbsPosition


def remap_sim_joints_to_policy_joints(
    sim_joints_state: JointsAbsPosition, policy_joints_config: dict[str, list[str]]
) -> dict[str, np.ndarray]:
    """
    Remap the state or actions joints from simulation joint orders to policy joint orders
    """
    data = {}
    assert isinstance(sim_joints_state, JointsAbsPosition)
    for group, joints_list in policy_joints_config.items():
        data[group] = []

        for joint_name in joints_list:
            if joint_name in sim_joints_state.joints_order_config:
                joint_index = sim_joints_state.joints_order_config[joint_name]
                data[group].append(sim_joints_state.joints_pos[:, joint_index])
            else:
                raise ValueError(f"Joint {joint_name} not found in {sim_joints_state.joints_order_config}")

        data[group] = np.stack(data[group], axis=1)
    return data


def remap_policy_joints_to_sim_joints(
    policy_joints: dict[str, np.array],
    policy_joints_config: dict[str, list[str]],
    sim_joints_config: dict[str, int],
    device: torch.device,
) -> JointsAbsPosition:
    """
    Remap the actions joints from policy joint orders to simulation joint orders
    """
    # assert all values in policy_joint keys are the same shape and save the shape to init data
    policy_joint_shape = None
    for _, joint_pos in policy_joints.items():
        if policy_joint_shape is None:
            policy_joint_shape = joint_pos.shape
        else:
            assert joint_pos.ndim == 3
            assert joint_pos.shape[:2] == policy_joint_shape[:2]

    assert policy_joint_shape is not None
    data = torch.zeros([policy_joint_shape[0], policy_joint_shape[1], len(sim_joints_config)], device=device)
    for joint_name, joint_index in sim_joints_config.items():
        match joint_name.split("_")[0]:
            # For GR1
            case "L":
                joint_group = "left_hand"
            # For GR1
            case "R":
                joint_group = "right_hand"
            case "left":
                # For G1
                if "hand" in joint_name:
                    joint_group = "left_hand"
                else:
                    joint_group = "left_arm"
            case "right":
                # For G1
                if "hand" in joint_name:
                    joint_group = "right_hand"
                else:
                    joint_group = "right_arm"
            case "waist":
                # For G1
                joint_group = "waist"
            case _:
                continue
        if joint_name in policy_joints_config[joint_group]:
            if f"action.{joint_group}" in policy_joints:
                gr00t_index = policy_joints_config[joint_group].index(joint_name)
                data[..., joint_index] = torch.from_numpy(policy_joints[f"action.{joint_group}"][..., gr00t_index]).to(
                    device
                )

    sim_joints = JointsAbsPosition(joints_pos=data, joints_order_config=sim_joints_config)
    return sim_joints
