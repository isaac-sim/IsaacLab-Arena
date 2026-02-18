# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from isaaclab_arena_gr00t.utils.robot_joints import JointsAbsPosition


def _joint_name_to_policy_group(
    joint_name: str,
    embodiment_tag: str,
    policy_joints_config: dict[str, list[str]],
) -> str | None:
    """Map a sim joint name to the policy joint group for the given embodiment.

    Returns the policy group key (e.g. 'left_arm', 'joint_position') or None if not mapped.
    """
    prefix = joint_name.split("_")[0].lower()
    tag = embodiment_tag.upper()

    if tag == "GR1":
        if prefix == "l":
            return "left_hand"
        if prefix == "r":
            return "right_hand"
        if prefix == "left":
            return "left_hand" if "hand" in joint_name else "left_arm"
        if prefix == "right":
            return "right_hand" if "hand" in joint_name else "right_arm"
        return None

    if tag == "NEW_EMBODIMENT":
        if prefix == "left":
            return "left_hand" if "hand" in joint_name else "left_arm"
        if prefix == "right":
            return "right_hand" if "hand" in joint_name else "right_arm"
        if prefix == "waist":
            return "waist"
        return None

    if tag == "OXE_DROID":
        # DROID: policy groups are joint_position, gripper_position
        for group, names in policy_joints_config.items():
            if joint_name in names:
                return group
        return None

    # Fallback: find the policy group that contains this joint name
    for group, names in policy_joints_config.items():
        if joint_name in names:
            return group
    return None


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
    embodiment_tag: str = "NEW_EMBODIMENT",
) -> JointsAbsPosition:
    """
    Remap the actions joints from policy joint orders to simulation joint orders.

    embodiment_tag: Robot embodiment (e.g. GR1, NEW_EMBODIMENT, OXE_DROID) for joint name mapping.
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
        joint_group = _joint_name_to_policy_group(joint_name, embodiment_tag, policy_joints_config)
        if joint_group is None:
            continue
        if joint_name in policy_joints_config[joint_group]:
            if joint_group in policy_joints:
                gr00t_index = policy_joints_config[joint_group].index(joint_name)
                data[..., joint_index] = torch.from_numpy(policy_joints[f"{joint_group}"][..., gr00t_index]).to(device)

    sim_joints = JointsAbsPosition(joints_pos=data, joints_order_config=sim_joints_config)
    return sim_joints
