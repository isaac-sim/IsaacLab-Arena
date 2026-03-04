# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Joint order remapping: sim <-> policy (numpy as single implementation, torch as thin wrappers)."""

import numpy as np
import torch

from isaaclab_arena_gr00t.utils.io_utils import to_numpy
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


def reorder_sim_joints_to_config_order_np(
    joint_pos_sim_np: np.ndarray,
    sim_joint_names: list[str],
    config_joint_name_to_index: dict[str, int],
) -> np.ndarray:
    """Reorder joint positions from sim order to config (e.g. state YAML) order by joint name.

    Use when the sim returns joints in asset/URDF order but the config expects a specific
    order (e.g. 54dof). Ensures each value is read by joint name so state/action mapping
    is correct regardless of sim ordering.

    Args:
        joint_pos_sim_np: (N, num_sim_joints) in sim order (same order as sim_joint_names).
        sim_joint_names: Joint names in the order they appear in joint_pos_sim_np.
        config_joint_name_to_index: Joint name -> target index (e.g. from state_joints_config).

    Returns:
        (N, len(config_joint_name_to_index)) array in config order.
    """
    n_envs = joint_pos_sim_np.shape[0]
    num_config = len(config_joint_name_to_index)
    reordered = np.zeros((n_envs, num_config), dtype=joint_pos_sim_np.dtype)
    sim_name_to_idx = {name: i for i, name in enumerate(sim_joint_names)}
    for joint_name, config_idx in config_joint_name_to_index.items():
        if joint_name not in sim_name_to_idx:
            raise ValueError(
                f"Joint {joint_name} from config not found in sim joint names (first 5: {sim_joint_names[:5]}...)"
            )
        sim_idx = sim_name_to_idx[joint_name]
        reordered[:, config_idx] = joint_pos_sim_np[:, sim_idx]
    return reordered


def remap_sim_joints_to_policy_joints_from_np(
    joint_pos_sim_np: np.ndarray,
    state_joints_config: dict[str, int],
    policy_joints_config: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    """Remap joint positions from sim order to policy groups (single implementation).

    Args:
        joint_pos_sim_np: (N, num_sim_joints) joint positions in sim order.
        state_joints_config: Sim joint name -> column index (same as joints_order_config).
        policy_joints_config: Policy group name -> list of joint names in order.

    Returns:
        Dict of group name -> (N, num_joints_in_group) numpy array.
    """
    data: dict[str, np.ndarray] = {}
    for group, joints_list in policy_joints_config.items():
        parts = []
        for joint_name in joints_list:
            if joint_name not in state_joints_config:
                raise ValueError(
                    f"Joint {joint_name} not found in state_joints_config {list(state_joints_config.keys())}"
                )
            joint_index = state_joints_config[joint_name]
            parts.append(joint_pos_sim_np[:, joint_index])
        data[group] = np.stack(parts, axis=1)
    return data


def remap_sim_joints_to_policy_joints(
    sim_joints_state: JointsAbsPosition,
    policy_joints_config: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    """Remap joint state/actions from sim order to policy groups (torch wrapper).

    Delegates to :func:`remap_sim_joints_to_policy_joints_from_np` after
    converting JointsAbsPosition to numpy array and state config.

    Args:
        sim_joints_state: Sim joint state (torch tensor + joints_order_config).
        policy_joints_config: Policy group name -> list of joint names in order.

    Returns:
        Dict of group name -> (N, num_joints_in_group) numpy array.
    """
    assert isinstance(sim_joints_state, JointsAbsPosition)
    joint_pos_sim_np = to_numpy(sim_joints_state.joints_pos)
    state_joints_config = sim_joints_state.joints_order_config
    return remap_sim_joints_to_policy_joints_from_np(joint_pos_sim_np, state_joints_config, policy_joints_config)


def remap_policy_joints_to_sim_joints_np(
    policy_joints: dict[str, np.ndarray],
    policy_joints_config: dict[str, dict[str, int]],
    sim_joints_config: dict[str, int],
    embodiment_tag: str,
) -> np.ndarray:
    """Remap policy joint outputs to sim order as a single numpy array (single implementation).

    Args:
        policy_joints: Policy group name -> (N, horizon, num_joints_in_group) numpy array.
        policy_joints_config: Policy group name -> joint name -> joint index in policy order.
        sim_joints_config: Sim joint name -> joint index in sim order.
        embodiment_tag: Robot embodiment for joint name mapping.

    Returns:
        (N, horizon, num_sim_joints) numpy array in sim joint order.
    """
    policy_joint_shape = None
    for _, joint_pos in policy_joints.items():
        if policy_joint_shape is None:
            policy_joint_shape = joint_pos.shape
        else:
            assert joint_pos.ndim == 3
            assert joint_pos.shape[:2] == policy_joint_shape[:2]
    assert policy_joint_shape is not None
    data = np.zeros(
        (policy_joint_shape[0], policy_joint_shape[1], len(sim_joints_config)),
        dtype=np.float64,
    )
    for joint_name, joint_index in sim_joints_config.items():
        joint_group = _joint_name_to_policy_group(joint_name, embodiment_tag, policy_joints_config)
        if joint_group is None:
            continue
        if joint_group in policy_joints and joint_name in policy_joints_config[joint_group]:
            gr00t_index = policy_joints_config[joint_group].index(joint_name)
            data[..., joint_index] = np.asarray(policy_joints[joint_group][..., gr00t_index], dtype=np.float64)
    return data


def remap_policy_joints_to_sim_joints(
    policy_joints: dict[str, np.ndarray],
    policy_joints_config: dict[str, dict[str, int]],
    sim_joints_config: dict[str, int],
    device: torch.device,
    embodiment_tag: str,
) -> JointsAbsPosition:
    """Remap policy joint outputs to sim order (torch wrapper).

    Delegates to :func:`remap_policy_joints_to_sim_joints_np` then wraps the
    result in a torch tensor and JointsAbsPosition.

    Args:
        policy_joints: Policy group name -> (N, horizon, num_joints_in_group) numpy array.
        policy_joints_config: Policy group name -> joint name -> joint index in policy order.
        sim_joints_config: Sim joint name -> joint index in sim order.
        device: Target device for the output tensor.
        embodiment_tag: Robot embodiment for joint name mapping.

    Returns:
        JointsAbsPosition with (N, horizon, num_sim_joints) in sim order.
    """
    result_np = remap_policy_joints_to_sim_joints_np(
        policy_joints, policy_joints_config, sim_joints_config, embodiment_tag
    )
    data = torch.from_numpy(result_np).to(device)
    return JointsAbsPosition(joints_pos=data, joints_order_config=sim_joints_config)
