# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch
from enum import Enum

import warp as wp
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.terminations import root_height_below_minimum
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import combine_frame_transforms


class SuccessMode(str, Enum):
    """How `check_success` combines its results."""

    ALL = "ALL"
    """Success needs every predicate to be True."""

    ANY = "ANY"
    """Success needs at least one predicate to be True."""

    CHOOSE = "CHOOSE"
    """Success needs at least k predicates to be True."""


def check_success(
    env: ManagerBasedRLEnv,
    predicates: list[TerminationTermCfg],
    mode: SuccessMode | str = SuccessMode.ALL,
    k: int | None = None,
) -> torch.Tensor:
    """Compose multiple termination predicates into a single success signal.

    Each predicate is wrapped in a termination term and its `func` is evaluated with its
    `params` to produce a boolean tensor of shape (num_envs,). The results are then combined according to `mode`.

    - ALL: True where every predicate is True (logical AND).
    - ANY: True where at least one predicate is True (logical OR).
    - CHOOSE: True where at least k predicates are True.

    Args:
        env: The RL environment instance.
        predicates: Termination term configs to evaluate and combine.
        mode: How to combine the predicate results (SuccessMode or string).
        k: Number of predicates that must be True when `mode` is SuccessMode.CHOOSE.

    Returns:
        A boolean tensor of shape (num_envs,) indicating success.
    """

    if not predicates:
        raise ValueError("check_success requires at least one predicate.")
    valid_modes = [m.value for m in SuccessMode]
    if not (isinstance(mode, str) and mode.upper() in valid_modes):
        raise ValueError(f"Unknown mode '{mode}'. Expected one of {valid_modes}.")
    mode = SuccessMode(mode.upper())

    results = torch.stack([predicate.func(env, **predicate.params) for predicate in predicates], dim=0)

    if mode is SuccessMode.ALL:
        return results.all(dim=0)
    if mode is SuccessMode.ANY:
        return results.any(dim=0)

    if k is None:
        raise ValueError("mode SuccessMode.CHOOSE requires 'k' to be provided.")
    if not (1 <= k <= len(predicates)):
        raise ValueError(f"'k' must be in [1, {len(predicates)}] for {len(predicates)} predicates, got {k}.")
    return results.sum(dim=0) >= k


# NOTE(alexmillane, 2025.09.15): The velocity threshold is set high because some stationary
# seem to generate a "small" velocity.
def lift_object_il_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_position: tuple[float, float, float] | None = None,
    position_tolerance: float = 0.05,
) -> torch.Tensor:
    """Dynamic success termination for lift object task.

    Args:
        env: The RL environment instance.
        object_cfg: The configuration of the object to track.
        goal_position: Fixed goal position [x, y, z] to use if command goal not available.
        position_tolerance: Distance tolerance for success (m).

    Returns:
        A boolean tensor of shape (num_envs,) indicating success.
    """

    object_instance: RigidObject = env.scene[object_cfg.name]
    object_pos = wp.to_torch(object_instance.data.root_pos_w)

    goal_pos = torch.tensor([goal_position] * env.num_envs, device=env.device)

    # Check if object is within tolerance of goal
    distance = torch.norm(object_pos - goal_pos, dim=1)
    return distance < position_tolerance


def lift_object_rl_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rl_training: bool = False,
    command_name: str = "object_pose",
    position_tolerance: float = 0.05,
) -> torch.Tensor:
    """Dynamic success termination for lift object task.

    Supports multiple modes:
    - RL training: Always returns False (no early termination)
    - RL evaluation: Uses goal from command manager

    Args:
        env: The RL environment instance.
        object_cfg: The configuration of the object to track.
        robot_cfg: The robot configuration (needed to transform the goal to world frame).
        rl_training: If True, always returns False (disables success termination for RL training).
        command_name: The name of the command that is used to control the object.
        position_tolerance: Distance tolerance for success (m).

    Returns:
        A boolean tensor of shape (num_envs,) indicating success.
    """
    if rl_training:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    robot: RigidObject = env.scene[robot_cfg.name]
    object_instance: RigidObject = env.scene[object_cfg.name]

    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]

    # Transform goal from robot-base frame to world frame
    root_pos_w = wp.to_torch(robot.data.root_pos_w)
    root_quat_w = wp.to_torch(robot.data.root_quat_w)
    des_pos_w, _ = combine_frame_transforms(root_pos_w, root_quat_w, des_pos_b)

    object_pos_w = wp.to_torch(object_instance.data.root_pos_w)
    distance = torch.linalg.norm(des_pos_w - object_pos_w[:, :3], dim=1)
    return distance < position_tolerance


def goal_pose_task_termination(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_x_range: tuple[float, float] | None = None,
    target_y_range: tuple[float, float] | None = None,
    target_z_range: tuple[float, float] | None = None,
    target_orientation_xyzw: tuple[float, float, float, float] | None = None,
    target_orientation_tolerance_rad: float = 0.1,
) -> torch.Tensor:
    """Terminate when the object's pose is within the thresholds (BBox + Orientation).

    Args:
        env: The RL environment instance.
        object_cfg: The configuration of the object to track.
        target_x_range: Success zone x-range [min, max] in meters.
        target_y_range: Success zone y-range [min, max] in meters.
        target_z_range: Success zone z-range [min, max] in meters.
        target_orientation_xyzw: Target quaternion [x, y, z, w].
        target_orientation_tolerance_rad: Angular tolerance in radians (default: 0.1).

    Returns:
        A boolean tensor of shape (num_envs, )
    """
    object_instance: RigidObject = env.scene[object_cfg.name]
    object_root_pos_w = wp.to_torch(object_instance.data.root_pos_w)
    object_root_quat_w = wp.to_torch(object_instance.data.root_quat_w)

    device = env.device
    num_envs = env.num_envs

    has_any_threshold = any([
        target_x_range is not None,
        target_y_range is not None,
        target_z_range is not None,
        target_orientation_xyzw is not None,
    ])

    if not has_any_threshold:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)

    success = torch.ones(num_envs, dtype=torch.bool, device=device)

    # Position range checks
    ranges = [target_x_range, target_y_range, target_z_range]
    for idx, range_val in enumerate(ranges):
        if range_val is not None:
            range_min, range_max = range_val
            in_range = (object_root_pos_w[:, idx] >= range_min) & (object_root_pos_w[:, idx] <= range_max)
            success &= in_range

    # Orientation check
    if target_orientation_xyzw is not None:
        target_quat = torch.tensor(target_orientation_xyzw, device=device, dtype=torch.float32).unsqueeze(0)

        # Formula: |<q1, q2>| > cos(tolerance / 2)
        quat_dot = torch.sum(object_root_quat_w * target_quat, dim=-1)
        abs_dot = torch.abs(quat_dot)
        min_cos = math.cos(target_orientation_tolerance_rad / 2.0)

        ori_success = abs_dot >= min_cos
        success &= ori_success

    return success


def root_height_below_minimum_multi_objects(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("robot")],
) -> torch.Tensor:
    """Terminate when any asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    outs = [
        root_height_below_minimum(env=env, minimum_height=minimum_height, asset_cfg=asset_cfg)
        for asset_cfg in asset_cfg_list
    ]
    outs_tensor = torch.stack(outs, dim=0)  # [X, N]
    terminated = outs_tensor.any(dim=0)  # [N], bool
    return terminated
