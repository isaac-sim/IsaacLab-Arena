# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch

import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.terminations import root_height_below_minimum
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor
from isaaclab.utils.math import combine_frame_transforms


# NOTE(alexmillane, 2025.09.15): The velocity threshold is set high because some stationary
# seem to generate a "small" velocity.
def object_on_destination(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object_contact_sensor"),
    force_threshold: float = 1.0,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    object: RigidObject = env.unwrapped.scene[object_cfg.name]
    sensor: ContactSensor = env.unwrapped.scene[contact_sensor_cfg.name]

    # force_matrix_w shape is (N, B, M, 3), where N is the number of sensors, B is number of bodies in each sensor
    # and ``M`` is the number of filtered bodies.
    # We assume B = 1 and M = 1
    assert sensor.data.force_matrix_w.shape[2] == 1
    assert sensor.data.force_matrix_w.shape[1] == 1
    # NOTE(alexmillane, 2025-08-04): We expect the binary flags to have shape (N, )
    # where N is the number of envs.
    force_matrix_norm = torch.norm(wp.to_torch(sensor.data.force_matrix_w), dim=-1).reshape(-1)
    force_above_threshold = force_matrix_norm > force_threshold

    velocity_w = wp.to_torch(object.data.root_lin_vel_w)
    velocity_w_norm = torch.norm(velocity_w, dim=-1)
    velocity_below_threshold = velocity_w_norm < velocity_threshold

    condition_met = torch.logical_and(force_above_threshold, velocity_below_threshold)
    return condition_met


def objects_on_destinations(
    env: ManagerBasedRLEnv,
    object_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("pick_up_object")],
    contact_sensor_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("pick_up_object_contact_sensor")],
    force_threshold: float = 1.0,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    """Multi-object version of `object_on_destination`.

    Returns True only when ALL objects in the list satisfy the destination condition.
    See `object_on_destination` for details on the single-object logic.
    """
    condition_met = torch.ones((env.unwrapped.num_envs), device=env.unwrapped.device, dtype=torch.bool)
    for object_cfg, contact_sensor_cfg in zip(object_cfg_list, contact_sensor_cfg_list):
        single_condition = object_on_destination(
            env=env,
            object_cfg=object_cfg,
            contact_sensor_cfg=contact_sensor_cfg,
            force_threshold=force_threshold,
            velocity_threshold=velocity_threshold,
        )
        condition_met = torch.logical_and(condition_met, single_condition)
    return condition_met


def objects_in_proximity(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_object_cfg: SceneEntityCfg,
    max_y_separation: float,
    max_x_separation: float,
    max_z_separation: float,
) -> torch.Tensor:
    """Determine if two objects are within a certain proximity of each other.

    Returns:
        Boolean tensor indicating when objects are within a certain proximity of each other.
    """
    # Get object entities from the scene
    object: RigidObject = env.scene[object_cfg.name]
    target_object: RigidObject = env.scene[target_object_cfg.name]

    # Get positions relative to environment origin
    object_pos = wp.to_torch(object.data.root_pos_w) - env.scene.env_origins
    target_object_pos = wp.to_torch(target_object.data.root_pos_w) - env.scene.env_origins

    # object to target object
    x_separation = torch.abs(object_pos[:, 0] - target_object_pos[:, 0])
    y_separation = torch.abs(object_pos[:, 1] - target_object_pos[:, 1])
    z_separation = torch.abs(object_pos[:, 2] - target_object_pos[:, 2])

    done = x_separation < max_x_separation
    done = torch.logical_and(done, y_separation < max_y_separation)
    done = torch.logical_and(done, z_separation < max_z_separation)

    return done


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
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("robot")]
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


def gear_mesh_insertion_success(
    env: ManagerBasedRLEnv,
    held_object_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
    fixed_object_cfg: SceneEntityCfg = SceneEntityCfg("gears_and_base"),
    gear_base_offset: list[float] = [2.025e-2, 0.0, 0.0],
    held_gear_base_offset: list[float] | None = None,
    gear_peg_height: float = 0.02,
    success_z_fraction: float = 0.80,
    xy_threshold: float = 0.0025,
    rl_training: bool = False,
) -> torch.Tensor:
    """Terminate when the gear is inserted onto the peg to the required depth.

    Checks that the held gear's base (root + held_gear_base_offset in gear frame)
    is centered on the peg (XY) and lowered past a fraction of the peg height (Z).
    Peg position is fixed_asset pose + gear_base_offset in the fixed asset's local frame.

    When ``rl_training`` is True, always returns False (no early termination)
    but the term stays registered so that ``SuccessRateMetric`` can query it.
    """
    if rl_training:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    held_object: RigidObject = env.scene[held_object_cfg.name]
    fixed_object: RigidObject = env.scene[fixed_object_cfg.name]

    held_root = wp.to_torch(held_object.data.root_pos_w) - env.scene.env_origins
    held_quat = wp.to_torch(held_object.data.root_quat_w)
    h_offset = held_gear_base_offset if held_gear_base_offset is not None else gear_base_offset
    held_off = torch.tensor(h_offset, device=env.device, dtype=torch.float32).unsqueeze(0).expand(env.num_envs, 3)
    held_base_pos = held_root + math_utils.quat_apply(held_quat, held_off)

    fixed_pos = wp.to_torch(fixed_object.data.root_pos_w) - env.scene.env_origins
    fixed_quat = wp.to_torch(fixed_object.data.root_quat_w)
    offset = torch.tensor(gear_base_offset, device=env.device, dtype=torch.float32).unsqueeze(0).expand(env.num_envs, 3)
    peg_pos = fixed_pos + math_utils.quat_apply(fixed_quat, offset)

    xy_dist = torch.linalg.vector_norm(peg_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
    is_centered = xy_dist < xy_threshold

    z_disp = held_base_pos[:, 2] - peg_pos[:, 2]
    height_threshold = gear_peg_height * success_z_fraction
    is_inserted = z_disp < height_threshold

    return torch.logical_and(is_centered, is_inserted)


def gear_dropped_from_gripper(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_name: str = "panda_hand",
    distance_threshold: float = 0.15,
) -> torch.Tensor:
    """Reset when the gear has fallen too far from the end-effector."""
    gear: RigidObject = env.scene[gear_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    eef_indices, _ = robot.find_bodies([ee_body_name])
    ee_pos = wp.to_torch(robot.data.body_pos_w)[:, eef_indices[0]]
    gear_pos = wp.to_torch(gear.data.root_pos_w)
    distance = torch.norm(gear_pos - ee_pos, dim=-1)
    return distance > distance_threshold


def gear_orientation_exceeded(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_name: str = "panda_hand",
    roll_threshold_deg: float = 15.0,
    pitch_threshold_deg: float = 15.0,
) -> torch.Tensor:
    """Reset when the gear has tilted too far relative to the end-effector."""
    if not hasattr(env, "_initial_gear_ee_relative_quat"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    gear: RigidObject = env.scene[gear_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    eef_indices, _ = robot.find_bodies([ee_body_name])
    eef_quat = wp.to_torch(robot.data.body_quat_w)[:, eef_indices[0]]
    current_relative = math_utils.quat_mul(wp.to_torch(gear.data.root_quat_w), math_utils.quat_conjugate(eef_quat))

    initial_relative = env._initial_gear_ee_relative_quat
    deviation = math_utils.quat_mul(current_relative, math_utils.quat_conjugate(initial_relative))

    roll, pitch, _yaw = math_utils.euler_xyz_from_quat(deviation)

    roll_limit = torch.deg2rad(torch.tensor(roll_threshold_deg, device=env.device))
    pitch_limit = torch.deg2rad(torch.tensor(pitch_threshold_deg, device=env.device))
    return (torch.abs(roll) > roll_limit) | (torch.abs(pitch) > pitch_limit)
