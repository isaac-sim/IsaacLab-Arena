# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import warp as wp
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor


def _to_torch(data: torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return wp.to_torch(data)


def object_near_fixed_position(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_position: tuple[float, float, float],
    max_x_separation: float,
    max_y_separation: float,
    max_z_separation: float,
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    object_pos = _to_torch(object.data.root_pos_w) - env.scene.env_origins
    target_pos = torch.tensor(target_position, device=env.device, dtype=torch.float32).unsqueeze(0)
    done = torch.abs(object_pos[:, 0] - target_pos[:, 0]) < max_x_separation
    done &= torch.abs(object_pos[:, 1] - target_pos[:, 1]) < max_y_separation
    done &= torch.abs(object_pos[:, 2] - target_pos[:, 2]) < max_z_separation
    return done


def object_lifted(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return _to_torch(asset.data.root_pos_w)[:, 2] > minimum_height


def is_upright(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    threshold: float = 0.01,
    z_axis_up: bool = False,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    quat = _to_torch(asset.data.root_quat_w)
    z_dot_up = 1.0 - 2.0 * (quat[:, 0] ** 2 + quat[:, 1] ** 2)
    if z_axis_up:
        return z_dot_up >= threshold
    return torch.abs(z_dot_up) <= threshold


def is_static(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    lin_vel_threshold: float = 0.15,
    ang_vel_threshold: float = 0.15,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_norm = torch.norm(_to_torch(asset.data.root_lin_vel_w), dim=-1)
    ang_vel_norm = torch.norm(_to_torch(asset.data.root_ang_vel_w), dim=-1)
    return (lin_vel_norm <= lin_vel_threshold) & (ang_vel_norm <= ang_vel_threshold)


def is_in_contact(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    force_threshold: float = 0.95,
) -> torch.Tensor:
    sensor: ContactSensor = env.unwrapped.scene[contact_sensor_cfg.name]
    fm = sensor.data.force_matrix_w
    if fm is not None:
        force_norm = torch.norm(_to_torch(fm), dim=-1).reshape(env.unwrapped.num_envs, -1)
        return force_norm.max(dim=-1).values > force_threshold
    force_norm = torch.norm(_to_torch(sensor.data.net_forces_w), dim=-1).reshape(env.unwrapped.num_envs, -1)
    return force_norm.max(dim=-1).values > force_threshold


G1_FINGER_BODY_NAMES: list[str] = [
    "left_hand_index_0_link",
    "left_hand_index_1_link",
    "left_hand_middle_0_link",
    "left_hand_middle_1_link",
    "left_hand_thumb_0_link",
    "left_hand_thumb_1_link",
    "left_hand_thumb_2_link",
    "right_hand_index_0_link",
    "right_hand_index_1_link",
    "right_hand_middle_0_link",
    "right_hand_middle_1_link",
    "right_hand_thumb_0_link",
    "right_hand_thumb_1_link",
    "right_hand_thumb_2_link",
]


def _gripper_obj_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    ee_body_name: str,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    ee_body_idx = robot.find_bodies(ee_body_name)[0][0]
    ee_pos = _to_torch(robot.data.body_pos_w)[:, ee_body_idx, :]
    obj_pos = _to_torch(obj.data.root_pos_w)
    return torch.norm(ee_pos - obj_pos, dim=-1)


def is_gripper_far(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_names: list[str] | str | None = None,
    threshold: float = 0.4,
) -> torch.Tensor:
    if ee_body_names is None:
        ee_body_names = G1_FINGER_BODY_NAMES
    elif isinstance(ee_body_names, str):
        ee_body_names = [ee_body_names]
    all_far = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for name in ee_body_names:
        all_far &= _gripper_obj_distance(env, object_cfg, robot_cfg, name) > threshold
    return all_far


G1_FINGER_JOINT_REGEX = ".*_hand_.*_joint"


def _fingers_closed(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    finger_joint_regex: str = G1_FINGER_JOINT_REGEX,
    min_deviation: float = 0.1,
    min_fingers_closed: int = 3,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    joint_ids, _ = robot.find_joints(finger_joint_regex)
    pos = _to_torch(robot.data.joint_pos)[:, joint_ids]
    default_pos = _to_torch(robot.data.default_joint_pos)[:, joint_ids]
    closed_count = (torch.abs(pos - default_pos) > min_deviation).sum(dim=-1)
    return closed_count >= min_fingers_closed


def is_grasped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_names: list[str] | str | None = None,
    proximity_threshold: float = 0.18,
    contact_sensor_cfgs: list[SceneEntityCfg] | SceneEntityCfg | None = None,
    contact_force_threshold: float = 1.0,
    require_closed_fingers: bool = True,
    finger_joint_regex: str = G1_FINGER_JOINT_REGEX,
    min_deviation: float = 0.1,
    min_fingers_closed: int = 3,
) -> torch.Tensor:
    if ee_body_names is None:
        ee_body_names = G1_FINGER_BODY_NAMES
    elif isinstance(ee_body_names, str):
        ee_body_names = [ee_body_names]
    if isinstance(contact_sensor_cfgs, SceneEntityCfg):
        contact_sensor_cfgs = [contact_sensor_cfgs] * len(ee_body_names)

    any_grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for i, name in enumerate(ee_body_names):
        gripper_close = _gripper_obj_distance(env, object_cfg, robot_cfg, name) <= proximity_threshold
        if contact_sensor_cfgs is not None:
            gripper_close &= is_in_contact(
                env,
                contact_sensor_cfg=contact_sensor_cfgs[i],
                force_threshold=contact_force_threshold,
            )
        any_grasped |= gripper_close

    if require_closed_fingers:
        any_grasped &= _fingers_closed(
            env,
            robot_cfg=robot_cfg,
            finger_joint_regex=finger_joint_regex,
            min_deviation=min_deviation,
            min_fingers_closed=min_fingers_closed,
        )
    return any_grasped
