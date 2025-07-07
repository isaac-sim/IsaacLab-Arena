# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_frame_pos(
    env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(
    env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


def gripper_pos(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    diff_threshold: float = 0.13,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    gripper_threshold: float = 0.005,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)
    grasped = torch.logical_and(
        pose_diff < diff_threshold,
        torch.abs(robot.data.joint_pos[:, -1] - gripper_open_val.to(env.device))
        > gripper_threshold,
    )
    grasped = torch.logical_and(
        grasped,
        torch.abs(robot.data.joint_pos[:, -2] - gripper_open_val.to(env.device))
        > gripper_threshold,
    )

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = (
        torch.max(torch.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids], dim=-1), dim=1)[
            0
        ]
        > contact_threshold
    )
    grasped = torch.logical_and(grasped, is_contact)

    return grasped
