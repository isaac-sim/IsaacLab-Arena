# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor


# NOTE(alexmillane, 2025.09.15): The velocity threshold is set high because some stationary
# seem to generate a "small" velocity.
def object_on_destination(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object_contact_sensor"),
    force_threshold: float = 1.0,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    sensor: ContactSensor = env.scene[contact_sensor_cfg.name]

    # force_matrix_w shape is (N, B, M, 3), where N is the number of sensors, B is number of bodies in each sensor
    # and ``M`` is the number of filtered bodies.
    # We assume B = 1 and M = 1
    assert sensor.data.force_matrix_w.shape[2] == 1
    assert sensor.data.force_matrix_w.shape[1] == 1
    # NOTE(alexmillane, 2025-08-04): We expect the binary flags to have shape (N, )
    # where N is the number of envs.
    force_matrix_norm = torch.norm(sensor.data.force_matrix_w.clone(), dim=-1).reshape(-1)
    force_above_threshold = force_matrix_norm > force_threshold

    velocity_w = object.data.root_lin_vel_w
    velocity_w_norm = torch.norm(velocity_w, dim=-1)
    velocity_below_threshold = velocity_w_norm < velocity_threshold

    condition_met = torch.logical_and(force_above_threshold, velocity_below_threshold)
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
    object_pos = object.data.root_pos_w - env.scene.env_origins

    # Get positions relative to environment origin
    object_pos = object.data.root_pos_w - env.scene.env_origins
    target_object_pos = target_object.data.root_pos_w - env.scene.env_origins

    # object to target object
    x_separation = torch.abs(object_pos[:, 0] - target_object_pos[:, 0])
    y_separation = torch.abs(object_pos[:, 1] - target_object_pos[:, 1])
    z_separation = torch.abs(object_pos[:, 2] - target_object_pos[:, 2])

    done = x_separation < max_x_separation
    done = torch.logical_and(done, y_separation < max_y_separation)
    done = torch.logical_and(done, z_separation < max_z_separation)

    return done


def adjust_pose_task_termination(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_thresholds: dict | None = None,
) -> torch.Tensor:
    """Terminate when the object's pose is within the thresholds (BBox + Orientation).

    Args:
        env: The RL environment instance.
        object_cfg: The configuration of the object to track.
        object_thresholds: Configuration dict following the BBox schema:
            {
                "success_zone": {
                    "x_range": [0.4, 0.6],  # Optional
                    "y_range": [-0.1, 0.1], # Optional
                    "z_range": [0.0, 0.5]   # Optional
                },
                "orientation": {
                    "target": [w, x, y, z],
                    "tolerance_rad": 0.1
                }
            }
    Returns:
        A boolean tensor of shape (num_envs, )
    """
    object_instance: RigidObject = env.scene[object_cfg.name]
    object_root_pos_w = object_instance.data.root_pos_w
    object_root_quat_w = object_instance.data.root_quat_w

    device = env.device
    num_envs = env.num_envs

    if not object_thresholds:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)

    success = torch.ones(num_envs, dtype=torch.bool, device=device)

    zone_cfg = object_thresholds.get("success_zone", {})

    # X Axis Check
    if "x_range" in zone_cfg:
        x_min, x_max = zone_cfg["x_range"]
        in_x = (object_root_pos_w[:, 0] >= x_min) & (object_root_pos_w[:, 0] <= x_max)
        success &= in_x

    # Y Axis Check
    if "y_range" in zone_cfg:
        y_min, y_max = zone_cfg["y_range"]
        in_y = (object_root_pos_w[:, 1] >= y_min) & (object_root_pos_w[:, 1] <= y_max)
        success &= in_y

    # Z Axis Check
    if "z_range" in zone_cfg:
        z_min, z_max = zone_cfg["z_range"]
        in_z = (object_root_pos_w[:, 2] >= z_min) & (object_root_pos_w[:, 2] <= z_max)
        success &= in_z

    # Orientation Check
    ori_cfg = object_thresholds.get("orientation")
    if ori_cfg:
        target_list = ori_cfg.get("target")
        tol_rad = ori_cfg.get("tolerance_rad", 0.1)

        if target_list is not None:
            target_quat = torch.tensor(target_list, device=device, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 4]

            # Formula: |<q1, q2>| > cos(tolerance / 2)
            quat_dot = torch.sum(object_root_quat_w * target_quat, dim=-1)
            abs_dot = torch.abs(quat_dot)
            min_cos = math.cos(tol_rad / 2.0)

            ori_success = abs_dot >= min_cos
            success &= ori_success

    return success


def object_above(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    maximum_height: float,
) -> torch.Tensor:
    """Determine if the object is lifted above a certain height."""
    object: RigidObject = env.scene[object_cfg.name]
    return object.data.root_pos_w[:, 2] > maximum_height
