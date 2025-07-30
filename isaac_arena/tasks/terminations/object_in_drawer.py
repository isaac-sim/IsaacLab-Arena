# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# TODO(cvolk): Generalize this to target_object_in()
def object_in_drawer(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object"),
    drawer_bottom_cfg: SceneEntityCfg = SceneEntityCfg("destination_object"),
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
) -> torch.Tensor:
    """Check if an object is dropped by the specified robot."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    drawer_bottom: RigidObject = env.scene[drawer_bottom_cfg.name]

    # Get the object and drawer bottom positions
    object_pos = object.data.root_pos_w
    drawer_bottom_pos = drawer_bottom.data.root_pos_w

    # Get the bounds wrt to the drawer bottom
    # This is a bit hacky, but it works for now
    # So if we have multiple environments we need to handle them.
    # TODO(remos): read size from env config
    drawer_size = torch.tensor([0.4, 0.65], device=env.device)
    DRAWER_HEIGHT = 0.1
    bounds_x_lower = drawer_bottom_pos[:, 0] - drawer_size[0] / 2
    bounds_x_upper = drawer_bottom_pos[:, 0] + drawer_size[0] / 2
    bounds_y_lower = drawer_bottom_pos[:, 1] - drawer_size[1] / 2
    bounds_y_upper = drawer_bottom_pos[:, 1] + drawer_size[1] / 2
    bounds_z_lower = drawer_bottom_pos[:, 2] - 1e-2  # 1 cm tolerance
    bounds_z_upper = drawer_bottom_pos[:, 2] + DRAWER_HEIGHT
    object_pos_in_bounds_x = (object_pos[:, 0] > bounds_x_lower) & (object_pos[:, 0] < bounds_x_upper)
    object_pos_in_bounds_y = (object_pos[:, 1] > bounds_y_lower) & (object_pos[:, 1] < bounds_y_upper)
    object_pos_in_bounds_z = (object_pos[:, 2] > bounds_z_lower) & (object_pos[:, 2] < bounds_z_upper)
    # Combine them to check if each environment’s object is fully in bounds
    object_in_drawer = object_pos_in_bounds_x & object_pos_in_bounds_y & object_pos_in_bounds_z

    # We also want to check if the gripper is in open position
    # Check this with some tolerance
    gripper_open_val = gripper_open_val.to(robot.data.joint_pos.device)
    gripper_open = torch.logical_and(
        torch.abs(robot.data.joint_pos[:, -2] - gripper_open_val) < 0.005,
        torch.abs(robot.data.joint_pos[:, -1] - gripper_open_val) < 0.005,
    )

    object_in_drawer = torch.logical_and(object_in_drawer, gripper_open)

    return object_in_drawer
