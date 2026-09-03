# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Predicates for inserting a rigid gear onto a target peg."""

from __future__ import annotations

import math
import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.assets.object_base import ObjectBase


def gear_is_inserted(
    env: ManagerBasedEnv,
    gear_cfg: SceneEntityCfg,
    insertion_target: ObjectBase,
    gear_insertion_offset_xyz: tuple[float, float, float],
    xy_threshold: float,
    z_threshold: float,
    upright_axis_threshold_deg: float,
    linear_velocity_threshold: float,
    angular_velocity_threshold: float,
    support_z_threshold: float,
) -> torch.Tensor:
    """Return which environments contain a seated, upright, and settled gear."""
    import isaaclab.utils.math as math_utils
    import warp as wp

    base_env = env.unwrapped
    assert gear_cfg.name in base_env.scene.rigid_objects, f"Gear insertion requires rigid object '{gear_cfg.name}'."
    gear: RigidObject = base_env.scene[gear_cfg.name]

    gear_position = wp.to_torch(gear.data.root_link_pos_w)
    gear_orientation = wp.to_torch(gear.data.root_link_quat_w)
    gear_velocity = wp.to_torch(gear.data.root_com_vel_w)
    target_pose = insertion_target.get_object_pose(base_env, is_relative=False)
    target_position = target_pose[:, :3]
    target_orientation = target_pose[:, 3:7]

    gear_offset = torch.as_tensor(
        gear_insertion_offset_xyz,
        device=base_env.device,
        dtype=gear_position.dtype,
    ).expand(base_env.num_envs, -1)
    gear_insertion_position = gear_position + math_utils.quat_apply(gear_orientation, gear_offset)
    position_error = gear_insertion_position - target_position

    xy_error = torch.linalg.vector_norm(position_error[:, :2], dim=-1)
    z_error = torch.abs(position_error[:, 2])
    height_above_target = torch.clamp(position_error[:, 2], min=0.0)
    local_up = torch.tensor((0.0, 0.0, 1.0), device=base_env.device, dtype=gear_position.dtype).expand(
        base_env.num_envs, -1
    )
    gear_up = math_utils.quat_apply(gear_orientation, local_up)
    target_up = math_utils.quat_apply(target_orientation, local_up)
    upright = (gear_up * target_up).sum(dim=-1) >= math.cos(math.radians(upright_axis_threshold_deg))
    settled = (torch.linalg.vector_norm(gear_velocity[:, :3], dim=-1) <= linear_velocity_threshold) & (
        torch.linalg.vector_norm(gear_velocity[:, 3:], dim=-1) <= angular_velocity_threshold
    )

    return (
        (xy_error <= xy_threshold)
        & (z_error <= z_threshold)
        & (height_above_target <= support_z_threshold)
        & upright
        & settled
    )
