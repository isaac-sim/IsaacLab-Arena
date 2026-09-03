# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Predicates for inserting a DisplayPort plug into its socket."""

from __future__ import annotations

import math
import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.assets.object_base import ObjectBase


def displayport_plug_is_inserted(
    env: ManagerBasedEnv,
    plug_cfg: SceneEntityCfg,
    insertion_target: ObjectBase,
    plug_mating_offset_xyz: tuple[float, float, float],
    position_threshold: float,
    orientation_threshold_deg: float,
    linear_velocity_threshold: float,
    angular_velocity_threshold: float,
) -> torch.Tensor:
    """Return which environments contain an aligned and settled plug."""
    import isaaclab.utils.math as math_utils
    import warp as wp

    base_env = env.unwrapped
    assert (
        plug_cfg.name in base_env.scene.rigid_objects
    ), f"DisplayPort insertion requires rigid object '{plug_cfg.name}'."
    plug: RigidObject = base_env.scene[plug_cfg.name]

    plug_position = wp.to_torch(plug.data.root_link_pos_w)
    plug_orientation = wp.to_torch(plug.data.root_link_quat_w)
    plug_velocity = wp.to_torch(plug.data.root_com_vel_w)
    target_pose = insertion_target.get_object_pose(base_env, is_relative=False)

    plug_offset = torch.as_tensor(
        plug_mating_offset_xyz,
        device=base_env.device,
        dtype=plug_position.dtype,
    ).expand(base_env.num_envs, -1)
    plug_mating_position = plug_position + math_utils.quat_apply(plug_orientation, plug_offset)
    position_error = torch.linalg.vector_norm(plug_mating_position - target_pose[:, :3], dim=-1)

    quaternion_alignment = torch.abs(torch.sum(plug_orientation * target_pose[:, 3:7], dim=-1))
    orientation_error = 2.0 * torch.acos(torch.clamp(quaternion_alignment, min=0.0, max=1.0))
    orientation_threshold = math.radians(orientation_threshold_deg)
    settled = (torch.linalg.vector_norm(plug_velocity[:, :3], dim=-1) <= linear_velocity_threshold) & (
        torch.linalg.vector_norm(plug_velocity[:, 3:], dim=-1) <= angular_velocity_threshold
    )

    return (position_error <= position_threshold) & (orientation_error <= orientation_threshold) & settled
