# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Predicates for inserting a rigid gear onto a target peg."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.tasks.predicates.spatial import (
    depth_in_range,
    lateral_in_proximity,
    tilt_axis_aligned,
    velocity_below_threshold,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


def gear_is_inserted(
    env: IsaacLabArenaManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg,
    insertion_target_cfg: SceneEntityCfg,
    gear_insertion_offset_xyz: tuple[float, float, float],
    xy_threshold: float,
    z_threshold: float,
    upright_axis_threshold_deg: float,
    linear_velocity_threshold: float,
    angular_velocity_threshold: float,
    support_z_threshold: float,
) -> torch.Tensor:
    """Return which environments contain a seated, upright, and settled gear.

    Args:
        env: Arena environment providing live poses and velocities.
        gear_cfg: Rigid gear whose insertion point is checked.
        insertion_target_cfg: Scene frame defining the seated pose and insertion axis.
        gear_insertion_offset_xyz: Insertion point expressed in the gear frame, in meters.
        xy_threshold: Maximum lateral error measured in the target frame, in meters.
        z_threshold: Maximum absolute depth error along the target's +Z axis, in meters.
        upright_axis_threshold_deg: Maximum angle between gear and target +Z axes, in degrees.
        linear_velocity_threshold: Maximum gear root linear speed, in meters per second.
        angular_velocity_threshold: Maximum gear angular speed, in radians per second.
        support_z_threshold: Maximum height above the seated target, in meters.

    Returns:
        One Boolean insertion result per environment.
    """
    base_env = env.unwrapped
    assert gear_cfg.name in base_env.scene.rigid_objects, f"Gear insertion requires rigid object '{gear_cfg.name}'."
    target_params = {
        "subject_name": gear_cfg.name,
        "receiver_name": insertion_target_cfg.name,
        "target_offset_xyz": (0.0, 0.0, 0.0),
        "subject_offset_xyz": gear_insertion_offset_xyz,
    }
    aligned = lateral_in_proximity(base_env, **target_params, tolerance_lateral=xy_threshold)
    seated = depth_in_range(
        base_env,
        **target_params,
        depth_min=-z_threshold,
        depth_max=min(z_threshold, support_z_threshold),
    )
    upright = tilt_axis_aligned(
        base_env,
        subject_name=gear_cfg.name,
        receiver_name=insertion_target_cfg.name,
        max_tilt_rad=math.radians(upright_axis_threshold_deg),
    )
    settled = velocity_below_threshold(
        base_env,
        subject_name=gear_cfg.name,
        linear_velocity_threshold=linear_velocity_threshold,
        angular_velocity_threshold=angular_velocity_threshold,
    )
    return aligned & seated & upright & settled
