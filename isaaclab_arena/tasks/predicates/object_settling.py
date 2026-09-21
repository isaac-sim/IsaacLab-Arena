# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Instantaneous object-settling checks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene

    from isaaclab_arena.environments.arena_world import ArenaWorld
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


DEFAULT_LINEAR_VELOCITY_THRESHOLD = 1e-2
DEFAULT_ANGULAR_VELOCITY_THRESHOLD = 5e-2


def compute_objects_settled_mask(
    arena_world: ArenaWorld,
    scene: InteractiveScene,
    object_names: list[str],
    lin_vel_threshold: float,
    ang_vel_threshold: float,
) -> torch.Tensor:
    """Return a per-env mask that is True when every named object is below velocity thresholds.

    Rigid objects use root linear and angular speed. Deformable objects use the 90th percentile of
    nodal linear speeds and have no angular-speed condition.

    Args:
        arena_world: Arena scene query facade.
        scene: Live scene (used to detect deformable objects).
        object_names: Object scene keys to check.
        lin_vel_threshold: Linear speed threshold in meters per second.
        ang_vel_threshold: Angular speed threshold in radians per second (rigid objects only).

    Returns:
        Boolean mask with shape ``(num_envs,)``.
    """
    if not object_names:
        return torch.ones(scene.num_envs, dtype=torch.bool, device=scene.device)

    per_object_settled = []
    for object_name in object_names:
        if object_name in scene.deformable_objects:
            nodal_velocity_w = arena_world.get_nodal_velocities_w(object_name)
            nodal_speed = torch.linalg.vector_norm(nodal_velocity_w, dim=-1)
            linear_speed = torch.quantile(nodal_speed, q=0.9, dim=1)
            per_object_settled.append(linear_speed < lin_vel_threshold)
            continue
        root_linear_velocity_w = arena_world.get_root_linear_velocity_w(object_name)
        linear_speed = torch.linalg.vector_norm(root_linear_velocity_w, dim=-1)
        root_angular_velocity_w = arena_world.get_root_angular_velocity_w(object_name)
        angular_speed = torch.linalg.vector_norm(root_angular_velocity_w, dim=-1)
        per_object_settled.append((linear_speed < lin_vel_threshold) & (angular_speed < ang_vel_threshold))
    return torch.stack(per_object_settled, dim=0).all(dim=0)


def objects_settled(
    env: IsaacLabArenaManagerBasedRLEnv,
    object_names: list[str],
    lin_vel_threshold: float = DEFAULT_LINEAR_VELOCITY_THRESHOLD,
    ang_vel_threshold: float = DEFAULT_ANGULAR_VELOCITY_THRESHOLD,
) -> torch.Tensor:
    """Check whether every named object is below the velocity thresholds.

    Rigid objects use root linear and angular speed. Deformable objects use the 90th percentile of
    nodal linear speeds and have no angular-speed condition.
    """

    settled = objects_below_velocity_thresholds(
        env,
        object_names=object_names,
        lin_vel_threshold=lin_vel_threshold,
        ang_vel_threshold=ang_vel_threshold,
    )

    return settled


def objects_below_velocity_thresholds(
    env: IsaacLabArenaManagerBasedRLEnv,
    object_names: list[str],
    lin_vel_threshold: float = DEFAULT_LINEAR_VELOCITY_THRESHOLD,
    ang_vel_threshold: float = DEFAULT_ANGULAR_VELOCITY_THRESHOLD,
) -> torch.Tensor:
    """Return whether all named objects meet their speed limits in each environment.

    This check does not record resting positions or count consecutive steps.
    """

    return compute_objects_settled_mask(
        env.arena_world,
        env.scene,
        object_names,
        lin_vel_threshold,
        ang_vel_threshold,
    )
