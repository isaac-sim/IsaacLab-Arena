# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.scene import InteractiveScene

    from isaaclab_arena.environments.arena_world import ArenaWorld


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


def step_physics(env: ManagerBasedEnv, num_steps: int, render: bool = False) -> None:
    """Advance physics, optionally rendering each step.

    Args:
        env: The Isaac Lab env to step.
        num_steps: Number of physics steps to advance.
        render: When True, render each step so the settle is visible in the GUI. Defaults to
            False (physics-only).
    """
    dt = env.unwrapped.sim.get_physics_dt()
    # Write scene data each substep while bypassing env.step() and its episode recorders.
    for _ in range(num_steps):
        env.unwrapped.scene.write_data_to_sim()
        env.unwrapped.sim.step(render=render)
        env.unwrapped.scene.update(dt)


def are_all_objects_settled_per_env(
    env: ManagerBasedEnv,
    env_ids: list[int],
    object_names: list[str],
    lin_vel_thresh: float,
    ang_vel_thresh: float,
) -> list[bool]:
    """Settled check for a batch of envs, reading each object's velocity once per env in parallel."""
    if not env_ids:
        return []
    arena_env = env.unwrapped
    settled_mask = compute_objects_settled_mask(
        arena_env.arena_world,
        arena_env.scene,
        object_names,
        lin_vel_thresh,
        ang_vel_thresh,
    )
    environment_ids = torch.as_tensor(env_ids, device=arena_env.device)
    return settled_mask[environment_ids].tolist()


def get_pose_drift(initial: torch.Tensor, current: torch.Tensor) -> tuple[float, float] | None:
    """Measure the maximum translation and rotation between corresponding poses.

    Args:
        initial: Initial xyz/xyzw poses shaped (..., 7), with positions in metres.
        current: Current poses with matching shape, expressed in the same frame.

    Returns:
        Maximum translation in metres and maximum rotation in degrees, reduced
        independently over all poses. Returns None if either input contains
        non-finite values; unchanged finite poses return (0.0, 0.0).
    """
    from isaaclab.utils.math import quat_error_magnitude

    if not torch.isfinite(initial).all() or not torch.isfinite(current).all():
        return None
    distance = float((current[..., :3] - initial[..., :3]).norm(dim=-1).max())
    angle = float(torch.rad2deg(quat_error_magnitude(current[..., 3:], initial[..., 3:])).max())
    return distance, angle
