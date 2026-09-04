# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject


def get_env(env):
    """Resolve to the unwrapped manager-based env regardless of wrapper depth."""
    seen = set()
    while hasattr(env, "unwrapped") and env.unwrapped is not env and id(env) not in seen:
        seen.add(id(env))
        env = env.unwrapped
    return env


def get_rigid_object(env, name: str) -> RigidObject:
    """Get a rigid object from the env's scene."""
    return get_env(env).scene[name]


def get_root_pos_w(env, name: str) -> torch.Tensor:
    """Get the aggregate object position in the world frame."""
    return get_env(env).arena_world.get_pose_w(name)[:, :3]


def get_root_lin_vel_w(env, name: str) -> torch.Tensor:
    """Get the aggregate object linear velocity in the world frame."""
    return get_env(env).arena_world.get_root_linear_velocity_w(name)


def get_root_ang_vel_w(env, name: str, required: bool = True) -> torch.Tensor:
    """Get root angular velocity, optionally returning zero for a deformable object."""
    unwrapped_env = get_env(env)
    if name in unwrapped_env.scene.deformable_objects:
        assert not required, f"Deformable object {name!r} has no aggregate angular velocity"
        return torch.zeros_like(unwrapped_env.arena_world.get_root_linear_velocity_w(name))
    return get_rigid_object(unwrapped_env, name).data.root_ang_vel_w.torch


def get_max_point_speed_w(env, name: str) -> torch.Tensor:
    """Get maximum nodal speed for a deformable, or root speed for a rigid object."""
    unwrapped_env = get_env(env)
    if name in unwrapped_env.scene.deformable_objects:
        nodal_velocity_w = unwrapped_env.scene.deformable_objects[name].data.nodal_vel_w.torch
        return torch.linalg.vector_norm(nodal_velocity_w, dim=-1).amax(dim=1)
    return torch.linalg.vector_norm(get_root_lin_vel_w(unwrapped_env, name), dim=-1)


def select(result: torch.Tensor, env_id: int | None) -> torch.Tensor:
    """Return the entry at ``env_id`` if requested, otherwise the full vector."""
    if env_id is None:
        return result
    return result[env_id]
