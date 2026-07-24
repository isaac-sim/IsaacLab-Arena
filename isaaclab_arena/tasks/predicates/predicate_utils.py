# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import warp as wp
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
    """Get the root (or deformable centroid) position in the world frame.

    ``root_pos_w`` is exposed identically by rigid and deformable object data, so this works for both.
    """
    return wp.to_torch(get_env(env).scene[name].data.root_pos_w)


def get_root_lin_vel_w(env, name: str) -> torch.Tensor:
    """Get the root (or deformable centroid) linear velocity in the world frame.

    Rigid objects expose ``root_lin_vel_w``; deformable objects expose the (linear) ``root_vel_w``.
    Dispatch is on which attribute the entity's data provides, so it holds across physics backends.
    """
    data = get_env(env).scene[name].data
    velocity = data.root_lin_vel_w if hasattr(data, "root_lin_vel_w") else data.root_vel_w
    return wp.to_torch(velocity)


def get_root_ang_vel_w(env, name: str) -> torch.Tensor:
    """Get the root angular velocity of a rigid object in the world frame."""
    return wp.to_torch(get_rigid_object(env, name).data.root_ang_vel_w)


def select(result: torch.Tensor, env_id: int | None) -> torch.Tensor:
    """Return the entry at ``env_id`` if requested, otherwise the full vector."""
    if env_id is None:
        return result
    return result[env_id]
