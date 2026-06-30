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
    return get_env(env).scene[name]


def get_root_pos_w(env, name: str) -> torch.Tensor:
    return wp.to_torch(get_rigid_object(env, name).data.root_pos_w)


def get_root_lin_vel_w(env, name: str) -> torch.Tensor:
    return wp.to_torch(get_rigid_object(env, name).data.root_lin_vel_w)


def select(result: torch.Tensor, env_id: int | None) -> torch.Tensor:
    """Return the scalar entry at ``env_id`` if requested, otherwise the full vector."""
    if env_id is None:
        return result
    return result[env_id]
