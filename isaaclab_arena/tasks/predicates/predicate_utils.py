# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

import warp as wp
from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


@dataclass(frozen=True, slots=True, repr=False)
class ArenaAssetHandle:
    """Keep an Arena asset by identity when manager configurations are copied.

    Termination configurations are copied before per-environment variants and
    relation-solved poses are finalized. Preserving this handle by identity keeps
    predicates connected to the original asset and its finalized bounds metadata.
    """

    asset: ObjectBase

    def __deepcopy__(self, memo: dict[int, object]) -> ArenaAssetHandle:
        memo[id(self)] = self
        return self

    def __repr__(self) -> str:
        return f"{type(self).__name__}(asset={self.asset.name!r})"


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
    """Get the root position of a rigid object in the world frame."""
    return wp.to_torch(get_rigid_object(env, name).data.root_pos_w)


def get_root_lin_vel_w(env, name: str) -> torch.Tensor:
    """Get the root linear velocity of a rigid object in the world frame."""
    return wp.to_torch(get_rigid_object(env, name).data.root_lin_vel_w)


def get_root_ang_vel_w(env, name: str) -> torch.Tensor:
    """Get the root angular velocity of a rigid object in the world frame."""
    return wp.to_torch(get_rigid_object(env, name).data.root_ang_vel_w)


def select(result: torch.Tensor, env_id: int | None) -> torch.Tensor:
    """Return the entry at ``env_id`` if requested, otherwise the full vector."""
    if env_id is None:
        return result
    return result[env_id]
