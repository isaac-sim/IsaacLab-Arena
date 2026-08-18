# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read-only, backend-neutral runtime state for Arena objects."""

from __future__ import annotations

import torch
from dataclasses import dataclass

import warp as wp

from isaaclab_arena.assets.object_type import ObjectType


@dataclass(frozen=True)
class KinematicState:
    """Kinematic channels for one aggregate object or its points."""

    position_w: torch.Tensor | None = None
    orientation_w: torch.Tensor | None = None
    linear_velocity_w: torch.Tensor | None = None
    angular_velocity_w: torch.Tensor | None = None


@dataclass(frozen=True)
class ObjectState:
    """Aggregate state and optional deformable point state."""

    aggregate: KinematicState
    points: KinematicState | None = None


def get_env(env):
    """Resolve to the unwrapped manager-based environment."""
    seen = set()
    while hasattr(env, "unwrapped") and env.unwrapped is not env and id(env) not in seen:
        seen.add(id(env))
        env = env.unwrapped
    return env


def object_state(env, name: str) -> ObjectState:
    """Return an object's state using its composed Arena object type."""
    env = get_env(env)
    assert name in env.scene.keys(), f"Asset {name!r} not found in scene"
    assert name in env.arena_object_types, f"Arena object type metadata is missing for {name!r}"

    entity = env.scene[name]
    object_type = env.arena_object_types[name]
    if object_type in (ObjectType.RIGID, ObjectType.ARTICULATION):
        return ObjectState(
            aggregate=KinematicState(
                position_w=_to_torch(entity.data.root_pos_w),
                orientation_w=_to_torch(entity.data.root_quat_w),
                linear_velocity_w=_to_torch(entity.data.root_lin_vel_w),
                angular_velocity_w=_to_torch(entity.data.root_ang_vel_w),
            )
        )
    if object_type == ObjectType.DEFORMABLE:
        return ObjectState(
            aggregate=KinematicState(
                position_w=_to_torch(entity.data.root_pos_w),
                linear_velocity_w=_to_torch(entity.data.root_vel_w),
            ),
            points=KinematicState(
                position_w=_to_torch(entity.data.nodal_pos_w),
                linear_velocity_w=_to_torch(entity.data.nodal_vel_w),
            ),
        )
    if object_type == ObjectType.BASE:
        position_w, orientation_w = entity.get_world_poses()
        return ObjectState(
            aggregate=KinematicState(
                position_w=_to_torch(position_w),
                orientation_w=_to_torch(orientation_w),
            )
        )
    raise ValueError(f"Unsupported Arena object type for {name!r}: {object_type!r}")


def _to_torch(value) -> torch.Tensor:
    """Convert an Isaac Lab state buffer to a torch tensor."""
    if isinstance(value, torch.Tensor):
        return value
    tensor = getattr(value, "torch", None)
    if isinstance(tensor, torch.Tensor):
        return tensor
    return wp.to_torch(value)
