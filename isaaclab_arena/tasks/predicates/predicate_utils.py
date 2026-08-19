# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject

from isaaclab_arena.scene.object_state import get_env, object_state


def get_rigid_object(env, name: str) -> RigidObject:
    """Get a rigid object from the env's scene."""
    return get_env(env).scene[name]


def get_root_pos_w(env, name: str) -> torch.Tensor:
    """Get the aggregate object position in the world frame."""
    position_w = object_state(env, name).aggregate.position_w
    assert position_w is not None, f"Object {name!r} has no aggregate position"
    return position_w


def get_root_lin_vel_w(env, name: str) -> torch.Tensor:
    """Get the aggregate object linear velocity in the world frame."""
    linear_velocity_w = object_state(env, name).aggregate.linear_velocity_w
    assert linear_velocity_w is not None, f"Object {name!r} has no aggregate linear velocity"
    return linear_velocity_w


def get_root_ang_vel_w(env, name: str, required: bool = True) -> torch.Tensor:
    """Get aggregate angular velocity, optionally returning zero when unavailable."""
    state = object_state(env, name).aggregate
    if state.angular_velocity_w is not None:
        return state.angular_velocity_w
    assert not required, f"Object {name!r} has no aggregate angular velocity"
    assert state.position_w is not None, f"Object {name!r} has no aggregate position"
    return torch.zeros_like(state.position_w)


def get_max_point_speed_w(env, name: str) -> torch.Tensor:
    """Get maximum point speed, using root speed for non-deformable objects."""
    state = object_state(env, name)
    kinematics = state.points if state.points is not None else state.aggregate
    assert kinematics.linear_velocity_w is not None, f"Object {name!r} has no linear velocity"
    speed = torch.linalg.vector_norm(kinematics.linear_velocity_w, dim=-1)
    return speed.amax(dim=1) if state.points is not None else speed


def select(result: torch.Tensor, env_id: int | None) -> torch.Tensor:
    """Return the entry at ``env_id`` if requested, otherwise the full vector."""
    if env_id is None:
        return result
    return result[env_id]
