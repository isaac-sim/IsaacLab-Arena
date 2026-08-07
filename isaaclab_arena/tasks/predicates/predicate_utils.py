# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject

from isaaclab_arena.scene.object_geometry import object_geometry
from isaaclab_arena.scene.object_state import get_env, object_state


def get_rigid_object(env, name: str) -> RigidObject:
    """Get a rigid object from the env's scene."""
    return get_env(env).scene[name]


def get_root_pos_w(env, name: str) -> torch.Tensor:
    """Get the root (or deformable centroid) position in the world frame.

    This compatibility wrapper delegates to the shared object-state view.
    """
    return object_state(env, name).position_w()


def get_root_lin_vel_w(env, name: str) -> torch.Tensor:
    """Get the root (or deformable centroid) linear velocity in the world frame.

    This compatibility wrapper delegates to the shared object-state view.
    """
    return object_state(env, name).linear_velocity_w()


def get_root_ang_vel_w(env, name: str, required: bool = True) -> torch.Tensor:
    """Get the root angular velocity of an object in the world frame when available."""
    return object_state(env, name).angular_velocity_w(required=required)


def get_max_point_speed_w(env, name: str) -> torch.Tensor:
    """Get max representative point speed for an object in world frame."""
    return object_geometry(env, name).max_point_speed()


def select(result: torch.Tensor, env_id: int | None) -> torch.Tensor:
    """Return the entry at ``env_id`` if requested, otherwise the full vector."""
    if env_id is None:
        return result
    return result[env_id]
