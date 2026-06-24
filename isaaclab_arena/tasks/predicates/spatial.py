# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Spatial predicates: position- and height-based scene queries.

All predicates share the signature ``(env, ..., *, env_id=None) -> torch.Tensor``.
- ``env_id=None`` returns a ``(num_envs,)`` bool tensor (vectorized).
- ``env_id=<int>`` returns a 0-D bool tensor for that env (useful for debugging).
Heights are read in world frame.
"""

from __future__ import annotations

import torch

from isaaclab_arena.tasks.predicates.predicate_utils import get_root_lin_vel_w, get_root_pos_w, select


def object_lifted(
    env,
    object_name: str,
    surface_object_name: str | None = None,
    surface_height: float | None = None,
    distance: float = 0.05,
    env_id: int | None = None,
) -> torch.Tensor:
    """True when ``object_name`` is at least ``distance`` m above a surface or surface height."""

    assert (surface_object_name is None) != (
        surface_height is None
    ), "object_lifted requires exactly one of surface_object_name or surface_height"

    object_z = get_root_pos_w(env, object_name)[:, 2]
    if surface_object_name is not None:
        surface_z = get_root_pos_w(env, surface_object_name)[:, 2]
        result = object_z > (surface_z + distance)
    else:
        result = object_z > (surface_height + distance)
    return select(result, env_id)


def object_moved(
    env,
    object_name: str,
    velocity_threshold: float = 0.1,
    env_id: int | None = None,
) -> torch.Tensor:
    """True when object_name's linear speed exceeds velocity_threshold (m/s)."""

    speed = torch.linalg.vector_norm(get_root_lin_vel_w(env, object_name), dim=-1)
    result = speed > velocity_threshold
    return select(result, env_id)
