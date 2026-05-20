# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Spatial predicates: position/distance/height-based scene queries.

All predicates share the signature ``(env, ..., *, env_id=None) -> torch.Tensor``.
- ``env_id=None`` returns a ``(num_envs,)`` bool tensor (vectorized).
- ``env_id=<int>`` returns a 0-D bool tensor for that env (useful for debugging).
Positions are read in world frame and converted to env-local using
``env.scene.env_origins`` so comparisons are consistent across environments.
"""

from __future__ import annotations

import torch

from isaaclab_arena.tasks.predicates.decorators import atomic, composite
from isaaclab_arena.tasks.predicates.geometry import (
    get_env,
    get_root_pos_w,
    logical_and,
    select,
)


def _env_local_pos(env, object_name: str) -> torch.Tensor:
    e = get_env(env)
    return get_root_pos_w(e, object_name) - e.scene.env_origins


@atomic
def object_above(
    env,
    object_name: str,
    reference_object_name: str | None = None,
    height: float | None = None,
    z_margin: float = 0.0,
    env_id: int | None = None,
) -> torch.Tensor:
    """True when ``object_name`` is above either ``reference_object_name`` or absolute ``height``.

    Exactly one of ``reference_object_name`` or ``height`` must be provided. The
    comparison is ``object_z > reference_z + z_margin``. Heights are in world
    frame; both objects are assumed to share the same z origin (Isaac Lab's
    env_origins are typically z=0 so this holds in practice).
    """
    assert (reference_object_name is None) != (height is None), (
        "object_above requires exactly one of reference_object_name or height"
    )
    obj_z = get_root_pos_w(env, object_name)[:, 2]
    if reference_object_name is not None:
        ref_z = get_root_pos_w(env, reference_object_name)[:, 2]
        result = obj_z > (ref_z + z_margin)
    else:
        result = obj_z > (height + z_margin)
    return select(result, env_id)


@atomic
def object_at(
    env,
    object_name: str,
    position: tuple[float, float, float],
    tolerance: float = 0.05,
    env_id: int | None = None,
) -> torch.Tensor:
    """True when ``object_name`` is within ``tolerance`` meters of ``position`` (env-local)."""
    e = get_env(env)
    pos = _env_local_pos(e, object_name)
    target = torch.tensor(position, device=pos.device, dtype=pos.dtype).unsqueeze(0)
    distance = torch.linalg.norm(pos - target, dim=1)
    return select(distance < tolerance, env_id)


@atomic
def object_near(
    env,
    object_name: str,
    target_object_name: str,
    max_distance: float,
    env_id: int | None = None,
) -> torch.Tensor:
    """True when the two objects' centroids are within ``max_distance`` meters."""
    e = get_env(env)
    a = _env_local_pos(e, object_name)
    b = _env_local_pos(e, target_object_name)
    distance = torch.linalg.norm(a - b, dim=1)
    return select(distance < max_distance, env_id)


@composite
def object_picked_up(
    env,
    object_name: str,
    surface_object_name: str | None = None,
    surface_height: float | None = None,
    distance: float = 0.05,
    env_id: int | None = None,
) -> torch.Tensor:
    """True when ``object_name`` is at least ``distance`` above a surface.

    A position-only "lifted" check — does not require a grasp signal. Combine
    with ``object_grabbed`` via ``and_`` / a dict-of-conditions if you want both.
    """
    return object_above(
        env,
        object_name=object_name,
        reference_object_name=surface_object_name,
        height=surface_height,
        z_margin=distance,
        env_id=env_id,
    )


@atomic
def objects_in_proximity_predicate(
    env,
    object_name: str,
    target_object_name: str,
    max_x: float,
    max_y: float,
    max_z: float,
    env_id: int | None = None,
) -> torch.Tensor:
    """Axis-wise proximity check between two objects (envelope-style placement)."""
    e = get_env(env)
    a = _env_local_pos(e, object_name)
    b = _env_local_pos(e, target_object_name)
    diff = torch.abs(a - b)
    result = diff[:, 0] < max_x
    result = logical_and(result, diff[:, 1] < max_y)
    result = logical_and(result, diff[:, 2] < max_z)
    return select(result, env_id)
