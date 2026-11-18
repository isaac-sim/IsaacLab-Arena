# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Convex-hull containment predicates."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply_inverse

from isaaclab_arena.utils.mesh_geometry import points_inside_convex_hull

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


def object_hull_centroid_in_closed_hull(
    env: IsaacLabArenaManagerBasedRLEnv,
    object_name: str,
    container_name: str,
) -> torch.Tensor:
    """Check whether an object's hull-vertex center is inside a container's closed hull.

    Tests only the mean of the object's hull vertices; parts of the object may protrude.

    Args:
        env: Environment supplying poses and geometry through ArenaWorld.
        object_name: Rigid-object or fixed-geometry scene-extra name for the subject.
        container_name: Rigid-object or fixed-geometry scene-extra name for the destination.

    Returns:
        One Boolean per parallel environment, including points on the hull boundary.
    """
    return _object_hull_centroid_in_container(env, object_name, container_name, open_top=False)


def object_hull_centroid_in_open_container(
    env: IsaacLabArenaManagerBasedRLEnv,
    object_name: str,
    container_name: str,
) -> torch.Tensor:
    """Check whether an object's hull-vertex center is inside a container's open-top hull.

    Omits faces with container-local normal Z >= 0.7. Retained slopes can limit
    points above the rim. Tests only the mean of the object's hull vertices;
    contact, release, uprightness, and motion are separate conditions.

    Args:
        env: Environment supplying poses and geometry through ArenaWorld.
        object_name: Rigid-object or fixed-geometry scene-extra name for the subject.
        container_name: Rigid-object or fixed-geometry scene-extra name for the destination.

    Returns:
        One Boolean per parallel environment, including points on retained boundaries.
    """
    return _object_hull_centroid_in_container(env, object_name, container_name, open_top=True)


def _object_hull_centroid_in_container(
    env: IsaacLabArenaManagerBasedRLEnv,
    object_name: str,
    container_name: str,
    *,
    open_top: bool,
) -> torch.Tensor:
    """Evaluate a subject's representative point in the container frame C."""
    arena_world = env.arena_world
    object_centroid_W = arena_world.get_hull_vertex_centroid_w(object_name)
    T_W_C = arena_world.get_pose_w(container_name)
    object_centroid_C = quat_apply_inverse(T_W_C[:, 3:], object_centroid_W - T_W_C[:, :3])

    result = torch.empty((env.num_envs,), dtype=torch.bool, device=object_centroid_C.device)
    for local_hull, environment_indices in arena_world.get_convex_hulls_in_local_frame(container_name):
        planes_C = local_hull.open_top_planes if open_top else local_hull.planes
        result[environment_indices] = points_inside_convex_hull(object_centroid_C[environment_indices], planes_C)
    return result
