# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Pose-state predicates: orientation, stillness."""

from __future__ import annotations

import math
import torch

from isaaclab_arena.tasks.predicates.decorators import atomic
from isaaclab_arena.tasks.predicates.geometry import (
    get_root_ang_vel_w,
    get_root_lin_vel_w,
    get_root_quat_w,
    logical_and,
    select,
)


@atomic
def object_upright(
    env,
    object_name: str,
    tolerance_rad: float = 0.2,
    up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    env_id: int | None = None,
) -> torch.Tensor:
    """True when ``object_name``'s body-frame up axis is within ``tolerance_rad`` of world up.

    Rotates the configured ``up_axis`` (default ``+z``) by the object's quaternion
    and compares the rotated z-component against ``cos(tolerance_rad)``.
    """
    quat_wxyz = get_root_quat_w(env, object_name)
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]

    ax, ay, az = up_axis
    rx = (1 - 2 * (y * y + z * z)) * ax + 2 * (x * y - w * z) * ay + 2 * (x * z + w * y) * az
    ry = 2 * (x * y + w * z) * ax + (1 - 2 * (x * x + z * z)) * ay + 2 * (y * z - w * x) * az
    rz = 2 * (x * z - w * y) * ax + 2 * (y * z + w * x) * ay + (1 - 2 * (x * x + y * y)) * az

    norm = torch.sqrt(rx * rx + ry * ry + rz * rz).clamp_min(1e-8)
    cos_theta = rz / norm
    result = cos_theta >= math.cos(tolerance_rad)
    return select(result, env_id)


@atomic
def object_stationary(
    env,
    object_name: str,
    linear_threshold: float = 0.05,
    angular_threshold: float = 0.5,
    check_angular: bool = True,
    env_id: int | None = None,
) -> torch.Tensor:
    """True when an object's linear (and optionally angular) speeds are below thresholds."""
    lin_vel_norm = torch.linalg.norm(get_root_lin_vel_w(env, object_name), dim=1)
    result = lin_vel_norm < linear_threshold
    if check_angular:
        ang_vel_norm = torch.linalg.norm(get_root_ang_vel_w(env, object_name), dim=1)
        result = logical_and(result, ang_vel_norm < angular_threshold)
    return select(result, env_id)
