# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_arena.utils.pose import PoseRange


def pose_range_from_quat(
    position_xyz_min: tuple[float, float, float],
    position_xyz_max: tuple[float, float, float],
    rotation_xyzw: tuple[float, float, float, float],
    yaw_jitter: float | tuple[float, float] = 0.0,
    roll_jitter: float | tuple[float, float] = 0.0,
    pitch_jitter: float | tuple[float, float] = 0.0,
) -> PoseRange:
    """Factory-local equivalent of the old ``PoseRange.from_quat`` helper."""
    q = torch.tensor(rotation_xyzw).unsqueeze(0)
    roll, pitch, yaw = euler_xyz_from_quat(q)
    r, p, y = roll.item(), pitch.item(), yaw.item()

    def _offset(center: float, jitter: float | tuple[float, float]) -> tuple[float, float]:
        if isinstance(jitter, (list, tuple)):
            return (center + jitter[0], center + jitter[1])
        return (center - jitter, center + jitter)

    roll_min, roll_max = _offset(r, roll_jitter)
    pitch_min, pitch_max = _offset(p, pitch_jitter)
    yaw_min, yaw_max = _offset(y, yaw_jitter)

    return PoseRange(
        position_xyz_min=position_xyz_min,
        position_xyz_max=position_xyz_max,
        rpy_min=(roll_min, pitch_min, yaw_min),
        rpy_max=(roll_max, pitch_max, yaw_max),
    )
