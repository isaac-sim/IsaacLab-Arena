# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Batched conditions used by the suite's success terms."""

from __future__ import annotations

import math
import torch

import isaaclab.utils.math as math_utils

AT_REST_SPEED_MPS = 0.02
"""Linear speed below which a part counts as at rest."""


def part_upright(env, name: str, max_tilt_deg: float = 15.0) -> torch.Tensor:
    """Return whether the part's z axis is within ``max_tilt_deg`` of vertical (N,)."""
    q_E_part = env.arena_world.get_pose_e(name)[:, 3:7]
    z_axis = torch.zeros(q_E_part.shape[0], 3, device=q_E_part.device)
    z_axis[:, 2] = 1.0
    return math_utils.quat_apply(q_E_part, z_axis)[:, 2] > math.cos(math.radians(max_tilt_deg))


def tool_position_e(env, logic) -> torch.Tensor:
    """Return the position (N, 3) of the tool centre point in the environment frame."""
    return logic.tool_pose_w(env)[:, :3] - env.scene.env_origins
