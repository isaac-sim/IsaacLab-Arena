# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Batched geometric conditions used by the suite's success terms."""

from __future__ import annotations

import math
import torch

import isaaclab.utils.math as math_utils


def item_pose_e(env, name: str) -> torch.Tensor:
    """Return the environment-frame pose (N, 7) of a scene part."""
    return env.arena_world.get_pose_e(name)


def item_upright(env, name: str, max_tilt_deg: float = 15.0) -> torch.Tensor:
    """Return whether the part's z axis is within ``max_tilt_deg`` of vertical (N,)."""
    quat = item_pose_e(env, name)[:, 3:7]
    z_axis = torch.zeros(quat.shape[0], 3, device=quat.device)
    z_axis[:, 2] = 1.0
    return math_utils.quat_apply(quat, z_axis)[:, 2] > math.cos(math.radians(max_tilt_deg))


def item_at_rest(env, name: str, max_speed_mps: float = 0.02) -> torch.Tensor:
    """Return whether the part's linear speed is below ``max_speed_mps`` (N,)."""
    return env.arena_world.get_root_linear_velocity_w(name).norm(dim=-1) < max_speed_mps


def tool_position_e(env, logic) -> torch.Tensor:
    """Return the tool position (N, 3) in the environment frame."""
    return logic.tool_pose_w(env)[:, :3] - env.scene.env_origins


def gripper_open(env, logic) -> torch.Tensor:
    """Return whether the gripper is commanded open (N,)."""
    return ~logic.gripper_closed(env)
