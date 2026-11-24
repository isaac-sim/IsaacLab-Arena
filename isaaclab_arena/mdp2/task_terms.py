# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""MDP2 terms of Arena tasks: success conditions and metrics."""

from __future__ import annotations

import math
import torch

import isaaclab.utils.math as math_utils
from isaaclab_experimental.mdp2 import State, Term


def _z_axis(quat: torch.Tensor) -> torch.Tensor:
    """Return the z axis of ``(x, y, z, w)`` orientations, without constants from the host."""
    x, y, z, w = quat.unbind(-1)
    return torch.stack((2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)), dim=-1)


def _seated_at_target(
    subject_pose_w: torch.Tensor,
    subject_vel_w: torch.Tensor,
    receiver_pose_w: torch.Tensor,
    *,
    target_offset: tuple[float, float, float],
    max_lateral: float,
    max_depth_below: float,
    max_height_above: float,
    max_tilt_rad: float,
    max_linear_speed: float,
    max_angular_speed: float,
) -> torch.Tensor:
    q_W_R = receiver_pose_w[:, 3:7]
    t_R_T = torch.zeros_like(receiver_pose_w[:, :3])
    for axis, value in enumerate(target_offset):
        t_R_T[:, axis] = value
    t_W_T = receiver_pose_w[:, :3] + math_utils.quat_apply(q_W_R, t_R_T)
    # The subject origin relative to the target, in the target frame, which has the receiver's orientation.
    t_T_S = math_utils.quat_apply_inverse(q_W_R, subject_pose_w[:, :3] - t_W_T)
    lateral = t_T_S[:, :2].norm(dim=-1) <= max_lateral
    depth = (t_T_S[:, 2] >= -max_depth_below) & (t_T_S[:, 2] <= max_height_above)
    upright = (_z_axis(subject_pose_w[:, 3:7]) * _z_axis(q_W_R)).sum(-1) >= math.cos(max_tilt_rad)
    linear_still = subject_vel_w[:, :3].norm(dim=-1) <= max_linear_speed
    angular_still = subject_vel_w[:, 3:6].norm(dim=-1) <= max_angular_speed
    return (lateral & depth & upright & linear_still & angular_still).float()


seated_at_target = Term(torch=_seated_at_target)
"""One where the subject sits at a target point of the receiver: within ``max_lateral`` [m] of it across the
receiver's z axis, between ``max_depth_below`` below and ``max_height_above`` above it along that axis [m], tilted
less than ``max_tilt_rad`` from the receiver's z axis, and slower than ``max_linear_speed`` [m/s] and
``max_angular_speed`` [rad/s]. The target is ``target_offset`` in the receiver frame [m]. Together, these are
the ``lateral_in_proximity``, ``depth_in_range``, ``tilt_axis_aligned``, and ``velocity_below_threshold`` predicates."""


def _held_for(condition: torch.Tensor, count: torch.Tensor, steps: int) -> torch.Tensor:
    count.copy_(torch.where(condition > 0.0, count + 1.0, torch.zeros_like(count)))
    return (count >= steps).float()


held_for = Term(torch=_held_for, state={"count": State()})
"""One once ``condition`` held for ``steps`` consecutive control steps, like ``TrueForConsecutiveStepsCfg``. Its state
``count`` holds the current run of steps."""


def _moved(root_com_vel_w: torch.Tensor, moved: torch.Tensor, speed: float) -> torch.Tensor:
    moved.logical_or_(root_com_vel_w[:, :3].norm(dim=1) > speed)
    return moved.clone()


moved = Term(torch=_moved, state={"moved": State()})
"""One once the object moved faster than ``speed`` [m/s] in the episode, like the object-moved-rate metric."""
