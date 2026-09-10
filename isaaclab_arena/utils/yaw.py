# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch


def wrap_angle_to_pi(angle_rad: float) -> float:
    """Wrap an angle in radians to [-pi, pi)."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


MINIMUM_FACING_DIRECTION_XY_M = 1e-6
"""XY distances at or below this are too short to define a facing direction."""


def yaw_toward_positions(
    subject_positions: torch.Tensor,
    target_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Z-yaws from subject positions toward target positions.

    N is the number of position pairs. Directions at most MINIMUM_FACING_DIRECTION_XY_M
    long in XY are undefined.

    Args:
        subject_positions: Subject world positions with shape (N, 3).
        target_positions: Target world positions with shape (N, 3).

    Returns:
        A tuple ``(yaws, is_defined)`` containing the world Z-yaws and
        valid-direction mask as two tensors of shape (N,).
    """
    assert subject_positions.shape == target_positions.shape
    assert subject_positions.ndim == 2 and subject_positions.shape[1] == 3
    delta_xy = target_positions[:, :2] - subject_positions[:, :2]
    is_defined = torch.linalg.vector_norm(delta_xy, dim=1) > MINIMUM_FACING_DIRECTION_XY_M
    return torch.atan2(delta_xy[:, 1], delta_xy[:, 0]), is_defined


def yaw_from_quat_xyzw(quat_xyzw: tuple[float, float, float, float]) -> float:
    """Extract Z-axis yaw (radians) from an (x, y, z, w) quaternion.

    Returns 0.0 if the quaternion has non-trivial roll or pitch (|qx| or |qy| > 1e-6).
    """
    qx, qy, qz, qw = quat_xyzw
    if abs(qx) > 1e-6 or abs(qy) > 1e-6:
        return 0.0
    return 2.0 * math.atan2(qz, qw)


def rotate_quat_by_yaw(
    base_xyzw: tuple[float, float, float, float], yaw_rad: float
) -> tuple[float, float, float, float]:
    """Rotate base_xyzw (xyzw) by an extra yaw about Z. Returns base unchanged when yaw is 0."""
    yaw_rad = wrap_angle_to_pi(yaw_rad)  # keep half-angle small for precision; canonicalize 2pi -> 0
    if yaw_rad == 0.0:
        return base_xyzw
    bx, by, bz, bw = base_xyzw
    sz = math.sin(yaw_rad / 2.0)
    cz = math.cos(yaw_rad / 2.0)
    # Hamilton product base ⊗ (0, 0, sz, cz). Both rotations are about Z, so they commute.
    return (bx * cz + by * sz, -bx * sz + by * cz, bz * cz + bw * sz, -bz * sz + bw * cz)


def rotate_points_by_yaw(points: torch.Tensor, yaw: float) -> torch.Tensor:
    """Rotate (N, 3) points about the Z-axis by a scalar yaw (radians). Z is unchanged."""
    if yaw == 0.0:
        return points
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    x = points[:, 0] * cos_y - points[:, 1] * sin_y
    y = points[:, 0] * sin_y + points[:, 1] * cos_y
    return torch.stack([x, y, points[:, 2]], dim=-1)


def rotate_points_by_yaw_batch(points: torch.Tensor, yaws: torch.Tensor) -> torch.Tensor:
    """Rotate (N, 3) points about Z-axis by per-element yaw (N,) radians. Z is unchanged."""
    cos_y = torch.cos(yaws)
    sin_y = torch.sin(yaws)
    x = points[:, 0] * cos_y - points[:, 1] * sin_y
    y = points[:, 0] * sin_y + points[:, 1] * cos_y
    return torch.stack([x, y, points[:, 2]], dim=-1)


def centers_in_target_frame(
    centers_local: torch.Tensor,
    src_yaw: float,
    tgt_yaw: float,
    offset: torch.Tensor,
) -> torch.Tensor:
    """Transform source sphere centers into the target's local frame (Z-yaw only).

    Computes R(src_yaw - tgt_yaw) * centers_local + R(-tgt_yaw) * offset.

    Args:
        centers_local: (N, 3) sphere centers in the source's local frame.
        src_yaw: Source object yaw (radians).
        tgt_yaw: Target object yaw (radians).
        offset: (3,) vector from target position to source position (world frame).
    """
    net_yaw = src_yaw - tgt_yaw
    if net_yaw == 0.0 and tgt_yaw == 0.0:
        return centers_local + offset

    rotated_centers = rotate_points_by_yaw(centers_local, net_yaw)
    rotated_offset = rotate_points_by_yaw(offset.unsqueeze(0), -tgt_yaw).squeeze(0)
    return rotated_centers + rotated_offset
