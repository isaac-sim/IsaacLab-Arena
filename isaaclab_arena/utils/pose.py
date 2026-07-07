# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch
from dataclasses import dataclass


@dataclass
class Pose:
    """Transform taking frame A to frame B.

    T_A_B = (t_B_A, q_B_A)

    p_B = p_A + t_B_A
    q_B = q_A * q_B_A
    """

    position_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation vector from frame A to frame B."""

    rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Quaternion from frame A to frame B. Order is (x, y, z, w)."""

    def __post_init__(self):
        assert isinstance(self.position_xyz, tuple)
        assert isinstance(self.rotation_xyzw, tuple)
        assert len(self.position_xyz) == 3
        assert len(self.rotation_xyzw) == 4

    @staticmethod
    def identity() -> "Pose":
        return Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert the pose to a tensor.

        The returned tensor has shape (1, 7), and is of the order (x, y, z, qx, qy, qz, qw).

        Args:
            device: The device to convert the tensor to.

        Returns:
            The pose as a tensor of shape (1, 7).
        """
        position_tensor = torch.tensor(self.position_xyz, device=device)
        rotation_tensor = torch.tensor(self.rotation_xyzw, device=device)
        return torch.cat([position_tensor, rotation_tensor])

    def multiply(self, other: "Pose") -> "Pose":
        return compose_poses(self, other)


def wrap_angle_to_pi(angle_rad: float) -> float:
    """Wrap an angle in radians to [-pi, pi)."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


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


def compose_poses(T_C_B: Pose, T_B_A: Pose) -> Pose:
    """Compose two poses. T_C_A = T_C_B * T_B_A

    Args:
        T_B_A: The pose taking points from A to B.
        T_C_B: The pose taking points from B to C.

    Returns:
        The pose taking points from A to C.
    """
    from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

    R_B_A = matrix_from_quat(torch.tensor(T_B_A.rotation_xyzw))
    R_C_B = matrix_from_quat(torch.tensor(T_C_B.rotation_xyzw))
    # Compose the rotations
    R_C_A = R_C_B @ R_B_A
    q_C_A = quat_from_matrix(R_C_A)
    # Compose the translations
    t_C_A = R_C_B @ torch.tensor(T_B_A.position_xyz) + torch.tensor(T_C_B.position_xyz)
    return Pose(position_xyz=tuple(t_C_A.tolist()), rotation_xyzw=tuple(q_C_A.tolist()))


@dataclass
class PosePerEnv:
    """Per-environment poses (one Pose per env, used for batched placement)."""

    poses: list[Pose]
    """One Pose per environment."""


@dataclass
class PoseRange:
    """Range of poses.

    Args:
        position_xyz_min: The minimum position in x, y, z.
        position_xyz_max: The maximum position in x, y, z.
        rpy_min: The minimum rotation in roll, pitch, yaw (in radians).
        rpy_max: The maximum rotation in roll, pitch, yaw (in radians).
    """

    position_xyz_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_xyz_max: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy_max: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, tuple[float, float]]:
        return {
            "x": (self.position_xyz_min[0], self.position_xyz_max[0]),
            "y": (self.position_xyz_min[1], self.position_xyz_max[1]),
            "z": (self.position_xyz_min[2], self.position_xyz_max[2]),
            "roll": (self.rpy_min[0], self.rpy_max[0]),
            "pitch": (self.rpy_min[1], self.rpy_max[1]),
            "yaw": (self.rpy_min[2], self.rpy_max[2]),
        }

    def get_midpoint(self) -> Pose:
        from isaaclab.utils.math import quat_from_euler_xyz

        roll = torch.tensor((self.rpy_min[0] + self.rpy_max[0]) / 2)
        pitch = torch.tensor((self.rpy_min[1] + self.rpy_max[1]) / 2)
        yaw = torch.tensor((self.rpy_min[2] + self.rpy_max[2]) / 2)
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        position_xyz = torch.tensor([
            (self.position_xyz_min[0] + self.position_xyz_max[0]) / 2,
            (self.position_xyz_min[1] + self.position_xyz_max[1]) / 2,
            (self.position_xyz_min[2] + self.position_xyz_max[2]) / 2,
        ])
        return Pose(
            position_xyz=tuple(position_xyz.tolist()),
            rotation_xyzw=tuple(quat.tolist()),
        )
