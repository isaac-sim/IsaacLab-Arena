# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from dataclasses import dataclass

from isaaclab.utils.math import matrix_from_quat, quat_from_matrix


@dataclass
class Pose:
    """Transform taking frame A to frame B.

    T_A_B = (t_B_A, q_B_A)

    p_B = p_A + t_B_A
    q_B = q_A * q_B_A
    """

    position_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation vector from frame A to frame B."""

    rotation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Quaternion from frame A to frame B. Order is (w, x, y, z)."""

    def __post_init__(self):
        assert isinstance(self.position_xyz, tuple)
        assert isinstance(self.rotation_wxyz, tuple)
        assert len(self.position_xyz) == 3
        assert len(self.rotation_wxyz) == 4

    @staticmethod
    def identity() -> "Pose":
        return Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert the pose to a tensor.

        The returned tensor has shape (1, 7), and is of the order (x, y, z, qw, qx, qy, qz).

        Args:
            device: The device to convert the tensor to.

        Returns:
            The pose as a tensor of shape (1, 7).
        """
        position_tensor = torch.tensor(self.position_xyz, device=device)
        rotation_tensor = torch.tensor(self.rotation_wxyz, device=device)
        return torch.cat([position_tensor, rotation_tensor])

    def multiply(self, other: "Pose") -> "Pose":
        return compose_poses(self, other)


def compose_poses(T_C_B: Pose, T_B_A: Pose) -> Pose:
    """Compose two poses. T_C_A = T_C_B * T_B_A

    Args:
        T_B_A: The pose taking points from A to B.
        T_C_B: The pose taking points from B to C.

    Returns:
        The pose taking points from A to C.
    """
    R_B_A = matrix_from_quat(torch.tensor(T_B_A.rotation_wxyz))
    R_C_B = matrix_from_quat(torch.tensor(T_C_B.rotation_wxyz))
    # Compose the rotations
    R_C_A = R_C_B @ R_B_A
    q_C_A = quat_from_matrix(R_C_A)
    # Compose the translations
    t_C_A = R_C_B @ torch.tensor(T_B_A.position_xyz) + torch.tensor(T_C_B.position_xyz)
    return Pose(position_xyz=tuple(t_C_A.tolist()), rotation_wxyz=tuple(q_C_A.tolist()))
