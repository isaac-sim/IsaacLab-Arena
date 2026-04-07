# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from dataclasses import dataclass


@dataclass
class Velocity:
    """Linear and angular velocity of a rigid body in the world frame."""

    linear_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Linear velocity (vx, vy, vz) in the world frame."""

    angular_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Angular velocity (wx, wy, wz) in the world frame."""

    def __post_init__(self):
        assert isinstance(self.linear_xyz, tuple)
        assert isinstance(self.angular_xyz, tuple)
        assert len(self.linear_xyz) == 3
        assert len(self.angular_xyz) == 3

    @staticmethod
    def zero() -> "Velocity":
        return Velocity(linear_xyz=(0.0, 0.0, 0.0), angular_xyz=(0.0, 0.0, 0.0))

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert the velocity to a tensor.

        The returned tensor has shape (6,), ordered as (vx, vy, vz, wx, wy, wz).

        Args:
            device: The device to place the tensor on.

        Returns:
            The velocity as a tensor of shape (6,).
        """
        linear_tensor = torch.tensor(self.linear_xyz, device=device)
        angular_tensor = torch.tensor(self.angular_xyz, device=device)
        return torch.cat([linear_tensor, angular_tensor])
