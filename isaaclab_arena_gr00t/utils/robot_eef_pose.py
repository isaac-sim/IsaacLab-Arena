# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Container for the end-effector pose of a robot."""

import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class EefPose:
    """End-effector position and orientation (x, y, z, w quaternion) stored as batched tensors on a device."""

    eef_pos: torch.Tensor
    """Eef position in meters"""

    eef_quat: torch.Tensor
    """Eef orientation in quaternions x, y, z, w"""

    device: torch.device
    """Device to store the tensor on"""

    @staticmethod
    def identity(device: torch.device, batch_size: int = 1):
        """Create a batch of poses at the origin with identity orientation.

        Args:
            device: Device to store the tensors on.
            batch_size: Number of poses in the batch.
        """
        return EefPose(
            eef_pos=torch.zeros((batch_size, 3), device=device),
            eef_quat=torch.tensor([0, 0, 0, 1], device=device).repeat(batch_size, 1),
            device=device,
        )

    def to_array(self) -> np.ndarray:
        """Return the end-effector pose as a numpy array."""
        return self.eef_pose.cpu().numpy()

    @staticmethod
    def from_array(pos_array: np.ndarray, quat_array: np.ndarray, device: torch.device) -> "EefPose":
        """Create a pose from position and quaternion arrays.

        Args:
            pos_array: Positions of shape (batch_size, 3), in meters.
            quat_array: Quaternions of shape (batch_size, 4) in x, y, z, w order.
            device: Device to store the tensors on.
        """
        return EefPose(
            eef_pos=torch.from_numpy(pos_array).to(device),
            eef_quat=torch.from_numpy(quat_array).to(device),
            device=device,
        )

    def set_eef_pose(self, eef_pos: torch.Tensor, eef_quat: torch.Tensor):
        """Set the position and orientation, moving them to the device of this pose.

        Args:
            eef_pos: Positions of shape (batch_size, 3), in meters.
            eef_quat: Quaternions of shape (batch_size, 4) in x, y, z, w order.
        """
        self.eef_pos = eef_pos.to(self.device)
        self.eef_quat = eef_quat.to(self.device)

    def get_eef_pose(self, device: torch.device = None) -> torch.Tensor:
        # NOTE(xinjieyao, 2025-09-26): compose it on the fly to avoid invariance issues where eef_pos could be modified after eef_pose has been set
        """Return the position and quaternion concatenated along the last dimension, shape (batch_size, 7).

        Args:
            device: Device to move the result to. If None, the result stays on the device of this pose.
        """
        eef_pose = torch.cat([self.eef_pos, self.eef_quat], dim=1)
        if device is None:
            return eef_pose
        else:
            return eef_pose.to(device)
