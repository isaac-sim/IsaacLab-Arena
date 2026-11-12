# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Container for absolute robot joint positions."""

import numpy as np
import torch
from dataclasses import dataclass

from isaaclab_arena_gr00t.utils.io_utils import to_numpy


@dataclass
class JointsAbsPosition:
    """Absolute joint positions of a robot together with the configuration that gives the order of the joints."""

    joints_pos: torch.Tensor
    """Joint positions in radians"""

    joints_order_config: dict[str, int]
    """Joints order configuration"""

    @staticmethod
    def zero(joint_order_config: dict[str, int], device: torch.device):
        """Create joint positions that are all zero.

        Args:
            joint_order_config: Mapping from joint name to its index in the joint position vector.
            device: Device to store the tensor on.
        """
        return JointsAbsPosition(
            joints_pos=torch.zeros((len(joint_order_config)), device=device), joints_order_config=joint_order_config
        )

    def to_array(self) -> np.ndarray:
        """Return the joint positions as a numpy array."""
        return to_numpy(self.joints_pos)

    @staticmethod
    def from_array(array: np.ndarray, joint_order_config: dict[str, int], device: torch.device) -> "JointsAbsPosition":
        """Create joint positions from a numpy array.

        Args:
            array: Joint positions, in radians.
            joint_order_config: Mapping from joint name to its index in the joint position vector.
            device: Device to store the tensor on.
        """
        return JointsAbsPosition(joints_pos=torch.from_numpy(array).to(device), joints_order_config=joint_order_config)

    def set_joints_pos(self, joints_pos: torch.Tensor):
        """Set the joint positions, moving them to the device of this object.

        Args:
            joints_pos: Joint positions, in radians.
        """
        self.joints_pos = joints_pos.to(self.device)

    def get_joints_pos(self, device: torch.device = None) -> torch.Tensor:
        """Return the joint positions.

        Args:
            device: Device to move the result to. If None, the result stays on its current device.
        """
        if device is None:
            return self.joints_pos
        else:
            return self.joints_pos.to(device)
