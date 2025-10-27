# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

    rotation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Quaternion from frame A to frame B. Order is (w, x, y, z)."""

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
