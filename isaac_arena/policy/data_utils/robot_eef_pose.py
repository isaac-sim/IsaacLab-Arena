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

import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class EefPose:
    eef_pos: torch.Tensor
    """Eef position in meters"""

    eef_quat: torch.Tensor # wxyz
    """Eef orientation in quaternions w, x, y, z"""

    eef_pose: torch.Tensor
    """Eef pose in 6D format"""

    device: torch.device
    """Device to store the tensor on"""

    @staticmethod
    def zero(device: torch.device):
        return EefPose(
            eef_pos=torch.zeros((3), device=device),
            eef_quat=torch.zeros((4), device=device),
            eef_pose=torch.zeros((7), device=device),
            device=device,
        )

    def to_array(self) -> torch.Tensor:
        return self.eef_pose.cpu().numpy()

    @staticmethod
    def from_array(pos_array: np.ndarray, quat_array: np.ndarray, device: torch.device) -> "EefPose":
        return EefPose(
            eef_pos=torch.from_numpy(pos_array).to(device),
            eef_quat=torch.from_numpy(quat_array).to(device),
            eef_pose=torch.cat([torch.from_numpy(pos_array), torch.from_numpy(quat_array)], dim=1).to(device),
            device=device,
        )

    def set_eef_pose(self, eef_pos: torch.Tensor, eef_quat: torch.Tensor):
        self.eef_pos = eef_pos.to(self.device)
        self.eef_quat = eef_quat.to(self.device)
        self.eef_pose = torch.cat([self.eef_pos, self.eef_quat], dim=1).to(self.device)

    def get_eef_pose(self, device: torch.device = None) -> torch.Tensor:
        if device is None:
            return self.eef_pose
        else:
            return self.eef_pose.to(device)
