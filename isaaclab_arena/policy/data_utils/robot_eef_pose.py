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

    eef_quat: torch.Tensor
    """Eef orientation in quaternions w, x, y, z"""

    device: torch.device
    """Device to store the tensor on"""

    @staticmethod
    def identity(device: torch.device, batch_size: int = 1):
        return EefPose(
            eef_pos=torch.zeros((batch_size, 3), device=device),
            eef_quat=torch.tensor([1, 0, 0, 0], device=device).repeat(batch_size, 1),
            device=device,
        )

    def to_array(self) -> np.ndarray:
        return self.eef_pose.cpu().numpy()

    @staticmethod
    def from_array(pos_array: np.ndarray, quat_array: np.ndarray, device: torch.device) -> "EefPose":
        return EefPose(
            eef_pos=torch.from_numpy(pos_array).to(device),
            eef_quat=torch.from_numpy(quat_array).to(device),
            device=device,
        )

    def set_eef_pose(self, eef_pos: torch.Tensor, eef_quat: torch.Tensor):
        self.eef_pos = eef_pos.to(self.device)
        self.eef_quat = eef_quat.to(self.device)

    def get_eef_pose(self, device: torch.device = None) -> torch.Tensor:
        # NOTE(xinjieyao, 2025-09-26): compose it on the fly to avoid invariance issues where eef_pos could be modified after eef_pose has been set
        eef_pose = torch.cat([self.eef_pos, self.eef_quat], dim=1)
        if device is None:
            return eef_pose
        else:
            return eef_pose.to(device)
