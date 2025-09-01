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

from abc import ABC
from typing import Any


class TeleopDeviceBase(ABC):

    name: str | None = None

    def __init__(self):
        self._teleop_device_cfg: Any | None = None

    def build_cfg(self, *, sim_device: str | None = None, actions: object | None = None, xr_cfg: object | None = None):
        raise NotImplementedError

    def get_teleop_device_cfg(
        self, *, sim_device: str | None = None, actions: object | None = None, xr_cfg: object | None = None
    ):
        if self._teleop_device_cfg is None:
            self._teleop_device_cfg = self.build_cfg(sim_device=sim_device, actions=actions, xr_cfg=xr_cfg)
        return self._teleop_device_cfg
