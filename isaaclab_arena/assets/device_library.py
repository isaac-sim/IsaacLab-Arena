# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

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

from abc import ABC, abstractmethod

from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg
from isaaclab.devices.retargeter_base import RetargeterCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg

from isaaclab_arena.assets.register import register_device


class TeleopDeviceBase(ABC):

    name: str | None = None

    def __init__(self, sim_device: str | None = None):
        self.sim_device = sim_device

    @abstractmethod
    def get_device_cfg(self, retargeters: list[RetargeterCfg] | None = None, embodiment: object | None = None):
        raise NotImplementedError


@register_device
class OpenXRCfg(TeleopDeviceBase):
    name = "openxr"

    def __init__(self, sim_device: str | None = None):
        super().__init__(sim_device=sim_device)

    def get_device_cfg(
        self, retargeters: list[RetargeterCfg] | None = None, embodiment: object | None = None
    ) -> OpenXRDeviceCfg:
        return OpenXRDeviceCfg(
            retargeters=retargeters,
            sim_device=self.sim_device,
            xr_cfg=embodiment.get_xr_cfg(),
        )


@register_device
class KeyboardCfg(TeleopDeviceBase):
    name = "keyboard"

    def __init__(self, sim_device: str | None = None, pos_sensitivity: float = 0.05, rot_sensitivity: float = 0.05):
        super().__init__(sim_device=sim_device)
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

    def get_device_cfg(
        self, retargeters: list[RetargeterCfg] | None = None, embodiment: object | None = None
    ) -> Se3KeyboardCfg:
        return Se3KeyboardCfg(
            retargeters=retargeters,
            sim_device=self.sim_device,
            pos_sensitivity=self.pos_sensitivity,
            rot_sensitivity=self.rot_sensitivity,
        )


@register_device
class SpaceMouseCfg(TeleopDeviceBase):
    name = "spacemouse"

    def __init__(self, sim_device: str | None = None, pos_sensitivity: float = 0.05, rot_sensitivity: float = 0.05):
        super().__init__(sim_device=sim_device)
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

    def get_device_cfg(
        self, retargeters: list[RetargeterCfg] | None = None, embodiment: object | None = None
    ) -> Se3SpaceMouseCfg:
        return Se3SpaceMouseCfg(
            retargeters=retargeters,
            sim_device=self.sim_device,
            pos_sensitivity=self.pos_sensitivity,
            rot_sensitivity=self.rot_sensitivity,
        )
