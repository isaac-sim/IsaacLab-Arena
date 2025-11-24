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

from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg

from isaaclab_arena.assets.register import register_device
from isaaclab_arena.teleop_devices.teleop_device_base import TeleopDeviceBase


@register_device
class SpacemouseTeleopDevice(TeleopDeviceBase):
    """
    Teleop device for spacemouse.
    """

    name = "spacemouse"

    def __init__(self, sim_device: str | None = None, pos_sensitivity: float = 0.05, rot_sensitivity: float = 0.05):
        super().__init__(sim_device=sim_device)
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

    def get_teleop_device_cfg(self, embodiment: object | None = None):
        return DevicesCfg(
            devices={
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=self.pos_sensitivity,
                    rot_sensitivity=self.rot_sensitivity,
                    sim_device=self.sim_device,
                ),
            }
        )
