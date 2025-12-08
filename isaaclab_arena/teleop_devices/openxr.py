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
from isaaclab.devices.openxr import OpenXRDeviceCfg

from isaaclab_arena.assets.register import register_device
from isaaclab_arena.teleop_devices.teleop_device_base import TeleopDeviceBase


@register_device
class OpenXRTeleopDevice(TeleopDeviceBase):
    """
    Teleop device wrapping the OpenXRDevice.
    """

    name = "openxr"

    def __init__(self, sim_device: str | None = None):
        super().__init__(sim_device=sim_device)

    def get_teleop_device_cfg(self, embodiment: object | None = None):
        return DevicesCfg(
            devices={
                "openxr": OpenXRDeviceCfg(
                    retargeters=embodiment.get_retargeters(sim_device=self.sim_device)["openxr"],
                    sim_device=self.sim_device,
                    xr_cfg=embodiment.get_xr_cfg(),
                ),
            }
        )
