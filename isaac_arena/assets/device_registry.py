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

from typing import TYPE_CHECKING

from isaac_arena.assets.registry import Registry

if TYPE_CHECKING:
    from isaac_arena.teleop_devices.teleop_device_base import TeleopDeviceBase


class DeviceRegistry(Registry):

    def __init__(self):
        super().__init__()


def get_environment_configuration_from_device_registry(
    device_registry: DeviceRegistry,
    device_name: str | None = None,
) -> dict[str, type["TeleopDeviceBase"]]:
    from isaac_arena.assets.device_registry import DeviceRegistry

    device_registry = DeviceRegistry()
    if device_name:
        print(f"Getting device {device_name} from device registry")
        assert device_registry.is_registered(device_name), f"Device {device_name} not registered"
        device = device_registry.get_item_by_name(device_name)()
    else:
        device = None

    return {
        "device": device,
    }


# Register all the devices. Down here at the bottom of the file because
# the devices use the DeviceRegistry class in order to register themselves,
# so it needs to be fully defined to avoid a circular import.
from isaac_arena.teleop_devices.handtracking import *  # noqa: F403, F401
