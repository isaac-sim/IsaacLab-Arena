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

import random
from typing import Any

from isaac_arena.assets.asset import Asset
from isaac_arena.teleop_devices.teleop_device_base import TeleopDeviceBase
from isaac_arena.utils.singleton import SingletonMeta


# Have to define all classes here in order to avoid circular import.
class Registry(metaclass=SingletonMeta):

    def __init__(self):
        self.components = {}

    def register(self, component: Any):
        """Register an asset with a name.

        Args:
            name (str): The name of the asset.
            asset (Asset): The asset to register.
        """
        assert component.name not in self.components, f"component {component.name} already registered"
        assert component.name is not None, "component name is not set"
        if component.tagged:
            assert component.tags is not None, "Asset tags are not set"
        self.components[component.name] = component

    def is_registered(self, name: str) -> bool:
        """Check if an component is registered.

        Args:
            name (str): The name of the component.
        """
        return name in self.components

    def get_component_by_name(self, name: str) -> Any:
        """Get an component by name.

        Args:
            name (str): The name of the component.

        Returns:
            Asset: The component.
        """
        return self.components[name]


class AssetRegistry(Registry):

    def __init__(self):
        super().__init__()

    def get_asset_by_name(self, name: str) -> type["Asset"]:
        """Gets an asset by name.

        Args:
            name (str): The name of the asset.
        """
        return self.get_component_by_name(name)

    def get_assets_by_tag(self, tag: str) -> list[type["Asset"]]:
        """Gets a list of assets by tag.

        Args:
            tag (str): The tag of the assets.

        Returns:
            list[Asset]: The list of assets.
        """
        return [asset for asset in self.components.values() if tag in asset.tags]

    def get_random_asset_by_tag(self, tag: str) -> type["Asset"]:
        """Gets a random asset which has the given tag.

        Args:
            tag (str): The tag of the assets.

        Returns:
            Asset: The random asset.
        """
        assets = self.get_assets_by_tag(tag)
        if len(assets) == 0:
            raise ValueError(f"No assets found with tag {tag}")
        return random.choice(assets)


class DeviceRegistry(Registry):

    def __init__(self):
        super().__init__()

    def get_device_by_name(self, name: str) -> type["TeleopDeviceBase"]:
        """Gets a device by name.

        Args:
            name (str): The name of the device.
        """
        return self.get_component_by_name(name)


def get_environment_configuration_from_asset_registry(
    background_name: str | None = None,
    object_name: str | None = None,
    embodiment_name: str | None = None,
) -> dict[str, type["Asset"]]:

    asset_registry = AssetRegistry()
    if background_name:
        background = asset_registry.get_asset_by_name(background_name)()
    else:
        background = asset_registry.get_random_asset_by_tag("background")()
    if object_name:
        pick_up_object = asset_registry.get_asset_by_name(object_name)()
    else:
        pick_up_object = asset_registry.get_random_asset_by_tag("object")()
    if embodiment_name:
        embodiment = asset_registry.get_asset_by_name(embodiment_name)()
    else:
        embodiment = asset_registry.get_random_asset_by_tag("embodiment")()

    environment_configuration = {
        "background": background,
        "object": pick_up_object,
        "embodiment": embodiment,
    }

    return environment_configuration


def get_environment_configuration_from_device_registry(
    device_name: str | None = None,
) -> dict[str, type["TeleopDeviceBase"]]:

    device_registry = DeviceRegistry()
    if device_name:
        print(f"Getting device {device_name} from device registry")
        assert device_registry.is_registered(device_name), f"Device {device_name} not registered"
        device = device_registry.get_device_by_name(device_name)()
    else:
        device = None

    return {
        "device": device,
    }


# Register all the assets. Down here at the bottom of the file because
# the assets use the AssetRegistry class in order to register themselves,
# so it needs to be fully defined to avoid a circular import.
from isaac_arena.assets.background import *  # noqa: F403, F401
from isaac_arena.assets.objects import *  # noqa: F403, F401
from isaac_arena.embodiments.franka.franka import *  # noqa: F403, F401
from isaac_arena.embodiments.gr1t2.gr1t2 import *  # noqa: F403, F401
from isaac_arena.teleop_devices.avp_handtracking import *  # noqa: F403, F401
from isaac_arena.teleop_devices.keyboard import *  # noqa: F403, F401
from isaac_arena.teleop_devices.spacemouse import *  # noqa: F403, F401
