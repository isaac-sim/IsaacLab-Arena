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

from isaac_arena.utils.singleton import SingletonMeta

if TYPE_CHECKING:
    from isaac_arena.assets.asset import Asset
    from isaac_arena.teleop_devices.teleop_device_base import TeleopDeviceBase


class Registry(metaclass=SingletonMeta):

    def __init__(self):
        self.assets = {}

    def register(self, asset: type["Asset" | "TeleopDeviceBase"]):
        """Register an asset with a name.

        Args:
            name (str): The name of the asset.
            asset (Asset): The asset to register.
        """
        assert asset.name not in self.assets, f"Asset {asset.name} already registered"
        assert asset.name is not None, "Asset name is not set"
        assert asset.tags is not None, "Asset tags are not set"
        self.assets[asset.name] = asset

    def is_registered(self, name: str) -> bool:
        """Check if an asset is registered.

        Args:
            name (str): The name of the asset.
        """
        return name in self.assets

    def get_asset_by_name(self, name: str) -> type["Asset" | "TeleopDeviceBase"]:
        """Get an asset by name.

        Args:
            name (str): The name of the asset.

        Returns:
            Asset: The asset.
        """
        return self.assets[name]
