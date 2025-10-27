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


from isaaclab_arena.assets.asset_registry import AssetRegistry, DeviceRegistry


# Decorator to register an asset with the AssetRegistry.
def register_asset(cls):
    if AssetRegistry().is_registered(cls.name):
        print(f"WARNING: Asset {cls.name} is already registered. Doing nothing.")
    else:
        AssetRegistry().register(cls)
    return cls


# Decorator to register an device with the DeviceRegistry.
def register_device(cls):
    if DeviceRegistry().is_registered(cls.name):
        print(f"WARNING: Device {cls.name} is already registered. Doing nothing.")
    else:
        DeviceRegistry().register(cls)
    return cls
