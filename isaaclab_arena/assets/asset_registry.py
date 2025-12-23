# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import random
from typing import TYPE_CHECKING, Any

from isaaclab_arena.utils.singleton import SingletonMeta

if TYPE_CHECKING:
    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.assets.teleop_device_base import TeleopDeviceBase


# Have to define all classes here in order to avoid circular import.
class Registry(metaclass=SingletonMeta):

    def __init__(self):
        self._components = {}

    def register(self, component: Any, key: str | None = None):
        """Register an asset with a name.

        Args:
            key (str): The name of the asset.
            asset (Asset): The asset to register.
        """
        assert key not in self._components, f"component {key} already registered"
        assert key is not None, "component name is not set"
        self._components[key] = component

    def is_registered(self, key: str) -> bool:
        """Check if an component is registered.

        Args:
            key (str): The name of the component.
        """
        # For AssetRegistry and DeviceRegistry, ensure assets are registered before checking
        if isinstance(self, (AssetRegistry, DeviceRegistry, RetargeterRegistry)):
            ensure_assets_registered()
        return key in self._components

    def get_component_by_name(self, key: str) -> Any:
        """Get an component by name.

        Args:
            key (str): The name of the component.

        Returns:
            Asset: The component.
        """
        # For AssetRegistry and DeviceRegistry, ensure assets are registered before accessing
        if isinstance(self, (AssetRegistry, DeviceRegistry, RetargeterRegistry)):
            ensure_assets_registered()
        assert key in self._components, f"component {key} not found, please check if requested component is registered"
        return self._components[key]

    def get_all_keys(self) -> list[str]:
        """Get all the keys of the components.

        Returns:
            list[str | tuple[str, str]]: The list of keys.
        """
        # For AssetRegistry and DeviceRegistry, ensure assets are registered before accessing
        if isinstance(self, (AssetRegistry, DeviceRegistry, RetargeterRegistry)):
            ensure_assets_registered()
        return list(self._components.keys())


class AssetRegistry(Registry):

    def __init__(self):
        super().__init__()

    def get_asset_by_name(self, name: str) -> type["Asset"]:
        """Gets an asset by name.

        Args:
            name (str): The name of the asset.
        """
        ensure_assets_registered()
        return self.get_component_by_name(name)

    def get_assets_by_tag(self, tag: str) -> list[type["Asset"]]:
        """Gets a list of assets by tag.

        Args:
            tag (str): The tag of the assets.

        Returns:
            list[Asset]: The list of assets.
        """
        ensure_assets_registered()
        return [asset for asset in self._components.values() if tag in asset.tags]

    def get_random_asset_by_tag(self, tag: str) -> type["Asset"]:
        """Gets a random asset which has the given tag.

        Args:
            tag (str): The tag of the assets.

        Returns:
            Asset: The random asset.
        """
        ensure_assets_registered()
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
        ensure_assets_registered()
        return self.get_component_by_name(name)

    def get_teleop_device_cfg(self, device: type["TeleopDeviceBase"], embodiment: object):
        from isaaclab.devices.device_base import DevicesCfg

        retargeter_registry = RetargeterRegistry()
        retargeter_key = (device.name, embodiment.name)
        retargeter_key_str = retargeter_registry.convert_tuple_to_str(retargeter_key)
        retargeter = retargeter_registry.get_component_by_name(retargeter_key_str)()
        retargeter_cfg = retargeter.get_retargeter_cfg(embodiment, sim_device=device.sim_device)
        retargeters = [retargeter_cfg] if retargeter_cfg is not None else []
        device_cfg = device.get_device_cfg(retargeters=retargeters, embodiment=embodiment)
        return DevicesCfg(
            devices={
                device.name: device_cfg,
            }
        )


class RetargeterRegistry(Registry):
    def __init__(self):
        super().__init__()

    def convert_tuple_to_str(self, key: tuple[str, str]) -> str:
        # Double underscore is used to separate device and embodiment names.
        return f"{key[0]}__{key[1]}"

    def convert_str_to_tuple(self, key: str) -> tuple[str, str]:
        # Double underscore is used to separate device and embodiment names.
        return (key.split("__")[0], key.split("__")[1])


# Lazy registration to avoid circular imports
_assets_registered = False


def ensure_assets_registered():
    """Ensure all assets are registered. Call this before accessing the registry."""
    global _assets_registered
    if not _assets_registered:
        # Import modules to trigger asset registration via decorators
        import isaaclab_arena.assets.background_library  # noqa: F401
        import isaaclab_arena.assets.device_library  # noqa: F401
        import isaaclab_arena.assets.object_library  # noqa: F401
        import isaaclab_arena.assets.retargeter_library  # noqa: F401
        import isaaclab_arena.embodiments  # noqa: F401

        _assets_registered = True
