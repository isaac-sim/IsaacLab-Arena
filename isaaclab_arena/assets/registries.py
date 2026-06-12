# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import random
from typing import TYPE_CHECKING, Any

from isaaclab_arena.utils.singleton import SingletonMeta

if TYPE_CHECKING:
    from isaaclab.devices.device_base import DeviceCfg
    from isaaclab_teleop import IsaacTeleopCfg

    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.assets.hdr_image import HDRImage
    from isaaclab_arena.assets.teleop_device_base import TeleopDeviceBase
    from isaaclab_arena.policy.policy_base import PolicyBase
    from isaaclab_arena.relations.relations import RelationBase
    from isaaclab_arena.tasks.task_base import TaskBase


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

    def is_registered(self, key: str, ensure_loaded: bool = True) -> bool:
        """Check whether a component is already registered under ``key``.

        Args:
            key: The name to look up.
            ensure_loaded: Whether to load every component before answering.

                Components register themselves lazily: nothing is in the registry until
                ``ensure_assets_registered()`` imports all the library modules. So a plain
                membership check can say "not registered" simply because the libraries
                haven't been imported yet. With ``ensure_loaded=True`` (the default) we
                import them first, so the answer reflects everything that exists.

                The ``register_*`` decorators pass ``False``. They run *while* those
                library modules are being imported, and all they need is to spot a
                duplicate key. Forcing a full load at that moment would re-enter the
                import that's already in progress and pull in the task/environment modules
                — which import Isaac Sim's ``pxr``/USD packages. If that happens during
                pytest collection (before ``SimulationApp()`` starts) the simulator
                segfaults, because those packages must be imported only after it starts.
        """
        if ensure_loaded and isinstance(self, REGISTRIES):
            ensure_assets_registered()
        return key in self._components

    def get_component_by_name(self, key: str) -> Any:
        """Get an component by name.

        Args:
            key (str): The name of the component.

        Returns:
            Any: The component.
        """
        if isinstance(self, REGISTRIES):
            ensure_assets_registered()
        assert key in self._components, f"component {key} not found, please check if requested component is registered"
        return self._components[key]

    def get_all_keys(self) -> list[str]:
        """Get all the keys of the components.

        Returns:
            list[str]: The list of keys.
        """
        if isinstance(self, REGISTRIES):
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

    def get_assets_with_all_tags(self, tags: list[str]) -> list[str]:
        """Return asset names whose ``tags`` include every tag in ``tags``.

        Args:
            tags: Tags that must all be present on a candidate asset.
                When empty, every registered asset name is returned.

        Returns:
            Sorted asset names matching every tag, or all asset names when
            ``tags`` is empty.
        """
        ensure_assets_registered()
        return sorted(asset.name for asset in self._components.values() if all(tag in asset.tags for tag in tags))

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

    def get_teleop_device_cfg(
        self, device: type["TeleopDeviceBase"], embodiment: object
    ) -> "DeviceCfg | IsaacTeleopCfg":
        retargeter_registry = RetargeterRegistry()
        retargeter_key = (device.name, embodiment.name)
        retargeter_key_str = retargeter_registry.convert_tuple_to_str(retargeter_key)
        retargeter = retargeter_registry.get_component_by_name(retargeter_key_str)()
        pipeline_builder = retargeter.get_pipeline_builder(embodiment)
        return device.get_device_cfg(pipeline_builder=pipeline_builder, embodiment=embodiment)


class RetargeterRegistry(Registry):
    def __init__(self):
        super().__init__()

    def convert_tuple_to_str(self, key: tuple[str, str]) -> str:
        # Double underscore is used to separate device and embodiment names.
        return f"{key[0]}__{key[1]}"

    def convert_str_to_tuple(self, key: str) -> tuple[str, str]:
        # Double underscore is used to separate device and embodiment names.
        return (key.split("__")[0], key.split("__")[1])


class PolicyRegistry(Registry):
    def __init__(self):
        super().__init__()

    def get_policy(self, name: str) -> type["PolicyBase"]:
        """Gets a policy by name.

        Args:
            name (str): The name of the policy.
        """
        ensure_assets_registered()
        return self.get_component_by_name(name)


class HDRImageRegistry(Registry):
    """Registry for HDR/EXR environment map textures."""

    def __init__(self):
        super().__init__()

    def get_hdr_by_name(self, name: str) -> type["HDRImage"]:
        """Gets an HDRImage class by name.

        Args:
            name (str): The name of the HDRImage.
        """
        ensure_assets_registered()
        return self.get_component_by_name(name)

    def get_hdrs_by_tag(self, tag: str) -> list[type["HDRImage"]]:
        """Gets a list of HDRImage classes that have the given tag.

        Args:
            tag (str): The tag to filter by.

        Returns:
            list[type[HDRImage]]: The matching HDRImage classes.
        """
        ensure_assets_registered()
        return [hdr for hdr in self._components.values() if tag in hdr.tags]

    def get_random_hdr_by_tag(self, tag: str) -> type["HDRImage"]:
        """Gets a random HDRImage class which has the given tag.

        Args:
            tag (str): The tag to filter by.

        Returns:
            type[HDRImage]: A random HDRImage class.
        """
        ensure_assets_registered()
        hdrs = self.get_hdrs_by_tag(tag)
        if len(hdrs) == 0:
            raise ValueError(f"No HDRs found with tag {tag}")
        return random.choice(hdrs)


class EnvironmentRegistry(Registry):
    """Registry for example environment classes."""

    def __init__(self):
        super().__init__()


class ObjectRelationLibraryRegistry(Registry):
    """Registry for object relation classes."""

    def __init__(self):
        super().__init__()

    def get_object_relation_by_name(self, name: str) -> type["RelationBase"]:
        """Gets an object relation by name.

        Args:
            name (str): The name of the object relation.
        """
        ensure_assets_registered()
        return self.get_component_by_name(name)


class TaskRegistry(Registry):
    """Registry for TaskBase subclasses."""

    def __init__(self):
        super().__init__()

    def get_task_by_name(self, name: str) -> type["TaskBase"]:
        """Gets a task class by name.

        Args:
            name (str): The name of the task class (typically the class __name__).
        """
        ensure_assets_registered()
        return self.get_component_by_name(name)


# Registries populated lazily by ensure_assets_registered(). EnvironmentRegistry is
# excluded: triggering the cascade during env registration causes an env<->tasks cycle.
REGISTRIES = (
    AssetRegistry,
    DeviceRegistry,
    RetargeterRegistry,
    PolicyRegistry,
    HDRImageRegistry,
    ObjectRelationLibraryRegistry,
    TaskRegistry,
)


# Lazy registration to avoid circular imports
_assets_registered = False
# Blocks re-entry: registration decorators call is_registered() -> ensure_assets_registered()
# mid-import, which would re-import a partial module and raise a circular ImportError.
_registration_in_progress = False


def ensure_assets_registered():
    """Ensure all assets are registered. Call this before accessing the registry."""
    global _assets_registered, _registration_in_progress
    if _assets_registered or _registration_in_progress:
        return
    _registration_in_progress = True
    try:
        # Import modules to trigger asset registration via decorators
        import isaaclab_arena.assets.background_library  # noqa: F401
        import isaaclab_arena.assets.device_library  # noqa: F401
        import isaaclab_arena.assets.hdr_image_library  # noqa: F401
        import isaaclab_arena.assets.object_library  # noqa: F401
        import isaaclab_arena.assets.retargeter_library  # noqa: F401
        import isaaclab_arena.embodiments  # noqa: F401
        import isaaclab_arena.policy  # noqa: F401
        import isaaclab_arena.relations.relations  # noqa: F401
        import isaaclab_arena.tasks.task_library  # noqa: F401

        _assets_registered = True
    finally:
        _registration_in_progress = False
