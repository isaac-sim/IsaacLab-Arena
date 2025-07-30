# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

import importlib
import inspect
import pkgutil
import random

from isaac_arena.scene.asset import Asset
from isaac_arena.utils.singleton import SingletonMeta

SCANNED_PACKAGES = ["isaac_arena.scene"]


class AssetRegistry(metaclass=SingletonMeta):

    def __init__(self):
        self.base_packages = SCANNED_PACKAGES
        self.registry = {}
        self._auto_register()

    def register_asset(self, name: str, asset: Asset):
        """Register an asset with a name.

        Args:
            name (str): The name of the asset.
            asset (Asset): The asset to register.
        """
        assert name not in self.registry, f"Asset {name} already registered"
        self.registry[name] = asset

    def get_asset_by_name(self, name: str) -> Asset:
        """Get an asset by name.

        Args:
            name (str): The name of the asset.

        Returns:
            Asset: The asset.
        """
        return self.registry[name]

    def get_assets_by_tag(self, tag: str) -> list[Asset]:
        """Gets a list of assets by tag.

        Args:
            tag (str): The tag of the assets.

        Returns:
            list[Asset]: The list of assets.
        """
        return [asset for asset in self.registry.values() if tag in asset.tags]

    def get_random_asset_by_tag(self, tag: str) -> Asset:
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

    def _auto_register(self):
        """
        Walk through all submodules of each base package,
        import them, and register any Asset subclasses found.
        """
        for pkg_name in self.base_packages:
            pkg = importlib.import_module(pkg_name)
            # pkg.__path__ is only present if pkg is a package
            if not hasattr(pkg, "__path__"):
                continue

            # Go through all submodules of the base package
            for finder, submod_name, is_pkg in pkgutil.walk_packages(pkg.__path__, pkg_name + "."):
                try:
                    module = importlib.import_module(submod_name)
                except ImportError as e:
                    print(f"Fail to import a module during Asset search. Exception: {e}")
                    raise e
                self._register_from_module(module)

    def _register_from_module(self, module):
        """
        Inspect a module for any Asset subclasses (except Asset itself),
        instantiate them, and register under their `get_name()`.
        """
        for _, cls in inspect.getmembers(module, inspect.isclass):
            # must be in this module, subclass Asset, but not Asset itself
            if issubclass(cls, Asset) and cls is not Asset and cls.__module__ == module.__name__:
                # skip if __init__ has required (non-default) args beyond self
                # This is to avoid registering classes that have required arguments
                # beyond self, which are not needed for instantiation
                # All our assets have no required arguments beyond self for now.
                sig = inspect.signature(cls.__init__)
                # drop 'self'
                params = list(sig.parameters.values())[1:]
                # check if any param is positional/keyword without a default
                needs_arg = any(
                    p.default == inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                    for p in params
                )
                if needs_arg:
                    continue

                instance = cls()
                self.register_asset(instance.get_name(), instance)
