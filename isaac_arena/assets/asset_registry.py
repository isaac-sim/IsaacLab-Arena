# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

import random
from typing import TYPE_CHECKING

from isaac_arena.utils.singleton import SingletonMeta

# NOTE(alexmillane): Avoid circular import.
if TYPE_CHECKING:
    from isaac_arena.assets.asset import Asset  # only imported for type checking


class AssetRegistry(metaclass=SingletonMeta):

    def __init__(self):
        self.assets = {}

    def register(self, asset: type["Asset"]):
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

    def get_asset_by_name(self, name: str) -> type["Asset"]:
        """Get an asset by name.

        Args:
            name (str): The name of the asset.

        Returns:
            Asset: The asset.
        """
        return self.assets[name]

    def get_assets_by_tag(self, tag: str) -> list[type["Asset"]]:
        """Gets a list of assets by tag.

        Args:
            tag (str): The tag of the assets.

        Returns:
            list[Asset]: The list of assets.
        """
        return [asset for asset in self.assets.values() if tag in asset.tags]

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


# Register all the assets. Down here at the bottom of the file because
# the assets use the AssetRegistry class in order to register themselves,
# so it needs to be fully defined to avoid a circular import.
from isaac_arena.assets.background import *  # noqa: F403, F401
from isaac_arena.assets.objects import *  # noqa: F403, F401
