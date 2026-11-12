# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Location of the on-disk cache for assets downloaded by Arena."""


import pathlib


def get_arena_asset_cache_dir() -> pathlib.Path:
    """Return the asset cache directory under the user's home directory, creating it if it does not exist."""
    home_path = pathlib.Path.home()
    asset_cache_dir = home_path / ".cache" / "isaaclab_arena" / "assets"
    if not asset_cache_dir.exists():
        asset_cache_dir.mkdir(parents=True, exist_ok=True)
    return asset_cache_dir
