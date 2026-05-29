# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from isaaclab_arena.assets.lightwheel_utils import acquire_lightwheel_asset


class LightwheelLazyPath:
    """Class-attribute descriptor that resolves a Lightwheel USD path on first access and caches it."""

    def __init__(self, **acquire_by_registry_kwargs):
        self._kwargs = acquire_by_registry_kwargs
        self._cached_path: str | None = None

    def __get__(self, instance, owner):
        if self._cached_path is not None:
            return self._cached_path
        from lightwheel_sdk.loader import object_loader

        file_path, _, _ = acquire_lightwheel_asset(
            object_loader,
            object_loader.acquire_by_registry,
            description=f"Lightwheel asset {self._kwargs}",
            **self._kwargs,
        )
        self._cached_path = file_path
        return file_path
