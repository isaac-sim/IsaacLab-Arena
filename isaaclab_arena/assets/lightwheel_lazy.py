# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from isaaclab_arena.assets.lightwheel_utils import acquire_lightwheel_asset


class LightwheelLazyPath:
    """Class-attribute descriptor that resolves a Lightwheel USD path on first access and caches it."""

    def __init__(
        self,
        registry_type: str,
        file_name: str | None = None,
        registry_name: list[str] | None = None,
        file_type: str = "USD",
    ):
        assert (file_name is None) != (
            registry_name is None
        ), "Provide exactly one of file_name= or registry_name= (matches acquire_by_registry's signature)."
        identifier = file_name if file_name is not None else registry_name
        self._description = f"Lightwheel asset {identifier!r}"
        self._query_kwargs: dict = {"registry_type": registry_type, "file_type": file_type}
        if file_name is not None:
            self._query_kwargs["file_name"] = file_name
        else:
            self._query_kwargs["registry_name"] = registry_name
        self._cached_path: str | None = None

    def __get__(self, instance, owner):
        if self._cached_path is not None:
            return self._cached_path
        from lightwheel_sdk.loader import object_loader

        file_path, _, _ = acquire_lightwheel_asset(
            object_loader,
            object_loader.acquire_by_registry,
            description=self._description,
            **self._query_kwargs,
        )
        self._cached_path = file_path
        return file_path
