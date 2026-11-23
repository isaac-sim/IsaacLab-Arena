# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lazy Lightwheel USD acquisition."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from isaaclab_arena.assets.sources.base import UsdSource

_MISSING = object()

LIGHTWHEEL_SDK_CACHE_ROOT = "~/.cache/lightwheel_sdk"
"""Directory where the Lightwheel SDK keeps its downloaded assets, before ``~`` is expanded."""


def lightwheel_cache_root() -> Path:
    """Return the directory where the Lightwheel SDK keeps its downloaded assets."""
    return Path(LIGHTWHEEL_SDK_CACHE_ROOT).expanduser()


class LightwheelSource(UsdSource, ABC):
    """Acquire and memoize one Lightwheel USD, using the SDK cache directory when the asset is complete there.

    The cache lookup avoids error-prone API and remote calls when a local copy exists.
    """

    _asset_label: str = "asset"
    """Names the kind of asset in log and error messages."""

    def __init__(self, **sdk_kwargs: Any):
        """Store the request without contacting the SDK.

        Args:
            **sdk_kwargs: Keyword arguments passed to the SDK method that downloads the asset.
        """
        self._kwargs = sdk_kwargs
        self._local_file_path: str | None = None

    def to_local_usd_path(self) -> str:
        """Return the local USD path, resolving it on first use and reusing it thereafter."""
        if self._local_file_path is None:
            self._local_file_path = self._get_cached_or_download()
        return self._local_file_path

    @abstractmethod
    def _get_cache_key(self) -> str | None:
        """Return the stable name used in log lines for this asset, or None if the request has no stable name."""

    @abstractmethod
    def _get_cached_usd_path(self) -> str | None:
        """Return the USD path of this asset if it is completely cached on disk, else None."""

    @abstractmethod
    def _get_lightwheel_sdk_loader(self) -> Any:
        """Import the Lightwheel SDK lazily and return the loader object for this kind of asset."""

    @abstractmethod
    def _get_lightwheel_sdk_download_fn(self, loader: Any) -> Callable:
        """Return the method of ``loader`` that downloads this asset and returns a tuple starting with its path.

        Args:
            loader: The loader object returned by ``_get_lightwheel_sdk_loader``.
        """

    def _get_cached_or_download(self) -> str:
        """Return the cached USD path, or download the asset through the SDK on a cache miss."""
        key = self._get_cache_key()
        cached_path = None
        if key is None:
            print(f"[isaaclab-arena] Lightwheel request has no stable name and is not cacheable: {self._kwargs}")
        else:
            cached_path = self._get_cached_usd_path()
            print(f"[isaaclab-arena] Lightwheel cache {'hit' if cached_path is not None else 'miss'}: {key}")
        if cached_path is not None:
            usd_path = cached_path
        else:
            loader = self._get_lightwheel_sdk_loader()
            result = acquire_lightwheel_asset(
                loader,
                self._get_lightwheel_sdk_download_fn(loader),
                description=f"Lightwheel {self._asset_label} {self._kwargs}",
                **self._kwargs,
            )
            downloaded_path = result[0]
            assert downloaded_path, f"Lightwheel did not return a {self._asset_label} USD path for {self._kwargs}"
            usd_path = str(downloaded_path)
        return usd_path


class LightwheelObjectUsdSource(LightwheelSource):
    """Acquire and memoize one Lightwheel object USD on first use."""

    def _get_cache_key(self) -> str | None:
        file_name = self._kwargs.get("file_name")
        return f"lightwheel-object/{file_name.lower()}" if isinstance(file_name, str) else None

    def _get_cached_usd_path(self) -> str | None:
        """Return ``object/<Name>/<Name>.usd`` if it exists next to a ``*.txt`` version marker.

        The SDK writes the marker after extracting the object. ``<Name>`` matches ``file_name`` case-insensitively
        because the server names the directory from the download URL.
        """
        file_name = self._kwargs["file_name"]
        object_root = lightwheel_cache_root() / "object"
        if not object_root.is_dir():
            return None
        for object_dir in object_root.iterdir():
            if object_dir.name.lower() != file_name.lower() or not object_dir.is_dir():
                continue
            usd_path = object_dir / f"{object_dir.name}.usd"
            has_marker = any(object_dir.glob("*.txt"))
            if usd_path.is_file() and has_marker:
                return str(usd_path)
        return None

    def _get_lightwheel_sdk_loader(self) -> Any:
        from lightwheel_sdk.loader import object_loader

        return object_loader

    def _get_lightwheel_sdk_download_fn(self, loader: Any) -> Callable:
        """Return ``loader.acquire_by_registry``, the method that downloads this asset.

        Args:
            loader: The loader object returned by ``_get_lightwheel_sdk_loader``.
        """
        return loader.acquire_by_registry


class LightwheelFloorplanUsdSource(LightwheelSource):
    """Acquire and memoize one Lightwheel floorplan USD on first use."""

    _asset_label = "floorplan"

    def __init__(self, scene: str, layout_id: int, style_id: int, backend: str):
        """Describe one floorplan request.

        Args:
            scene: Name of the scene in the Lightwheel floorplan catalog.
            layout_id: Identifier of the scene layout.
            style_id: Identifier of the scene style.
            backend: Name of the simulation backend whose floorplan variant to fetch.
        """
        super().__init__(scene=scene, layout_id=layout_id, style_id=style_id, backend=backend)

    def _cache_name(self) -> str:
        """Return the name the SDK gives this floorplan's cache directory and its ``usd_version.json`` entry."""
        kwargs = self._kwargs
        return f"{kwargs['backend']}-{kwargs['scene']}-{kwargs['layout_id']}-{kwargs['style_id']}"

    def _get_cache_key(self) -> str:
        return f"lightwheel-floorplan/{self._cache_name()}"

    def _get_cached_usd_path(self) -> str | None:
        """Return ``floorplan/<name>/scene.usd`` if ``usd_version.json`` has a key for ``<name>``.

        The SDK writes the version entry after extracting the floorplan. Its value may be null, which the SDK writes
        when the server omits the version.
        """
        floorplan_root = lightwheel_cache_root() / "floorplan"
        name = self._cache_name()
        try:
            versions = json.loads((floorplan_root / "usd_version.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(versions, dict) or name not in versions:
            return None
        usd_path = floorplan_root / name / "scene.usd"
        return str(usd_path) if usd_path.is_file() else None

    def _get_lightwheel_sdk_loader(self) -> Any:
        from lightwheel_sdk.loader import floorplan_loader

        return floorplan_loader

    def _get_lightwheel_sdk_download_fn(self, loader: Any) -> Callable:
        """Return ``loader.get_usd``, the method that downloads this asset.

        Args:
            loader: The loader object returned by ``_get_lightwheel_sdk_loader``.
        """
        return loader.get_usd


def acquire_lightwheel_asset(
    loader: Any,
    acquire_fn: Callable,
    description: str,
    attempts: int = 3,
    timeout_sec: int | None = 60,
    delay_sec: float = 2.0,
    **kwargs: Any,
) -> Any:
    """Acquire a Lightwheel asset with scoped timeout and retry handling."""
    assert attempts > 0, "attempts must be positive"
    assert delay_sec >= 0, "delay_sec must be non-negative"

    client = getattr(loader, "client", None)
    old_timeout = getattr(client, "base_timeout", _MISSING)
    for attempt in range(1, attempts + 1):
        try:
            if timeout_sec is not None and old_timeout is not _MISSING:
                client.base_timeout = timeout_sec
            return acquire_fn(**kwargs)
        except Exception as exc:
            if not _looks_like_timeout(exc) or attempt == attempts:
                raise
            print(f"[isaaclab-arena] {description} timed out; retrying {attempt + 1}/{attempts} in {delay_sec:g}s.")
            if delay_sec > 0:
                time.sleep(delay_sec)
        finally:
            if old_timeout is not _MISSING:
                client.base_timeout = old_timeout

    raise AssertionError("unreachable")


def _looks_like_timeout(exc: BaseException) -> bool:
    """Return whether an exception is a timeout."""
    current: BaseException | None = exc
    while current is not None:
        text = f"{type(current).__name__}: {current}".lower()
        if "timeout" in text or "timed out" in text:
            return True
        current = current.__cause__ or current.__context__
    return False
