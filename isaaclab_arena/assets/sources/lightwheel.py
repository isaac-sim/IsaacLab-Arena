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


def lightwheel_cache_root() -> Path:
    """Return the directory where the Lightwheel SDK keeps its downloaded assets."""
    return Path("~/.cache/lightwheel_sdk").expanduser()


def find_cached_object_usd(file_name: str) -> str | None:
    """Return the path of a completely cached Lightwheel object USD, or None if it is not cached.

    The object is complete when ``object/<Name>/<Name>.usd`` exists and ``object/<Name>/`` holds a
    ``*.txt`` version marker. ``<Name>`` matches ``file_name`` case-insensitively.
    """
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


def floorplan_cache_name(scene: str, layout_id: int, style_id: int, backend: str) -> str:
    """Return the name the SDK gives a floorplan's cache directory and its ``usd_version.json`` entry."""
    return f"{backend}-{scene}-{layout_id}-{style_id}"


def find_cached_floorplan_usd(scene: str, layout_id: int, style_id: int, backend: str) -> str | None:
    """Return the path of a completely cached Lightwheel floorplan ``scene.usd``, or None if it is not cached.

    The floorplan is complete when ``usd_version.json`` has a key for ``{backend}-{scene}-{layout_id}-{style_id}``
    (its value may be null, which the SDK writes when the server omits the version) and
    ``floorplan/<that name>/scene.usd`` exists.
    """
    floorplan_root = lightwheel_cache_root() / "floorplan"
    name = floorplan_cache_name(scene, layout_id, style_id, backend)
    try:
        versions = json.loads((floorplan_root / "usd_version.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(versions, dict) or name not in versions:
        return None
    usd_path = floorplan_root / name / "scene.usd"
    return str(usd_path) if usd_path.is_file() else None


class LightwheelAssetSource(UsdSource, ABC):
    """Acquire and memoize one Lightwheel USD, using the SDK cache directory when the asset is complete there."""

    _asset_label: str = "asset"
    """Names the kind of asset in log and error messages."""

    def __init__(self, **sdk_kwargs: Any):
        self._kwargs = sdk_kwargs
        self._local_file_path: str | None = None

    def to_local_usd_path(self) -> str:
        """Return the local USD path, resolving it on first use and reusing it thereafter."""
        if self._local_file_path is None:
            self._local_file_path = self._resolve_usd_path()
        return self._local_file_path

    @abstractmethod
    def _cache_key(self) -> str | None:
        """Return the stable name used in log lines for this asset, or None if the request has no stable name."""

    @abstractmethod
    def _find_cached_usd(self) -> str | None:
        """Return the USD path of this asset if it is completely cached on disk, else None."""

    @abstractmethod
    def _sdk_loader(self) -> Any:
        """Import the Lightwheel SDK lazily and return the loader object for this kind of asset."""

    @abstractmethod
    def _sdk_acquire_fn(self, loader: Any) -> Callable:
        """Return the method of ``loader`` that fetches this asset and returns a tuple starting with its path."""

    def _resolve_usd_path(self) -> str:
        """Return the cached USD path, or acquire it through the SDK on a cache miss."""
        key = self._cache_key()
        if key is None:
            print(f"[isaaclab-arena] Lightwheel request has no stable name and is not cacheable: {self._kwargs}")
        else:
            cached_path = self._find_cached_usd()
            if cached_path is not None:
                print(f"[isaaclab-arena] Lightwheel cache hit: {key}")
                return cached_path
            print(f"[isaaclab-arena] Lightwheel cache miss: {key}")
        loader = self._sdk_loader()
        result = acquire_lightwheel_asset(
            loader,
            self._sdk_acquire_fn(loader),
            description=f"Lightwheel {self._asset_label} {self._kwargs}",
            **self._kwargs,
        )
        file_path = result[0]
        assert file_path, f"Lightwheel did not return a {self._asset_label} USD path for {self._kwargs}"
        return str(file_path)


class LightwheelUsdSource(LightwheelAssetSource):
    """Acquire and memoize one Lightwheel object USD on first use; the keyword arguments go to ``acquire_by_registry``."""

    def _cache_key(self) -> str | None:
        file_name = self._kwargs.get("file_name")
        return f"lightwheel-object/{file_name.lower()}" if isinstance(file_name, str) else None

    def _find_cached_usd(self) -> str | None:
        return find_cached_object_usd(self._kwargs["file_name"])

    def _sdk_loader(self) -> Any:
        from lightwheel_sdk.loader import object_loader

        return object_loader

    def _sdk_acquire_fn(self, loader: Any) -> Callable:
        return loader.acquire_by_registry


class LightwheelFloorplanUsdSource(LightwheelAssetSource):
    """Acquire and memoize one Lightwheel floorplan USD on first use."""

    _asset_label = "floorplan"

    def __init__(self, scene: str, layout_id: int, style_id: int, backend: str):
        super().__init__(scene=scene, layout_id=layout_id, style_id=style_id, backend=backend)

    def _cache_key(self) -> str:
        return f"lightwheel-floorplan/{floorplan_cache_name(**self._kwargs)}"

    def _find_cached_usd(self) -> str | None:
        return find_cached_floorplan_usd(**self._kwargs)

    def _sdk_loader(self) -> Any:
        from lightwheel_sdk.loader import floorplan_loader

        return floorplan_loader

    def _sdk_acquire_fn(self, loader: Any) -> Callable:
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
