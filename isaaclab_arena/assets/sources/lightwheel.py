# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lazy Lightwheel USD acquisition."""

from __future__ import annotations

import json
import time
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


def find_cached_floorplan_usd(scene: str, layout_id: int, style_id: int, backend: str) -> str | None:
    """Return the path of a completely cached Lightwheel floorplan ``scene.usd``, or None if it is not cached.

    The floorplan is complete when ``usd_version.json`` has an entry for ``{backend}-{scene}-{layout_id}-{style_id}``
    and ``floorplan/<that name>/scene.usd`` exists.
    """
    floorplan_root = lightwheel_cache_root() / "floorplan"
    name = f"{backend}-{scene}-{layout_id}-{style_id}"
    try:
        versions = json.loads((floorplan_root / "usd_version.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(versions, dict) or not versions.get(name):
        return None
    usd_path = floorplan_root / name / "scene.usd"
    return str(usd_path) if usd_path.is_file() else None


class LightwheelUsdSource(UsdSource):
    """Acquire and memoize one Lightwheel USD on first use."""

    def __init__(self, **acquire_by_registry_kwargs: Any):
        self._kwargs = acquire_by_registry_kwargs
        self._local_file_path: str | None = None

    def to_local_usd_path(self) -> str:
        """Acquire the USD on first use and reuse that path thereafter."""
        if self._local_file_path is None:
            from lightwheel_sdk.loader import object_loader

            file_path, _, _ = acquire_lightwheel_asset(
                object_loader,
                object_loader.acquire_by_registry,
                description=f"Lightwheel asset {self._kwargs}",
                **self._kwargs,
            )
            assert file_path, f"Lightwheel did not return a USD path for {self._kwargs}"
            self._local_file_path = str(file_path)
        return self._local_file_path


class LightwheelFloorplanUsdSource(UsdSource):
    """Acquire and memoize one Lightwheel floorplan USD on first use."""

    def __init__(self, scene: str, layout_id: int, style_id: int, backend: str):
        self._kwargs = {
            "scene": scene,
            "layout_id": layout_id,
            "style_id": style_id,
            "backend": backend,
        }
        self._local_file_path: str | None = None

    def to_local_usd_path(self) -> str:
        """Acquire the floorplan USD on first use and reuse that path thereafter."""
        if self._local_file_path is None:
            from lightwheel_sdk.loader import floorplan_loader

            file_path = acquire_lightwheel_asset(
                floorplan_loader,
                floorplan_loader.get_usd,
                description=f"Lightwheel floorplan {self._kwargs}",
                **self._kwargs,
            )[0]
            assert file_path, f"Lightwheel did not return a floorplan USD path for {self._kwargs}"
            self._local_file_path = str(file_path)
        return self._local_file_path


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
