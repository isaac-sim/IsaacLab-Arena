# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lazy Lightwheel USD acquisition."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from isaaclab_arena.assets.sources.base import UsdSource

_MISSING = object()


class LightwheelUsdSource(UsdSource):
    """Acquire and memoize one Lightwheel USD on first use."""

    def __init__(self, **acquire_by_registry_kwargs: Any):
        """Create a source for one Lightwheel asset.

        Args:
            **acquire_by_registry_kwargs: Keyword arguments of the Lightwheel object loader's `acquire_by_registry`
                call that selects the asset.
        """
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
        """Create a source for one Lightwheel floorplan.

        Args:
            scene: Name of the Lightwheel scene.
            layout_id: Floorplan layout index.
            style_id: Floorplan style index.
            backend: Name of the Lightwheel floorplan backend.
        """
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
    """Acquire a Lightwheel asset with scoped timeout and retry handling.

    Calls `acquire_fn` and retries it when it fails with a timeout. The loader client's timeout is set for the
    call and restored afterwards.

    Args:
        loader: Lightwheel loader whose client timeout is set while acquiring.
        acquire_fn: Callable that acquires the asset.
        description: Text that identifies the asset in retry messages.
        attempts: Maximum number of calls to `acquire_fn`. Must be positive.
        timeout_sec: Client timeout in seconds during each call. None leaves the client timeout unchanged.
        delay_sec: Seconds to wait before a retry. Must be non-negative.
        **kwargs: Keyword arguments for `acquire_fn`.

    Returns:
        The value returned by `acquire_fn`.
    """
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
