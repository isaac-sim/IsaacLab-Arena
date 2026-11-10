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
        self._kwargs = acquire_by_registry_kwargs
        self._resolved: str | None = None

    def to_local_usd_path(self) -> str:
        """Acquire the USD on first use and reuse that path thereafter."""
        if self._resolved is None:
            from lightwheel_sdk.loader import object_loader

            file_path, _, _ = acquire_lightwheel_asset(
                object_loader,
                object_loader.acquire_by_registry,
                description=f"Lightwheel asset {self._kwargs}",
                **self._kwargs,
            )
            self._resolved = str(file_path)
        return self._resolved


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
