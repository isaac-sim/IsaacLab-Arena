# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lazy resolver for Lightwheel-SDK-backed USD paths.

Background
----------
Several asset classes in :mod:`isaaclab_arena.assets.object_library` were initially
defined with eager Lightwheel SDK calls in their *class body*::

    @register_asset
    class Microwave(LibraryObject, Openable):
        from lightwheel_sdk.loader import object_loader  # class-body import
        file_path, _, _ = object_loader.acquire_by_registry(  # class-body HTTP
            registry_type="fixtures", file_name="Microwave039", file_type="USD"
        )
        usd_path = file_path

Class bodies are evaluated **at module import time**, which means importing
``object_library`` makes a synchronous HTTP request to the Lightwheel API. If the
request times out or fails (the SDK has a 10-second read timeout), the class body
raises and the module import / reload fails — *every* class defined below the failing
one in that module is left undefined and unregistered, including unrelated assets
like ``RubiksCubeHot3DRobolab``. This was the root cause of a registry-loss bug we
hit during a long multi-job eval sweep: a transient Lightwheel API stall broke an
``importlib.reload(object_library)`` call, ``rubiks_cube_hot3d_robolab`` failed to
re-register, and subsequent jobs crashed with a misleading "asset not found" error.

What this module provides
-------------------------
:class:`LightwheelLazyPath` is a class-attribute descriptor that **defers** the
Lightwheel SDK call from class-body evaluation to first attribute access. After the
fix the same class looks like::

    @register_asset
    class Microwave(LibraryObject, Openable):
        usd_path = LightwheelLazyPath("fixtures", "Microwave039")
        # ... no class-body network call ...

Import / reload of ``object_library`` no longer touches the network. Failures, when
they happen, surface at the moment the asset is actually instantiated rather than
silently corrupting unrelated registrations.

Caching is per-descriptor: the resolved path is stored on the descriptor instance
after the first successful resolution, so subsequent accesses are free and the
underlying SDK call is made at most once per process.

The actual SDK call is routed through :func:`isaaclab_arena.assets.lightwheel_utils.acquire_lightwheel_asset`,
which applies a longer scoped timeout and retries transient timeout failures.
"""

from __future__ import annotations

from isaaclab_arena.assets.lightwheel_utils import acquire_lightwheel_asset


class LightwheelLazyPath:
    """Descriptor that defers a Lightwheel ``object_loader`` call to first access.

    Instantiate as a class attribute on a Lightwheel-backed asset::

        usd_path = LightwheelLazyPath("fixtures", "Microwave039")

    or, for the list-style registry signature used by a couple of object classes::

        usd_path = LightwheelLazyPath("objects", registry_name=["broccoli"])

    The first access (``instance.usd_path`` or ``Class.usd_path``) imports the SDK,
    issues the registry query, caches the resolved path on the descriptor, and
    returns it. Subsequent accesses return the cached value without further work.

    Args:
        registry_type: Lightwheel registry partition (``"fixtures"``, ``"objects"``, …).
        file_name: File-name lookup key. Mutually exclusive with ``registry_name``;
            exactly one must be supplied.
        registry_name: List-style lookup key. Mutually exclusive with ``file_name``.
        file_type: Asset file format passed through to ``acquire_by_registry``.
            Defaults to ``"USD"`` which is what every existing call site uses.
    """

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
