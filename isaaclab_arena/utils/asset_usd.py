# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve registered Arena assets to on-disk USD paths."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab_arena.environment_spec.arena_env_graph_types import AssetSpec


def usd_cache_key(usd_path: str) -> str:
    """Return a stable short hash for caching artifacts keyed by USD path."""
    return hashlib.sha1(usd_path.encode("utf-8")).hexdigest()[:16]


def extract_asset_usd_path(asset_cls: type, **params: Any) -> str | None:
    """Return the asset's root USD path, or ``None`` if not extractable.

    Args:
        asset_cls: Registered asset class from :class:`AssetRegistry`.
        **params: Constructor kwargs (e.g. ``layout_id`` / ``style_id`` for lazy USD assets).
    """
    class_usd = getattr(asset_cls, "usd_path", None)
    if isinstance(class_usd, str) and class_usd:
        return class_usd

    try:
        instance = asset_cls(**params)
    except Exception:
        return None

    usd_path = getattr(instance, "usd_path", None)
    if usd_path:
        return str(usd_path)

    scene_config = getattr(instance, "scene_config", None)
    robot = getattr(scene_config, "robot", None) if scene_config is not None else None
    spawn = getattr(robot, "spawn", None) if robot is not None else None
    spawn_path = getattr(spawn, "usd_path", None) if spawn is not None else None
    return str(spawn_path) if spawn_path else None


def resolve_asset_usd_path(asset_spec: AssetSpec) -> str:
    """Return the on-disk USD path for an :class:`AssetSpec`."""
    import isaaclab_arena.assets.background_library  # noqa: F401
    import isaaclab_arena.assets.object_library  # noqa: F401
    from isaaclab_arena.assets.registries import AssetRegistry

    asset_cls = AssetRegistry().get_asset_by_name(asset_spec.registry_name)
    usd_path = extract_asset_usd_path(asset_cls, **asset_spec.params)
    assert usd_path, f"asset {asset_spec.registry_name!r} has no usd_path"
    return usd_path
