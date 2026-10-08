# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve instantiated graph assets to USD paths and local AABB dimensions (no Kit viewport)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils.dict import dict_to_md5_hash

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectBase
from isaaclab_arena.assets.object_geometry import ObjectGeometry
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.scene.scene import Scene

AabbDimensionsM = tuple[float, float, float]


def aabb_dimensions_from_asset(
    asset: ObjectBase | ObjectGeometry,
) -> AabbDimensionsM | None:
    """Return local axis-aligned bounding box size (x, y, z) in meters for one live asset."""
    try:
        bbox = asset.get_bounding_box()
        size = bbox.size[0]
        return (float(size[0]), float(size[1]), float(size[2]))
    except Exception as exc:
        name = getattr(asset, "name", "?")
        print(f"[asset_usd]   {name}: bbox failed: {exc}", file=sys.stderr)
        return None


def resolve_aabb_dimensions_m(
    assets_by_node_id: dict[str, Any],
) -> dict[str, AabbDimensionsM]:
    """Return axis-aligned bounding box sizes in meters for each snapshot asset (objects and references)."""
    dimensions: dict[str, AabbDimensionsM] = {}
    for node_id, asset in assets_by_node_id.items():
        if not isinstance(asset, ObjectBase) or getattr(asset, "has_variants", False):
            continue
        dims = aabb_dimensions_from_asset(asset)
        if dims is not None:
            dimensions[node_id] = dims
    return dimensions


def resolve_node_usd_paths(assets_by_node_id: dict[str, object], node_ids: list[str]) -> dict[str, str]:
    """Map each requested ``node_id`` to its ``usd_path``, skipping assets without one."""
    paths: dict[str, str] = {}
    for node_id in node_ids:
        asset = assets_by_node_id.get(node_id)
        if asset is None:
            print(f"[asset_usd]   {node_id}: not found in instantiated assets, skipping.", file=sys.stderr)
            continue
        usd_path = getattr(getattr(asset, "spawn_cfg", None), "usd_path", None)
        if usd_path:
            paths[node_id] = usd_path
    return paths


def cached_rigid_asset_preview(spawn_cfg: SpawnerCfg, cache_dir: Path) -> str:
    """Return a cached preview USD with the native source selections, scale, and materials applied.

    Args:
        spawn_cfg: Native configuration for one rigid asset.
        cache_dir: Directory used for cached review-GUI thumbnails and preview assets.

    Returns:
        Path to a flattened USD authored by the native spawner.
    """
    config_hash = dict_to_md5_hash(spawn_cfg)
    preview_path = cache_dir / f"native_asset_{config_hash}.usd"
    if not preview_path.is_file() or preview_path.stat().st_size == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
        asset = Object(name="asset", spawner_cfg=spawn_cfg, object_type=ObjectType.RIGID)
        Scene([asset]).export_to_usd(preview_path)
    return str(preview_path)


def usd_cache_key(usd_path: str) -> str:
    """Return a stable short hash for caching thumbnails keyed by USD path."""
    return hashlib.sha1(usd_path.encode("utf-8")).hexdigest()[:16]


def object_reference_cache_key(usd_path: str, relative_prim_path: str) -> str:
    """Return a stable cache key for an object_reference subtree snapshot."""
    return hashlib.sha1(f"{usd_path}::{relative_prim_path}".encode()).hexdigest()[:16]


def absolute_prim_path(stage, relative_suffix: str) -> str:
    """Join a default-prim-relative suffix to the stage default prim."""
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise RuntimeError("USD stage has no default prim")
    base = str(default_prim.GetPath())
    if not relative_suffix:
        return base
    return f"{base}/{relative_suffix.lstrip('/')}"
