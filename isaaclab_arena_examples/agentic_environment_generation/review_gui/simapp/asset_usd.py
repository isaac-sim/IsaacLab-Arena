# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve graph assets to USD paths and local AABB dimensions (no Kit viewport)."""

from __future__ import annotations

import hashlib
import sys

from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_types import AssetSpec
from isaaclab_arena.utils.usd_helpers import compute_local_bounding_box_from_usd

AabbDimensionsM = tuple[float, float, float]


def usd_cache_key(usd_path: str) -> str:
    """Return a stable short hash for caching PNGs keyed by USD path."""
    return hashlib.sha1(usd_path.encode("utf-8")).hexdigest()[:16]


def resolve_node_usd_paths(spec: ArenaEnvGraphSpec) -> dict[str, str]:
    """Map ``asset.id → usd_path`` via :class:`AssetRegistry`."""
    registry = AssetRegistry()
    paths: dict[str, str] = {}
    for asset_spec in [spec.embodiment, spec.background, *spec.objects]:
        try:
            if not registry.is_registered(asset_spec.registry_name):
                print(
                    f"[asset_usd]   {asset_spec.id}: asset '{asset_spec.registry_name}' not registered, skipping.",
                    file=sys.stderr,
                )
                continue
            cls = registry.get_asset_by_name(asset_spec.registry_name)
            usd_path = extract_usd_path(cls)
            if not usd_path:
                print(
                    f"[asset_usd]   {asset_spec.id}: '{asset_spec.registry_name}' has no usd_path, skipping.",
                    file=sys.stderr,
                )
                continue
            paths[asset_spec.id] = usd_path
        except Exception as exc:
            print(
                f"[asset_usd]   {asset_spec.id}: lookup failed for '{asset_spec.registry_name}': {exc}",
                file=sys.stderr,
            )
    return paths


def extract_usd_path(cls) -> str | None:
    """Return the asset's root USD path, or ``None`` if not extractable."""
    usd_path = getattr(cls, "usd_path", None)
    if usd_path:
        return usd_path

    try:
        instance = cls()
    except Exception:
        return None
    scene_config = getattr(instance, "scene_config", None)
    robot = getattr(scene_config, "robot", None) if scene_config is not None else None
    spawn = getattr(robot, "spawn", None) if robot is not None else None
    return getattr(spawn, "usd_path", None) if spawn is not None else None


def scale_for_asset_spec(asset_spec: AssetSpec, asset_cls) -> tuple[float, float, float]:
    """Return spawn scale for an asset spec, preferring spec params over library defaults."""
    param_scale = asset_spec.params.get("scale")
    if param_scale is not None:
        return (float(param_scale[0]), float(param_scale[1]), float(param_scale[2]))
    class_scale = getattr(asset_cls, "scale", None)
    if class_scale is not None:
        return (float(class_scale[0]), float(class_scale[1]), float(class_scale[2]))
    return (1.0, 1.0, 1.0)


def aabb_dimensions_from_usd(
    usd_path: str,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> AabbDimensionsM | None:
    """Return local axis-aligned bounding box size (x, y, z) in meters for a USD asset."""
    try:
        bbox = compute_local_bounding_box_from_usd(usd_path, scale)
        size = bbox.size[0]
        return (float(size[0]), float(size[1]), float(size[2]))
    except Exception as exc:
        print(f"[asset_usd]   bbox failed for {usd_path}: {exc}", file=sys.stderr)
        return None


def resolve_node_aabb_dimensions_m(spec: ArenaEnvGraphSpec) -> dict[str, AabbDimensionsM]:
    """Return axis-aligned bounding box sizes in meters for each asset with a resolvable USD."""
    registry = AssetRegistry()
    dimensions: dict[str, AabbDimensionsM] = {}
    for asset_spec in [spec.embodiment, spec.background, *spec.objects]:
        try:
            if not registry.is_registered(asset_spec.registry_name):
                continue
            asset_cls = registry.get_asset_by_name(asset_spec.registry_name)
            usd_path = extract_usd_path(asset_cls)
            if not usd_path:
                continue
            dims = aabb_dimensions_from_usd(usd_path, scale_for_asset_spec(asset_spec, asset_cls))
            if dims is not None:
                dimensions[asset_spec.id] = dims
        except Exception as exc:
            print(
                f"[asset_usd]   {asset_spec.id}: bbox lookup failed for '{asset_spec.registry_name}': {exc}",
                file=sys.stderr,
            )
    return dimensions
