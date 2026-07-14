# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve graph assets to USD paths and local AABB dimensions (no Kit viewport)."""

from __future__ import annotations

import sys

from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_types import AssetSpec
from isaaclab_arena.utils.usd_helpers import compute_local_bounding_box_from_usd

AabbDimensionsM = tuple[float, float, float]


def _snapshot_asset_specs(spec: ArenaEnvGraphSpec) -> list[AssetSpec]:
    """Assets included in review GUI USD thumbnails (excludes embodiment)."""
    return [spec.background, *spec.objects]


def resolve_node_usd_paths(spec: ArenaEnvGraphSpec) -> dict[str, str]:
    """Map ``asset.id → usd_path`` via :class:`AssetRegistry`."""
    registry = AssetRegistry()
    paths: dict[str, str] = {}
    for asset_spec in _snapshot_asset_specs(spec):
        try:
            if not registry.is_registered(asset_spec.registry_name):
                print(
                    f"[asset_usd]   {asset_spec.id}: asset '{asset_spec.registry_name}' not registered, skipping.",
                    file=sys.stderr,
                )
                continue
            paths[asset_spec.id] = asset_spec.resolve_usd_path()
        except Exception as exc:
            print(
                f"[asset_usd]   {asset_spec.id}: lookup failed for '{asset_spec.registry_name}': {exc}",
                file=sys.stderr,
            )
    return paths


def viewer_cfg_for_asset_spec(asset_spec: AssetSpec):
    """Return a custom :class:`ViewerCfg` for snapshot rendering, or ``None`` to auto-frame."""
    from isaaclab.envs.common import ViewerCfg

    from isaaclab_arena.assets.background import Background

    registry = AssetRegistry()
    if not registry.is_registered(asset_spec.registry_name):
        return None
    asset_cls = registry.get_asset_by_name(asset_spec.registry_name)
    if not issubclass(asset_cls, Background):
        return None
    try:
        instance = asset_cls(**asset_spec.params)
    except Exception as exc:
        print(
            f"[asset_usd]   {asset_spec.id}: viewer cfg lookup failed for '{asset_spec.registry_name}': {exc}",
            file=sys.stderr,
        )
        return None
    viewer_cfg = instance.get_viewer_cfg()
    default = ViewerCfg()
    if viewer_cfg.eye == default.eye and viewer_cfg.lookat == default.lookat:
        return None
    return viewer_cfg


def resolve_background_viewer_cfgs(spec: ArenaEnvGraphSpec) -> dict[str, object]:
    """Map background ``asset.id`` to its :class:`ViewerCfg`` for snapshot rendering."""
    if spec.background is None:
        return {}
    viewer_cfg = viewer_cfg_for_asset_spec(spec.background)
    if viewer_cfg is None:
        return {}
    return {spec.background.id: viewer_cfg}


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
    for asset_spec in _snapshot_asset_specs(spec):
        try:
            if not registry.is_registered(asset_spec.registry_name):
                continue
            asset_cls = registry.get_asset_by_name(asset_spec.registry_name)
            usd_path = asset_spec.resolve_usd_path()
            dims = aabb_dimensions_from_usd(usd_path, scale_for_asset_spec(asset_spec, asset_cls))
            if dims is not None:
                dimensions[asset_spec.id] = dims
        except Exception as exc:
            print(
                f"[asset_usd]   {asset_spec.id}: bbox lookup failed for '{asset_spec.registry_name}': {exc}",
                file=sys.stderr,
            )
    return dimensions
