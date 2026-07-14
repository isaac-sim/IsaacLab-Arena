# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve graph assets to USD paths and local AABB dimensions (no Kit viewport)."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass

from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_types import AssetSpec
from isaaclab_arena.utils.usd_helpers import (
    compute_local_bounding_box_from_prim,
    compute_local_bounding_box_from_usd,
    open_stage,
)

AabbDimensionsM = tuple[float, float, float]


@dataclass(frozen=True)
class ObjectReferenceUsdTarget:
    """Parent USD plus a default-prim-relative suffix for an object_reference snapshot."""

    usd_path: str
    relative_prim_path: str


def object_reference_cache_key(usd_path: str, relative_prim_path: str) -> str:
    """Return a stable cache key for an object_reference subtree snapshot."""
    return hashlib.sha1(f"{usd_path}::{relative_prim_path}".encode()).hexdigest()[:16]


def is_resolved_prim_path(prim_path: str | None) -> bool:
    """Return True when an object_reference prim_path is ready for USD rendering."""
    if prim_path is None:
        return False
    cleaned = prim_path.strip()
    return bool(cleaned) and cleaned.lower() != "unknown"


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


def resolve_object_reference_usd_targets(spec: ArenaEnvGraphSpec) -> dict[str, ObjectReferenceUsdTarget]:
    """Map ``object_reference.id`` to parent USD and resolved prim suffix for snapshots."""
    if not spec.object_references:
        return {}

    registry = AssetRegistry()
    targets: dict[str, ObjectReferenceUsdTarget] = {}
    for ref in spec.object_references:
        if not is_resolved_prim_path(ref.prim_path):
            continue
        try:
            parent_spec = spec._asset_by_id(ref.parent_id)
        except KeyError:
            print(
                f"[asset_usd]   {ref.id}: parent '{ref.parent_id}' not found, skipping object_reference snapshot.",
                file=sys.stderr,
            )
            continue
        try:
            if not registry.is_registered(parent_spec.registry_name):
                continue
            usd_path = parent_spec.resolve_usd_path()
            if not usd_path:
                continue
            targets[ref.id] = ObjectReferenceUsdTarget(
                usd_path=usd_path,
                relative_prim_path=ref.prim_path.strip(),  # type: ignore[union-attr]
            )
        except Exception as exc:
            print(
                f"[asset_usd]   {ref.id}: object_reference lookup failed for parent '{ref.parent_id}': {exc}",
                file=sys.stderr,
            )
    return targets


def _absolute_prim_path(stage, relative_suffix: str) -> str:
    """Join a default-prim-relative suffix to the stage default prim."""
    default_prim = stage.GetDefaultPrim()
    assert default_prim and default_prim.IsValid(), "USD stage has no default prim"
    base = str(default_prim.GetPath())
    if not relative_suffix:
        return base
    return f"{base}/{relative_suffix.lstrip('/')}"


def resolve_object_reference_aabb_dimensions_m(spec: ArenaEnvGraphSpec) -> dict[str, AabbDimensionsM]:
    """Return axis-aligned bounding box sizes in meters for each resolved object_reference prim."""
    targets = resolve_object_reference_usd_targets(spec)
    if not targets:
        return {}

    by_usd: dict[str, list[tuple[str, str]]] = {}
    for ref_id, target in targets.items():
        by_usd.setdefault(target.usd_path, []).append((ref_id, target.relative_prim_path))

    dimensions: dict[str, AabbDimensionsM] = {}
    for usd_path, refs in by_usd.items():
        try:
            with open_stage(usd_path) as stage:
                for ref_id, relative_prim_path in refs:
                    try:
                        abs_path = _absolute_prim_path(stage, relative_prim_path)
                        bbox = compute_local_bounding_box_from_prim(stage, abs_path)
                        size = bbox.size[0]
                        dimensions[ref_id] = (float(size[0]), float(size[1]), float(size[2]))
                    except Exception as exc:
                        print(
                            f"[asset_usd]   {ref_id}: bbox failed for prim '{relative_prim_path}': {exc}",
                            file=sys.stderr,
                        )
        except Exception as exc:
            print(f"[asset_usd]   bbox failed to open {usd_path}: {exc}", file=sys.stderr)
    return dimensions
