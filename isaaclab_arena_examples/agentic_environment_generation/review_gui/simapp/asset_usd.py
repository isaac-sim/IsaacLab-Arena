# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve instantiated graph assets to USD paths and local AABB dimensions (no Kit viewport)."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from typing import Any

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_reference import ObjectReference
from isaaclab_arena.utils.usd_helpers import open_stage

AabbDimensionsM = tuple[float, float, float]


@dataclass(frozen=True)
class ObjectReferenceUsdTarget:
    """Parent USD plus a default-prim-relative suffix for an object_reference snapshot."""

    usd_path: str
    relative_prim_path: str


def object_reference_cache_key(usd_path: str, relative_prim_path: str) -> str:
    """Return a stable cache key for an object_reference subtree snapshot."""
    return hashlib.sha1(f"{usd_path}::{relative_prim_path}".encode()).hexdigest()[:16]


def aabb_dimensions_from_asset(asset: Any) -> AabbDimensionsM | None:
    """Return local axis-aligned bounding box size (x, y, z) in meters for one live asset."""
    if not isinstance(asset, (Object, ObjectReference)):
        return None
    try:
        bbox = asset.get_bounding_box()
        size = bbox.size[0]
        return (float(size[0]), float(size[1]), float(size[2]))
    except Exception as exc:
        name = getattr(asset, "name", "?")
        print(f"[asset_usd]   {name}: bbox failed: {exc}", file=sys.stderr)
        return None


def resolve_aabb_dimensions_m(assets_by_node_id: dict[str, Any]) -> dict[str, AabbDimensionsM]:
    """Return axis-aligned bounding box sizes in meters for each snapshot asset (objects and references)."""
    dimensions: dict[str, AabbDimensionsM] = {}
    for node_id, asset in assets_by_node_id.items():
        dims = aabb_dimensions_from_asset(asset)
        if dims is not None:
            dimensions[node_id] = dims
    return dimensions


def resolve_object_reference_usd_targets(assets_by_node_id: dict[str, Any]) -> dict[str, ObjectReferenceUsdTarget]:
    """Map ``object_reference.id`` to parent USD and resolved prim suffix for snapshots."""
    targets: dict[str, ObjectReferenceUsdTarget] = {}
    for node_id, asset in assets_by_node_id.items():
        if not isinstance(asset, ObjectReference):
            continue
        parent = asset.parent_asset
        usd_path = getattr(parent, "usd_path", None)
        if not usd_path:
            continue
        try:
            with open_stage(usd_path) as stage:
                abs_path = asset.isaaclab_prim_path_to_original_prim_path(asset.prim_path, parent, stage)
                default_prim_path = str(stage.GetDefaultPrim().GetPath())
                relative_prim_path = abs_path.removeprefix(default_prim_path).lstrip("/")
                targets[node_id] = ObjectReferenceUsdTarget(
                    usd_path=usd_path,
                    relative_prim_path=relative_prim_path,
                )
        except Exception as exc:
            print(f"[asset_usd]   {node_id}: object_reference target failed: {exc}", file=sys.stderr)
    return targets


def absolute_prim_path(stage, relative_suffix: str) -> str:
    """Join a default-prim-relative suffix to the stage default prim."""
    default_prim = stage.GetDefaultPrim()
    assert default_prim and default_prim.IsValid(), "USD stage has no default prim"
    base = str(default_prim.GetPath())
    if not relative_suffix:
        return base
    return f"{base}/{relative_suffix.lstrip('/')}"
