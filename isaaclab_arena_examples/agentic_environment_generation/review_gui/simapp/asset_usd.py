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
from isaaclab_arena.environment_spec.arena_env_graph_types import ObjectReferenceSpec

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


def resolve_object_reference_usd_targets(
    object_references: list[ObjectReferenceSpec] | None,
    parent_usd_paths: dict[str, str],
) -> dict[str, ObjectReferenceUsdTarget]:
    """Map each resolved object_reference id to its parent USD and default-prim-relative prim suffix."""
    targets: dict[str, ObjectReferenceUsdTarget] = {}
    for ref in object_references or []:
        if ref.prim_path is None:
            continue
        usd_path = parent_usd_paths.get(ref.parent_id)
        if not usd_path:
            continue
        targets[ref.id] = ObjectReferenceUsdTarget(
            usd_path=usd_path,
            relative_prim_path=ref.prim_path.lstrip("/"),
        )
    return targets


def absolute_prim_path(stage, relative_suffix: str) -> str:
    """Join a default-prim-relative suffix to the stage default prim."""
    default_prim = stage.GetDefaultPrim()
    assert default_prim and default_prim.IsValid(), "USD stage has no default prim"
    base = str(default_prim.GetPath())
    if not relative_suffix:
        return base
    return f"{base}/{relative_suffix.lstrip('/')}"
