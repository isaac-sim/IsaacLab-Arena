# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_types import AssetSpec, ObjectReferenceSpec


@dataclass(frozen=True)
class AssetCard:
    """Structured per-asset data for rendering a snapshot card as native Streamlit widgets."""

    spec: AssetSpec | ObjectReferenceSpec
    role: str
    thumbnail_bytes: bytes | None = None
    aabb_dimensions_m: tuple[float, float, float] | None = None


def build_asset_cards(
    spec: ArenaEnvGraphSpec,
    thumbnails: dict[str, bytes] | None = None,
    aabb_dimensions_m: dict[str, tuple[float, float, float]] | None = None,
) -> list[AssetCard]:
    """Build one AssetCard per node (background, object references, objects) for native rendering."""
    thumbnails = thumbnails or {}
    aabb_dimensions_m = aabb_dimensions_m or {}
    entries: list[tuple[str, AssetSpec | ObjectReferenceSpec]] = []
    entries.append(("background", spec.background))
    entries.extend(("object_reference", ref) for ref in (spec.object_references or []))
    entries.extend(("object", obj) for obj in spec.objects)

    return [
        AssetCard(
            spec=asset,
            role=role,
            thumbnail_bytes=thumbnails.get(asset.id),
            aabb_dimensions_m=aabb_dimensions_m.get(asset.id),
        )
        for role, asset in entries
    ]
