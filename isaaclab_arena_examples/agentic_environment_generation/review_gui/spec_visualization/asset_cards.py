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
    is_panorama: bool = False


def object_set_member_key(object_set_id: str, registry_name: str) -> str:
    """Return the thumbnail and AABB lookup key for one member of an object set."""
    return f"{object_set_id}::{registry_name}"


def build_asset_cards(
    spec: ArenaEnvGraphSpec,
    thumbnails: dict[str, bytes] | None = None,
    aabb_dimensions_m: dict[str, tuple[float, float, float]] | None = None,
    panorama_node_ids: set[str] | None = None,
) -> list[AssetCard]:
    """Build one AssetCard per node (background, object references, objects, object-set members) for native rendering."""
    thumbnails = thumbnails or {}
    aabb_dimensions_m = aabb_dimensions_m or {}
    panorama_node_ids = panorama_node_ids or set()

    entries: list[tuple[str, AssetSpec | ObjectReferenceSpec, str]] = []
    entries.append(("background", spec.background, spec.background.id))
    entries.extend(("object_reference", ref, ref.id) for ref in (spec.object_references or []))
    entries.extend(("object", obj, obj.id) for obj in spec.objects)
    # carry one card per object set member
    entries.extend(
        (
            "object_set",
            AssetSpec(id=object_set.id, registry_name=registry_name),
            object_set_member_key(object_set.id, registry_name),
        )
        for object_set in (spec.object_sets or [])
        for registry_name in object_set.members
    )

    return [
        AssetCard(
            spec=asset,
            role=role,
            thumbnail_bytes=thumbnails.get(lookup_key),
            aabb_dimensions_m=aabb_dimensions_m.get(lookup_key),
            is_panorama=asset.id in panorama_node_ids,
        )
        for role, asset, lookup_key in entries
    ]
