# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import yaml
from dataclasses import dataclass, field

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_types import AssetSpec, ObjectReferenceSpec


@dataclass(frozen=True)
class AssetCard:
    """Structured per-asset data for rendering a snapshot card as native Streamlit widgets."""

    node_id: str
    """Graph node id shown as the card heading."""
    role: str
    """Node role: ``background``, ``object``, or ``object_reference``."""
    label: str
    """Registry name for assets, or the node id for object references."""
    yaml_text: str
    """Pretty-printed spec YAML for the node."""
    png_bytes: bytes | None = None
    """USD snapshot PNG, or ``None`` when no capture is available."""
    aabb_dimensions_m: tuple[float, float, float] | None = None
    """Axis-aligned bounding box size in metres, when known."""
    is_panorama: bool = False
    """Whether the snapshot is a 360 panorama capture (rendered full width)."""
    is_object_reference: bool = False
    """Whether the node is an object_reference (collision-mesh preview)."""
    prim_unresolved: bool = False
    """Object reference whose prim_path is unresolved, so no snapshot exists."""


@dataclass(frozen=True)
class DashboardRender:
    """Cached dashboard output: per-node asset cards rendered as native Streamlit widgets."""

    asset_cards: list[AssetCard] = field(default_factory=list)
    """Per-node snapshot cards rendered natively so they get Streamlit's fullscreen zoom."""


def build_asset_cards(
    spec: ArenaEnvGraphSpec,
    thumbnails: dict[str, bytes] | None = None,
    aabb_dimensions_m: dict[str, tuple[float, float, float]] | None = None,
    panorama_node_ids: set[str] | None = None,
) -> list[AssetCard]:
    """Build one AssetCard per node (background, object references, objects) for native rendering."""
    thumbnails = thumbnails or {}
    aabb_dimensions_m = aabb_dimensions_m or {}
    panorama_node_ids = panorama_node_ids or set()
    entries: list[tuple[str, AssetSpec | ObjectReferenceSpec]] = []
    if spec.background is not None:
        entries.append(("background", spec.background))
    entries.extend(("object_reference", ref) for ref in (spec.object_references or []))
    entries.extend(("object", obj) for obj in spec.objects)

    cards: list[AssetCard] = []
    for role, asset in entries:
        png_bytes = thumbnails.get(asset.id)
        node_yaml = yaml.safe_dump(asset.model_dump(mode="json", exclude_none=True), sort_keys=False).rstrip()
        if role == "object_reference":
            assert isinstance(asset, ObjectReferenceSpec)
            cards.append(
                AssetCard(
                    node_id=asset.id,
                    role=role,
                    label=asset.id,
                    yaml_text=node_yaml,
                    png_bytes=png_bytes,
                    is_object_reference=True,
                    prim_unresolved=png_bytes is None,
                )
            )
        else:
            assert isinstance(asset, AssetSpec)
            cards.append(
                AssetCard(
                    node_id=asset.id,
                    role=role,
                    label=asset.registry_name,
                    yaml_text=node_yaml,
                    png_bytes=png_bytes,
                    aabb_dimensions_m=aabb_dimensions_m.get(asset.id),
                    is_panorama=asset.id in panorama_node_ids,
                )
            )
    return cards
