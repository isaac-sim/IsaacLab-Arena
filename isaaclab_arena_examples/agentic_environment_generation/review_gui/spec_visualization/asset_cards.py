# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import yaml
from dataclasses import dataclass

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_types import AssetSpec


@dataclass(frozen=True)
class AssetCard:
    """Structured per-asset data for rendering a snapshot card as native Streamlit widgets."""

    node_id: str
    """Graph node id shown as the card heading."""
    role: str
    """Node role: ``background`` or ``object``."""
    label: str
    """Registry name for the asset."""
    yaml_text: str
    """Pretty-printed spec YAML for the node."""
    png_bytes: bytes | None = None
    """USD snapshot PNG, or ``None`` when no capture is available."""
    aabb_dimensions_m: tuple[float, float, float] | None = None
    """Axis-aligned bounding box size in metres, when known."""


def build_asset_cards(
    spec: ArenaEnvGraphSpec,
    thumbnails: dict[str, bytes] | None = None,
    aabb_dimensions_m: dict[str, tuple[float, float, float]] | None = None,
) -> list[AssetCard]:
    """Build one AssetCard per node (background, objects) for native rendering."""
    thumbnails = thumbnails or {}
    aabb_dimensions_m = aabb_dimensions_m or {}
    entries: list[tuple[str, AssetSpec]] = []
    if spec.background is not None:
        entries.append(("background", spec.background))
    entries.extend(("object", obj) for obj in spec.objects)

    cards: list[AssetCard] = []
    for role, asset in entries:
        node_yaml = yaml.safe_dump(asset.model_dump(mode="json", exclude_none=True), sort_keys=False).rstrip()
        cards.append(
            AssetCard(
                node_id=asset.id,
                role=role,
                label=asset.registry_name,
                yaml_text=node_yaml,
                png_bytes=thumbnails.get(asset.id),
                aabb_dimensions_m=aabb_dimensions_m.get(asset.id),
            )
        )
    return cards
