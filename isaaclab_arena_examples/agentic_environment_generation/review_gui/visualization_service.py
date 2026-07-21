# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build review GUI asset cards with optional SimApp thumbnails."""

from __future__ import annotations

import hashlib
import json
import sys
import yaml
from dataclasses import dataclass

import streamlit as st

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_types import AssetSpec
from isaaclab_arena.utils.usd_prim_tree import UsdPrimRecord
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.client import (
    SimAppError,
    simapp_socket_from_env,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.kit_viewport import (
    panorama_cache_dir,
    thumbnail_cache_dir,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp_connector import (
    clear_simapp_client,
    ensure_simapp,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.spec_visualization.asset_cards import (
    AssetCard,
    build_asset_cards,
)


@dataclass(frozen=True)
class BackgroundPreview:
    """Resolved background USD path and prim tree when extractable from YAML."""

    usd_path: str | None
    prim_tree: list[UsdPrimRecord]


def _load_background_prim_tree(usd_path: str, *, registry_name: str) -> list[UsdPrimRecord]:
    """Load prim tree records for one background USD path."""
    from isaaclab_arena.utils.usd_prim_tree import load_usd_prim_tree

    try:
        return load_usd_prim_tree(usd_path)
    except Exception as exc:
        print(
            f"[visualization_service] background prim tree lookup failed for '{registry_name}': {exc}",
            file=sys.stderr,
        )
        return []


def _background_asset_from_yaml_text(text: str) -> AssetSpec | None:
    """Parse a ``background`` asset block from YAML text when the full spec is invalid."""
    try:
        raw = yaml.safe_load(text)
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    background = raw.get("background")
    if not isinstance(background, dict):
        return None
    try:
        return AssetSpec.model_validate(background)
    except Exception:
        return None


def _resolve_background_preview_from_asset(asset: AssetSpec) -> BackgroundPreview:
    """Resolve USD path and prim tree for one background asset spec."""
    try:
        usd_path = asset.resolve_usd_path()
    except Exception as exc:
        print(
            f"[visualization_service] background USD lookup failed for '{asset.registry_name}': {exc}",
            file=sys.stderr,
        )
        return BackgroundPreview(usd_path=None, prim_tree=[])
    prim_tree = _load_background_prim_tree(usd_path, registry_name=asset.registry_name)
    return BackgroundPreview(usd_path=usd_path, prim_tree=prim_tree)


def resolve_background_preview(text: str, *, spec: ArenaEnvGraphSpec | None = None) -> BackgroundPreview:
    """Return background USD path and prim tree when extractable from YAML text.

    Uses the validated spec when available; otherwise parses only the ``background`` block.
    """
    cached_text = st.session_state.get("_background_preview_text")
    cached_result = st.session_state.get("_background_preview_result")
    if cached_text == text and isinstance(cached_result, BackgroundPreview):
        return cached_result

    if spec is not None:
        result = _resolve_background_preview_from_asset(spec.background)
    else:
        asset = _background_asset_from_yaml_text(text)
        result = _resolve_background_preview_from_asset(asset) if asset is not None else BackgroundPreview(None, [])

    st.session_state["_background_preview_text"] = text
    st.session_state["_background_preview_result"] = result
    return result


def resolve_background_prim_tree(spec: ArenaEnvGraphSpec) -> list[UsdPrimRecord]:
    """Return the background USD prim tree records, empty when unavailable."""
    return _resolve_background_preview_from_asset(spec.background).prim_tree


def _spec_render_key(spec: ArenaEnvGraphSpec, *, background_panorama: bool) -> str:
    payload = json.dumps({"spec": spec.to_dict(), "background_panorama": background_panorama}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cached_asset_cards(spec_key: str) -> list[AssetCard] | None:
    cache = st.session_state.get("_asset_cards_cache")
    if isinstance(cache, dict) and cache.get("key") == spec_key:
        asset_cards = cache.get("asset_cards")
        if isinstance(asset_cards, list):
            return asset_cards
    return None


def _store_asset_cards(spec_key: str, asset_cards: list[AssetCard]) -> None:
    st.session_state["_asset_cards_cache"] = {
        "key": spec_key,
        "asset_cards": asset_cards,
    }


def clear_snapshot_render_caches() -> int:
    """Delete cached review GUI snapshot PNGs and return how many files were removed."""
    paths = [path for cache_dir in (thumbnail_cache_dir(), panorama_cache_dir()) for path in cache_dir.glob("*.png")]
    for path in paths:
        path.unlink()
    return len(paths)


def clear_asset_cards_cache() -> None:
    """Drop the in-memory asset-card cache for the current Streamlit session."""
    st.session_state.pop("_asset_cards_cache", None)


def _warn_simapp_unavailable_once() -> None:
    if st.session_state.get("_simapp_unavailable_warned"):
        return
    st.session_state["_simapp_unavailable_warned"] = True
    st.warning(
        "Isaac Sim is unavailable — showing placeholder thumbnails. "
        "Check the gui_runner terminal for SimApp boot errors.",
        icon="⚠️",
    )


def _show_simapp_render_error_once(exc: SimAppError) -> None:
    if st.session_state.get("_simapp_render_error_shown"):
        return
    st.session_state["_simapp_render_error_shown"] = True
    st.error(
        f"SimApp render failed; showing placeholder thumbnails.\n\n```\n{exc}\n```",
        icon="🛑",
    )


def build_asset_cards_with_thumbnails(
    spec: ArenaEnvGraphSpec,
    *,
    background_panorama: bool = False,
) -> list[AssetCard]:
    """Build per-node asset cards with optional SimApp thumbnails.

    Args:
        spec: Environment graph spec to visualize.
        background_panorama: When True, render the background as a 360° panorama.
    """
    spec_key = _spec_render_key(spec, background_panorama=background_panorama)
    cached = _cached_asset_cards(spec_key)
    if cached is not None:
        return cached

    panorama_node_ids: set[str] = set()
    if background_panorama and spec.background is not None:
        panorama_node_ids.add(spec.background.id)

    thumbnails: dict[str, bytes] = {}
    aabb_dimensions_m: dict[str, tuple[float, float, float]] = {}

    simapp_expected = simapp_socket_from_env() is not None
    client = ensure_simapp() if simapp_expected else None
    if client is None:
        if simapp_expected:
            _warn_simapp_unavailable_once()
    else:
        try:
            thumbnails, aabb_dimensions_m = client.render_spec(spec, background_panorama=background_panorama)
        except SimAppError as exc:
            _show_simapp_render_error_once(exc)
            thumbnails, aabb_dimensions_m = {}, {}
        finally:
            # Release the socket so the sequential SimApp server can accept other tabs.
            clear_simapp_client()

    asset_cards = build_asset_cards(
        spec,
        thumbnails or None,
        aabb_dimensions_m or None,
        panorama_node_ids or None,
    )
    _store_asset_cards(spec_key, asset_cards)
    return asset_cards
