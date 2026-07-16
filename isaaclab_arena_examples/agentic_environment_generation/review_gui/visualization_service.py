# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build review GUI asset cards with optional SimApp thumbnails."""

from __future__ import annotations

import hashlib
import json

import streamlit as st

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.client import (
    SimAppError,
    simapp_socket_from_env,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.kit_viewport import thumbnail_cache_dir
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp_connector import (
    clear_simapp_client,
    ensure_simapp,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.spec_visualization.asset_cards import (
    AssetCard,
    build_asset_cards,
)


def _spec_render_key(spec: ArenaEnvGraphSpec) -> str:
    payload = json.dumps({"spec": spec.to_dict()}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cached_asset_cards(spec_key: str) -> list[AssetCard] | None:
    cache = st.session_state.get("_asset_cards_cache")
    if isinstance(cache, dict) and cache.get("key") == spec_key:
        asset_cards = cache.get("asset_cards")
        if isinstance(asset_cards, list):
            return asset_cards
    return None


def _store_asset_cards(spec_key: str, asset_cards: list[AssetCard]) -> None:
    st.session_state["_asset_cards_cache"] = {"key": spec_key, "asset_cards": asset_cards}


def clear_snapshot_render_caches() -> int:
    """Delete cached review GUI snapshot PNGs and return how many files were removed."""
    paths = list(thumbnail_cache_dir().glob("*.png"))
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


def build_asset_cards_with_thumbnails(spec: ArenaEnvGraphSpec) -> list[AssetCard]:
    """Build per-node asset cards, asking the SimApp server for live USD thumbnails when available."""
    spec_key = _spec_render_key(spec)
    cached = _cached_asset_cards(spec_key)
    if cached is not None:
        return cached

    thumbnails: dict[str, bytes] = {}
    aabb_dimensions_m: dict[str, tuple[float, float, float]] = {}

    simapp_expected = simapp_socket_from_env() is not None
    client = ensure_simapp() if simapp_expected else None
    if client is None:
        if simapp_expected:
            _warn_simapp_unavailable_once()
    else:
        try:
            thumbnails, aabb_dimensions_m = client.render_spec(spec)
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
    )
    _store_asset_cards(spec_key, asset_cards)
    return asset_cards
