# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`~isaaclab_arena.agentic_environment_generation.asset_matcher`.

Covers :func:`~asset_matcher.match_asset` resolution strategies:

  - Exact name match (no fuzzy search triggered)
  - Substring match within a tag-narrowed pool
  - Tag-pool relaxation when the preferred pool is empty or yields no match
  - Fuzzy (difflib) fallback
  - Miss behaviour
  - Embodiment bare-family expansion via the ``["embodiment", "ik"]`` tag pool
  - Empty required-tag pools
"""

from __future__ import annotations

from isaaclab_arena.agentic_environment_generation.asset_matcher import match_asset
from isaaclab_arena.tests._asset_matcher_test_helpers import FakeAsset, make_registry


def test_item_exact_name_match():
    # A query that is already a registered asset name skips all fuzzy matching.
    chosen, events = match_asset(make_registry(), "cracker_box", "item", ["object"], ["graspable"])
    assert chosen == "cracker_box"
    assert any(e.stage == "item.exact" for e in events)


def test_item_substring_match_in_tag_pool():
    chosen, events = match_asset(make_registry(), "bowl", "item", ["object"], ["bowl"])
    assert chosen == "bowl_ycb_robolab"
    assert any(e.stage == "item.preferred_tags.substring" for e in events)


def test_item_relaxes_when_tag_pool_yields_no_match():
    # ``category_tags`` points to a real pool ("fruit") but the query
    # ("cracker") doesn't substring-match any fruit asset.  The matcher
    # relaxes to the full object pool and finds ``cracker_box``.
    chosen, events = match_asset(make_registry(), "cracker", "item", ["object"], ["fruit"])
    assert chosen == "cracker_box"
    trace_stages = [e.stage for e in events]
    assert "item.preferred_tags.miss" in trace_stages
    assert any(s.startswith("item.required_tags") for s in trace_stages)


def test_item_relaxes_when_tag_pool_empty():
    # Unknown tag → empty pool → matcher relaxes to the required-tag pool.
    chosen, events = match_asset(make_registry(), "cracker", "item", ["object"], ["nonexistent"])
    assert chosen == "cracker_box"
    assert any(e.stage == "item.preferred_tags.empty_pool" for e in events)


def test_item_miss_returns_none():
    chosen, events = match_asset(make_registry(), "zzz_no_match_anywhere", "item", ["object"])
    assert chosen is None
    assert any(e.stage == "item.required_tags.miss" for e in events)


def test_embodiment_exact_match():
    chosen, events = match_asset(make_registry(), "franka_ik", "embodiment", ["embodiment"], ["ik"])
    assert chosen == "franka_ik"
    assert any(e.stage == "embodiment.exact" for e in events)


def test_embodiment_joint_pos_not_fuzzy_matched_to_ik():
    # Exact hit on the required-tag pool must run before the ik-narrowed
    # preferred pool, so joint-position control is not fuzzy-matched to franka_ik.
    chosen, events = match_asset(make_registry(), "franka_joint_pos", "embodiment", ["embodiment"], ["ik"])
    assert chosen == "franka_joint_pos"
    trace_stages = [e.stage for e in events]
    assert "embodiment.exact" in trace_stages
    assert "embodiment.preferred_tags.fuzzy" not in trace_stages


def test_embodiment_ik_default_for_bare_family():
    # Bare family names (e.g. "franka") are expanded to the IK variant by
    # narrowing embodiment candidates by the "ik" tag and picking the shortest
    # match.
    chosen, events = match_asset(make_registry(), "franka", "embodiment", ["embodiment"], ["ik"])
    assert chosen == "franka_ik"
    assert any(e.stage == "embodiment.preferred_tags.substring" for e in events)


def test_embodiment_unknown_records_miss():
    chosen, events = match_asset(make_registry(), "totally_unknown_robot", "embodiment", ["embodiment"], ["ik"])
    assert chosen is None
    assert any(e.stage == "embodiment.required_tags.miss" for e in events)


def test_background_wrong_tag_yields_empty_pool():
    assets = [
        FakeAsset(name="franka_ik", tags=["embodiment"]),
        FakeAsset(name="maple_table", tags=["object"]),  # wrong tag
    ]
    chosen, events = match_asset(make_registry(assets), "maple_table", "background", ["background"])
    assert chosen is None
    assert any(e.stage == "background.required_tags.empty_pool" for e in events)


def test_embodiment_required_tag_pool_empty():
    assets = [
        FakeAsset(name="maple_table", tags=["background"]),
        FakeAsset(name="cracker_box", tags=["object"]),
    ]
    chosen, events = match_asset(make_registry(assets), "franka_ik", "embodiment", ["embodiment"], ["ik"])
    assert chosen is None
    assert any(e.stage == "embodiment.required_tags.empty_pool" for e in events)


def test_item_required_tag_pool_empty():
    assets = [
        FakeAsset(name="maple_table", tags=["background"]),
        FakeAsset(name="franka_ik", tags=["embodiment"]),
    ]
    chosen, events = match_asset(make_registry(assets), "bowl", "item", ["object"], ["bowl"])
    assert chosen is None
    assert any(e.stage == "item.required_tags.empty_pool" for e in events)
