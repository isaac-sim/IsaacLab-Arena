# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`~isaaclab_arena.agentic_environment_generation.asset_resolver.AssetResolver`.

Covers the catalog-binding strategies exercised by
:meth:`~AssetResolver.resolve_item`, :meth:`~AssetResolver.resolve_name`,
and :meth:`~AssetResolver.resolve_embodiment`:

- Exact name match (no fuzzy search triggered)
- Substring match within a tag-narrowed pool
- Tag-pool relaxation when the tag pool is empty or yields no close match
- Fuzzy (difflib) fallback
- Miss / omission behaviour
- Embodiment bare-family expansion via the ``["embodiment", "ik"]`` tag pool
- Background tag enforcement
"""

from __future__ import annotations

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphNodeType

from ._resolver_test_helpers import FakeAsset, make_resolver, make_scene

# ---------------------------------------------------------------------------
# Item resolution strategies
# ---------------------------------------------------------------------------


def test_item_exact_name_match():
    # A query that is already a registered asset name skips all fuzzy matching.
    from isaaclab_arena.agentic_environment_generation.environment_intent_spec import Item

    items = [Item(query="cracker_box", category_tags=["graspable"])]
    resolver = make_resolver()
    spec = resolver.resolve(make_scene(items=items))
    assert spec.nodes_by_id["cracker_box"].name == "cracker_box"
    assert any(e.stage == "item.exact" for e in resolver.trace)


def test_item_substring_match_in_tag_pool():
    from isaaclab_arena.agentic_environment_generation.environment_intent_spec import Item

    items = [Item(query="bowl", category_tags=["bowl"])]
    resolver = make_resolver()
    spec = resolver.resolve(make_scene(items=items))
    assert spec.nodes_by_id["bowl"].name == "bowl_ycb_robolab"
    assert any(e.stage == "item.in_tags.substring" for e in resolver.trace)


def test_item_relaxes_when_tag_pool_yields_no_match():
    # ``category_tags`` points to a real pool ("fruit") but the query
    # ("cracker") doesn't substring-match any fruit asset.  The resolver
    # relaxes to the full object pool and finds ``cracker_box``.
    from isaaclab_arena.agentic_environment_generation.environment_intent_spec import Item

    items = [Item(query="cracker", category_tags=["fruit"])]
    resolver = make_resolver()
    spec = resolver.resolve(make_scene(items=items))
    assert spec.nodes_by_id["cracker"].name == "cracker_box"
    trace_stages = [e.stage for e in resolver.trace]
    assert "item.no_match_in_tags" in trace_stages
    assert any(s.startswith("item.relaxed") for s in trace_stages)


def test_item_relaxes_when_tag_pool_empty():
    # Unknown tag → empty pool → resolver short-circuits and relaxes immediately.
    from isaaclab_arena.agentic_environment_generation.environment_intent_spec import Item

    items = [Item(query="cracker", category_tags=["nonexistent"])]
    resolver = make_resolver()
    spec = resolver.resolve(make_scene(items=items))
    assert spec.nodes_by_id["cracker"].name == "cracker_box"
    assert any(e.stage == "item.tag_pool_empty" for e in resolver.trace)


def test_item_miss_omits_node():
    # A query with no substring or fuzzy candidate is silently dropped;
    # the resolver records a trace event but never raises.
    from isaaclab_arena.agentic_environment_generation.environment_intent_spec import Item

    items = [Item(query="zzz_no_match_anywhere", category_tags=["object"])]
    resolver = make_resolver()
    spec = resolver.resolve(make_scene(items=items))
    assert "zzz_no_match_anywhere" not in spec.nodes_by_id
    assert any(e.stage == "item.miss" for e in resolver.trace)


def test_item_instance_name_overrides_query_for_node_id():
    # ``instance_name`` becomes the graph node id; ``name`` still reflects the
    # resolved asset, allowing the same asset to appear under different ids.
    from isaaclab_arena.agentic_environment_generation.environment_intent_spec import Item

    items = [Item(query="bowl", category_tags=["bowl"], instance_name="serving_bowl")]
    spec = make_resolver().resolve(make_scene(items=items))
    assert "serving_bowl" in spec.nodes_by_id
    assert "bowl" not in spec.nodes_by_id
    assert spec.nodes_by_id["serving_bowl"].name == "bowl_ycb_robolab"


# ---------------------------------------------------------------------------
# Embodiment resolution
# ---------------------------------------------------------------------------


def test_embodiment_exact_match():
    # Node ID matches the original query so task params can reference it by the agent-emitted name.
    spec = make_resolver().resolve(make_scene(embodiment="franka_joint_pos"))
    node = spec.nodes_by_id["franka_joint_pos"]
    assert node.type == ArenaEnvGraphNodeType.EMBODIMENT
    assert node.name == "franka_joint_pos"


def test_embodiment_ik_default_for_bare_family():
    # Bare family names (e.g. "franka") are expanded to the IK variant by
    # querying the ["embodiment", "ik"] tag pool and picking the shortest match.
    # The node ID stays as the original query "franka" so downstream task params
    # that reference the robot by its original name resolve correctly.
    resolver = make_resolver()
    spec = resolver.resolve(make_scene(embodiment="franka"))
    node = spec.nodes_by_id["franka"]  # ID = original query, not "franka_ik"
    assert node.type == ArenaEnvGraphNodeType.EMBODIMENT
    assert node.name == "franka_ik"  # name = resolved asset
    assert any(e.stage == "embodiment.ik_family" for e in resolver.trace)


def test_embodiment_unknown_records_miss_and_omits_node():
    # Completely unknown names emit an "embodiment.miss" trace event and
    # produce no embodiment node (no silent fallback).
    resolver = make_resolver()
    spec = resolver.resolve(make_scene(embodiment="totally_unknown_robot"))
    assert not any(n.type == ArenaEnvGraphNodeType.EMBODIMENT for n in spec.nodes)
    assert any(e.stage == "embodiment.miss" for e in resolver.trace)


# ---------------------------------------------------------------------------
# Background resolution
# ---------------------------------------------------------------------------


def test_background_with_wrong_tag_omitted():
    # An asset registered under the right name but lacking the "background" tag
    # is rejected; the background node is absent from the resulting spec.
    assets = [
        FakeAsset(name="franka_ik", tags=["embodiment"]),
        FakeAsset(name="maple_table", tags=["object"]),  # wrong tag
    ]
    resolver = make_resolver(assets)
    spec = resolver.resolve(make_scene(background="maple_table"))
    assert "maple_table" not in spec.nodes_by_id
    assert any(e.stage == "name.wrong_tag" for e in resolver.trace)
