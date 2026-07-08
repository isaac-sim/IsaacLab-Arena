# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for pass-2 object_reference prim_path resolution."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from isaaclab_arena.agentic_environment_generation.object_reference_prim_resolver import (
    ObjectReferencePrimResolver,
    merge_resolved_object_references,
)
from isaaclab_arena.agentic_environment_generation.query_backend import QueryBackend
from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environments.arena_env_graph_types import (
    ResolveObjectReferencePrimPathsResult,
    isaaclab_prim_path_for_background_ref,
)
from isaaclab_arena.utils.usd_prim_tree import UsdPrimRecord

_KITCHEN_PASS1: dict = {
    "env_name": "llm_gen_kitchen_pick_and_open",
    "embodiment": {"id": "droid_abs_joint_pos", "registry_name": "droid_abs_joint_pos"},
    "background": {
        "id": "lightwheel_robocasa_kitchen",
        "registry_name": "lightwheel_robocasa_kitchen",
        "params": {"layout_id": 1, "style_id": 1},
    },
    "objects": [
        {"id": "avocado", "registry_name": "avocado01_fruits_veggies_robolab"},
        {"id": "plate", "registry_name": "plate_large_vomp_robolab"},
    ],
    "object_references": [
        {
            "id": "counter_top",
            "parent_id": "lightwheel_robocasa_kitchen",
            "prim_path": "unknown",
            "object_type": "base",
        },
        {
            "id": "fridge",
            "parent_id": "lightwheel_robocasa_kitchen",
            "prim_path": "unknown",
            "object_type": "articulation",
        },
    ],
    "relations": [
        {"kind": "is_anchor", "subject": "counter_top"},
        {"kind": "on", "subject": "avocado", "reference": "counter_top"},
        {"kind": "on", "subject": "plate", "reference": "counter_top"},
    ],
    "tasks": [
        {
            "kind": "PickAndPlaceTask",
            "params": {
                "pick_up_object": "avocado",
                "destination_location": "plate",
                "background_scene": "lightwheel_robocasa_kitchen",
            },
            "description": "pick avocado and place on plate",
        },
        {
            "kind": "OpenDoorTask",
            "params": {"openable_object": "fridge"},
            "description": "open the fridge door",
        },
    ],
}

_KITCHEN_PASS1_RESOLVED_PRIM: dict = {
    **_KITCHEN_PASS1,
    "object_references": [
        {
            "id": "counter_top",
            "parent_id": "lightwheel_robocasa_kitchen",
            "prim_path": "counter_right_main_group/top_geometry",
            "object_type": "base",
        },
        {
            "id": "fridge",
            "parent_id": "lightwheel_robocasa_kitchen",
            "prim_path": "fridge_main_group",
            "object_type": "articulation",
        },
    ],
}

_PRIM_TREE = [
    UsdPrimRecord("counter_right_main_group/top_geometry", "base"),
    UsdPrimRecord("fridge_main_group", "articulation", ("fridge_door_joint",)),
]

_RESOLVE_RESPONSE = {
    "object_references": [
        {
            "id": "counter_top",
            "parent_id": "lightwheel_robocasa_kitchen",
            "prim_path": "counter_right_main_group/top_geometry",
            "object_type": "base",
        },
        {
            "id": "fridge",
            "parent_id": "lightwheel_robocasa_kitchen",
            "prim_path": "fridge_main_group",
            "object_type": "articulation",
            "params": {"openable_joint_name": "fridge_door_joint"},
        },
    ],
}


def _chat_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].message.reasoning_content = None
    return resp


def test_isaaclab_prim_path_expands_relative_suffix():
    path = isaaclab_prim_path_for_background_ref("lightwheel_robocasa_kitchen", "fridge_main_group")
    assert path == "{ENV_REGEX_NS}/lightwheel_robocasa_kitchen/fridge_main_group"


def test_isaaclab_prim_path_preserves_full_path():
    full = "{ENV_REGEX_NS}/lightwheel_robocasa_kitchen/table"
    assert isaaclab_prim_path_for_background_ref("lightwheel_robocasa_kitchen", full) == full


def test_merge_resolved_object_references():
    spec = ArenaEnvGraphSpec.model_validate(_KITCHEN_PASS1)
    resolved = ResolveObjectReferencePrimPathsResult.model_validate(_RESOLVE_RESPONSE).object_references
    merged = merge_resolved_object_references(spec, resolved)
    counter = next(ref for ref in merged.object_references if ref.id == "counter_top")
    fridge = next(ref for ref in merged.object_references if ref.id == "fridge")
    assert counter.prim_path == "counter_right_main_group/top_geometry"
    assert fridge.prim_path == "fridge_main_group"
    assert fridge.params["openable_joint_name"] == "fridge_door_joint"
    merged.validate_resolved()


@patch("isaaclab_arena.agentic_environment_generation.object_reference_prim_resolver.load_usd_prim_tree")
@patch("isaaclab_arena.agentic_environment_generation.object_reference_prim_resolver.resolve_asset_usd_path")
def test_object_reference_prim_resolver_infer_always_calls_llm(mock_resolve_usd, mock_load_tree):
    mock_resolve_usd.return_value = "/tmp/scene.usd"
    mock_load_tree.return_value = _PRIM_TREE
    client = MagicMock()
    client.chat.completions.create.return_value = _chat_response(json.dumps(_RESOLVE_RESPONSE))
    resolver = ObjectReferencePrimResolver(QueryBackend(client, "test-model"))
    spec = ArenaEnvGraphSpec.model_validate(_KITCHEN_PASS1_RESOLVED_PRIM)
    traces: list[str] = []
    merged = resolver.infer(spec, traces)
    client.chat.completions.create.assert_called_once()
    counter = next(ref for ref in merged.object_references if ref.id == "counter_top")
    fridge = next(ref for ref in merged.object_references if ref.id == "fridge")
    assert counter.prim_path == "counter_right_main_group/top_geometry"
    assert fridge.params["openable_joint_name"] == "fridge_door_joint"
    merged.validate_resolved()
