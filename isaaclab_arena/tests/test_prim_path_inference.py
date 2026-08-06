# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for object_reference prim_path inference."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from isaaclab_arena.agentic_environment_generation.prim_path_inference import PrimPathInference, _prim_tree_catalog
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.tests.utils.agentic_environment_generation import (
    chat_response,
    inference_backend,
    kitchen_pass1_dict,
    kitchen_prim_tree,
    kitchen_resolve_response,
)
from isaaclab_arena.utils.usd_prim_tree import UsdPrimRecord


def test_prim_tree_catalog_nested_format():
    tree = [
        UsdPrimRecord("cab_1_main_group", ObjectType.ARTICULATION, ("right_door_joint",)),
        UsdPrimRecord("cab_1_main_group/corpus", ObjectType.RIGID),
        UsdPrimRecord("cab_1_main_group/corpus/back", ObjectType.BASE),
    ]
    assert (
        _prim_tree_catalog(tree)
        == "BACKGROUND PRIM TREE:\ncab_1_main_group (articulation right_door_joint)\n"
        "  cab_1_main_group/corpus (rigid)\n"
        "    cab_1_main_group/corpus/back (base)"
    )


def test_prim_tree_catalog_shows_full_path_for_nested_floor():
    tree = [
        UsdPrimRecord("Kitchen", ObjectType.BASE),
        UsdPrimRecord("Kitchen/Floor", ObjectType.BASE),
        UsdPrimRecord("Counter_north_0", ObjectType.BASE),
    ]
    catalog = _prim_tree_catalog(tree)
    assert "Kitchen/Floor (base)" in catalog
    assert "\n  Floor (base)" not in catalog


@patch("isaaclab_arena.utils.usd_prim_tree.load_usd_prim_tree")
@patch("isaaclab_arena.environment_spec.arena_env_graph_types.AssetSpec.resolve_usd_path")
def test_prim_path_inference_infer_merges_llm_output(mock_resolve_usd, mock_load_tree, stub_openai):
    mock_resolve_usd.return_value = "/tmp/scene.usd"
    mock_load_tree.return_value = kitchen_prim_tree()
    _, client = stub_openai
    backend = inference_backend(stub_openai)
    client.chat.completions.create.return_value = chat_response(content=json.dumps(kitchen_resolve_response()))
    inference = PrimPathInference(backend)
    spec = ArenaEnvGraphSpec.model_validate(kitchen_pass1_dict())
    merged = inference.infer(spec, [])
    client.chat.completions.create.assert_called_once()
    counter = next(ref for ref in merged.object_references if ref.id == "counter_top")
    fridge = next(ref for ref in merged.object_references if ref.id == "fridge")
    assert counter.prim_path == "counter_right_main_group/top_geometry"
    assert fridge.prim_path == "fridge_main_group"
    assert fridge.params["openable_joint_name"] == "fridge_door_joint"


@patch("isaaclab_arena.utils.usd_prim_tree.load_usd_prim_tree")
@patch("isaaclab_arena.environment_spec.arena_env_graph_types.AssetSpec.resolve_usd_path")
def test_prim_path_inference_strips_leading_slash(mock_resolve_usd, mock_load_tree, stub_openai):
    mock_resolve_usd.return_value = "/tmp/scene.usd"
    mock_load_tree.return_value = kitchen_prim_tree()
    response = kitchen_resolve_response()
    for ref in response["object_references"]:
        ref["prim_path"] = "/" + ref["prim_path"]
    _, client = stub_openai
    backend = inference_backend(stub_openai)
    client.chat.completions.create.return_value = chat_response(content=json.dumps(response))
    inference = PrimPathInference(backend)
    spec = ArenaEnvGraphSpec.model_validate(kitchen_pass1_dict())
    merged = inference.infer(spec, [])
    counter = next(ref for ref in merged.object_references if ref.id == "counter_top")
    fridge = next(ref for ref in merged.object_references if ref.id == "fridge")
    assert counter.prim_path == "counter_right_main_group/top_geometry"
    assert fridge.prim_path == "fridge_main_group"


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (
            {
                "object_references": [{
                    "id": "counter_top",
                    "parent_id": "lightwheel_robocasa_kitchen",
                    "prim_path": None,
                    "object_type": "base",
                }]
            },
            "requires a prim_path",
        ),
        (
            {
                "object_references": [{
                    "id": "counter_top",
                    "parent_id": "lightwheel_robocasa_kitchen",
                    "prim_path": "missing_prim",
                    "object_type": "base",
                }]
            },
            "is not in the background prim tree",
        ),
        (
            {
                "object_references": [{
                    "id": "counter_top",
                    "parent_id": "lightwheel_robocasa_kitchen",
                    "prim_path": "counter_right_main_group/top_geometry",
                    "object_type": "rigid",
                }]
            },
            "does not match prim tree",
        ),
    ],
)
@patch("isaaclab_arena.utils.usd_prim_tree.load_usd_prim_tree")
@patch("isaaclab_arena.environment_spec.arena_env_graph_types.AssetSpec.resolve_usd_path")
def test_prim_path_inference_infer_records_invalid_llm_output(
    mock_resolve_usd,
    mock_load_tree,
    stub_openai,
    response,
    match,
):
    mock_resolve_usd.return_value = "/tmp/scene.usd"
    mock_load_tree.return_value = kitchen_prim_tree()
    _, client = stub_openai
    backend = inference_backend(stub_openai)
    client.chat.completions.create.return_value = chat_response(content=json.dumps(response))
    inference = PrimPathInference(backend)
    spec = ArenaEnvGraphSpec.model_validate(kitchen_pass1_dict())
    traces: list[str] = []
    result = inference.infer(spec, traces)
    assert result is None
    assert any(match in line for line in traces), traces
