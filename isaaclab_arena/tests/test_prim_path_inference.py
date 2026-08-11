# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for object_reference prim_path inference."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from isaaclab_arena.agentic_environment_generation.prim_path_inference import (
    PrimPathInference,
    _enforce_object_reference_types,
    _prim_tree_catalog,
    _validate_against_prim_tree,
)
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_types import ObjectReferenceSpec
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
        == "BACKGROUND PRIM TREE:\ncab_1_main_group (articulation right_door_joint)\n  corpus (rigid)\n    back (base)"
    )


@pytest.mark.parametrize("prim_type", list(ObjectType))
def test_validate_against_prim_tree_allows_base_reference_to_any_prim_type(prim_type):
    reference = ObjectReferenceSpec(
        id="target",
        parent_id="background",
        prim_path="target",
        object_type=ObjectType.BASE,
    )
    _validate_against_prim_tree([reference], [UsdPrimRecord("target", prim_type)])


@pytest.mark.parametrize(
    ("reference_type", "prim_type"),
    [
        (ObjectType.RIGID, ObjectType.BASE),
        (ObjectType.RIGID, ObjectType.ARTICULATION),
        (ObjectType.ARTICULATION, ObjectType.BASE),
        (ObjectType.ARTICULATION, ObjectType.RIGID),
    ],
)
def test_validate_against_prim_tree_rejects_non_base_type_mismatch(reference_type, prim_type):
    reference = ObjectReferenceSpec(
        id="target",
        parent_id="background",
        prim_path="target",
        object_type=reference_type,
    )
    with pytest.raises(AssertionError, match="does not match prim tree"):
        _validate_against_prim_tree([reference], [UsdPrimRecord("target", prim_type)])


def test_enforce_object_reference_types_uses_task_roles(capsys):
    data = kitchen_pass1_dict()
    references = {reference["id"]: reference for reference in data["object_references"]}
    references["counter_top"]["object_type"] = "articulation"
    references["fridge"]["object_type"] = "rigid"
    data["object_references"].append({
        "id": "floor",
        "parent_id": "lightwheel_robocasa_kitchen",
        "object_type": "rigid",
    })
    data["task"]["subtasks"][0]["params"]["pick_up_object"] = "counter_top"
    spec = ArenaEnvGraphSpec.model_validate(data)

    updated = _enforce_object_reference_types(spec)

    types = {reference.id: reference.object_type for reference in updated.object_references}
    assert types == {
        "counter_top": ObjectType.RIGID,
        "fridge": ObjectType.ARTICULATION,
        "floor": ObjectType.BASE,
    }
    assert {reference.id: reference.object_type for reference in spec.object_references} == {
        "counter_top": ObjectType.ARTICULATION,
        "fridge": ObjectType.RIGID,
        "floor": ObjectType.RIGID,
    }
    output = capsys.readouterr().out
    assert "'counter_top' type 'articulation' -> 'rigid'" in output
    assert "'fridge' type 'rigid' -> 'articulation'" in output
    assert "'floor' type 'rigid' -> 'base'" in output


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
