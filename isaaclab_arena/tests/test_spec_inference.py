# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for environment graph spec inference."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from isaaclab_arena.agentic_environment_generation.spec_inference import SpecInference
from isaaclab_arena.assets.simready_constants import SIMREADY_USD_OBJECT_REGISTRY_NAME
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_types import TaskCompositionType
from isaaclab_arena.tests.utils.agentic_environment_generation import (
    catalogs,
    chat_response,
    inference_backend,
    minimal_spec_dict,
    object_set_spec_dict,
)


@pytest.fixture
def spec_inference(stub_openai):
    """A ``SpecInference`` backed by a mocked OpenAI client."""
    _, client = stub_openai
    return SpecInference(inference_backend(stub_openai)), client


def _infer(
    spec_inference: tuple[SpecInference, MagicMock],
    data: dict | None = None,
    *,
    prompt: str = "p",
    traces: list[str] | None = None,
    **catalog_texts: str,
):
    """Infer a spec from ``data`` as the model's response, or from whatever the test stubbed."""
    inference, client = spec_inference
    if data is not None:
        client.chat.completions.create.return_value = chat_response(content=json.dumps(data))
    return inference.infer(prompt, traces if traces is not None else [], **catalogs(**catalog_texts))


def test_infer_sets_response_format_to_json_schema(spec_inference):
    inference, client = spec_inference
    _infer(spec_inference, minimal_spec_dict())
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["name"] == "ArenaEnvGraphSpec"
    assert kwargs["response_format"]["json_schema"]["strict"] is True
    assert kwargs["response_format"]["json_schema"]["schema"] is inference._schema


def test_infer_user_message_contains_catalog_and_prompt(spec_inference):
    _, client = spec_inference
    _infer(
        spec_inference,
        minimal_spec_dict(),
        prompt="user wants avocado on kitchen",
        assets="<<CATALOG-MARKER>>",
        relations="<<RELATIONS-MARKER>>",
        tasks="<<TASKS-MARKER>>",
    )
    msgs = client.chat.completions.create.call_args.kwargs["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"]
    user_msg = msgs[1]["content"]
    assert "<<CATALOG-MARKER>>" in user_msg
    assert "<<RELATIONS-MARKER>>" in user_msg
    assert "<<TASKS-MARKER>>" in user_msg
    assert "user wants avocado on kitchen" in user_msg


def test_infer_retries_after_api_error_then_succeeds(spec_inference):
    _, client = spec_inference
    client.chat.completions.create.side_effect = [
        ConnectionError("timeout"),
        chat_response(content=json.dumps(minimal_spec_dict())),
    ]
    spec, _ = _infer(spec_inference)
    assert isinstance(spec, ArenaEnvGraphSpec)
    assert spec.background.registry_name == "maple_table_robolab"
    assert client.chat.completions.create.call_count == 2


def test_infer_returns_none_with_validation_traces_on_invalid_spec(spec_inference):
    invalid = minimal_spec_dict()
    invalid["embodiment"]["registry_name"] = "not_a_real_asset"
    traces: list[str] = []
    spec, data = _infer(spec_inference, invalid, traces=traces)
    assert spec is None
    assert data["embodiment"]["registry_name"] == "not_a_real_asset"
    assert any("registry_name" in line for line in traces)


def test_infer_expands_an_empty_relation_reference_to_none(spec_inference):
    data = minimal_spec_dict()
    data["relations"][0]["reference"] = ""
    spec, _ = _infer(spec_inference, data)
    assert spec.relations[0].kind == "is_anchor"
    assert spec.relations[0].reference is None


def test_infer_expands_a_registry_name_reference_into_a_node_id(spec_inference):
    data = minimal_spec_dict()
    data["background"]["id"] = "bg_1"
    data["relations"][0]["subject"] = "bg_1"
    spec, _ = _infer(spec_inference, data)
    # The remaining relations and the task params still name the background by its registry_name.
    assert spec.relations[1].reference == "bg_1"
    assert spec.task.subtasks[0].params["background_scene"] == "bg_1"


def test_infer_leaves_an_ambiguous_registry_name_alone(spec_inference):
    # Two objects share the name, so nothing says which one a reference to it means.
    data = minimal_spec_dict()
    data["objects"] = [
        {"id": "bowl_1", "registry_name": "bowl_ycb_robolab"},
        {"id": "bowl_2", "registry_name": "bowl_ycb_robolab"},
    ]
    data["relations"] = [{"kind": "is_anchor", "subject": "maple_table_robolab"}]
    data["task"]["subtasks"][0]["params"]["pick_up_object"] = "bowl_1"
    data["task"]["subtasks"][0]["params"]["destination_location"] = "bowl_ycb_robolab"
    traces: list[str] = []
    spec, _ = _infer(spec_inference, data, traces=traces)
    assert spec is None
    assert any("bowl_ycb_robolab" in line for line in traces)


def test_infer_expands_single_subtask_composition_to_atomic(spec_inference):
    data = minimal_spec_dict()
    data["task"]["composition"] = "sequential"
    spec, _ = _infer(spec_inference, data)
    assert spec.task.composition is TaskCompositionType.ATOMIC


def test_infer_leaves_multi_subtask_composition_alone(spec_inference):
    data = minimal_spec_dict()
    data["task"]["composition"] = "sequential"
    data["task"]["subtasks"] *= 2
    spec, _ = _infer(spec_inference, data)
    assert spec.task.composition is TaskCompositionType.SEQUENTIAL


def _with_nested_object_sets() -> dict:
    """Return the object-set spec with each member wrapped in a one-member set of its own."""
    data = object_set_spec_dict()
    data["object_sets"] = [
        {"id": "pick_up_object_set", "members": ["sweet_potato_set", "jug_set"]},
        {"id": "sweet_potato_set", "members": ["sweet_potato"]},
        {"id": "jug_set", "members": ["jug"]},
    ]
    return data


def test_infer_expands_nested_object_sets_into_their_members(spec_inference):
    spec, _ = _infer(spec_inference, _with_nested_object_sets())
    assert [object_set.id for object_set in spec.object_sets] == ["pick_up_object_set"]
    assert spec.object_sets[0].members == ["sweet_potato", "jug"]


def test_infer_keeps_a_nested_object_set_a_relation_names(spec_inference):
    # Inlining a set into another does not unname it; a relation still puts it in the scene.
    data = _with_nested_object_sets()
    data["relations"].append({"kind": "on", "subject": "jug_set", "reference": "maple_table_robolab"})
    spec, _ = _infer(spec_inference, data)
    assert "jug_set" in [object_set.id for object_set in spec.object_sets]


def test_infer_never_mentions_the_asset_search(spec_inference):
    # Searching for assets is a separate pass that extends the catalog before this one runs, so
    # every object here is spawnable and spec inference has nothing to say about where it came from.
    _, client = spec_inference
    _infer(spec_inference, minimal_spec_dict())
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert SIMREADY_USD_OBJECT_REGISTRY_NAME not in messages[0]["content"]
    assert "simready" not in messages[0]["content"].lower()
