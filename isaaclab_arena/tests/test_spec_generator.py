# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for pass-1 environment graph spec generation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from isaaclab_arena.agentic_environment_generation.environment_generation_agent import (
    AssetCatalogue,
    RelationCatalogue,
    TaskCatalogue,
)
from isaaclab_arena.agentic_environment_generation.query_backend import QueryBackend
from isaaclab_arena.agentic_environment_generation.spec_generator import SpecGenerator
from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec

_MINIMAL_SPEC: dict = {
    "env_name": "llm_gen_maple_table_robolab_PickAndPlaceTask",
    "embodiment": {"id": "franka_ik", "registry_name": "franka_ik"},
    "background": {"id": "maple_table_robolab", "registry_name": "maple_table_robolab"},
    "objects": [
        {"id": "rubiks_cube_hot3d_robolab", "registry_name": "rubiks_cube_hot3d_robolab"},
        {"id": "bowl_ycb_robolab", "registry_name": "bowl_ycb_robolab"},
    ],
    "relations": [
        {"kind": "is_anchor", "subject": "maple_table_robolab"},
        {"kind": "on", "subject": "rubiks_cube_hot3d_robolab", "reference": "maple_table_robolab"},
        {"kind": "on", "subject": "bowl_ycb_robolab", "reference": "maple_table_robolab"},
    ],
    "tasks": [{
        "kind": "PickAndPlaceTask",
        "params": {
            "pick_up_object": "rubiks_cube_hot3d_robolab",
            "destination_location": "bowl_ycb_robolab",
            "background_scene": "maple_table_robolab",
        },
        "description": "pick up the rubiks cube and place it in the bowl",
    }],
}


def _chat_response(content: str | None = None, reasoning_content: str | None = None):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].message.reasoning_content = reasoning_content
    return resp


def _catalog(text: str) -> AssetCatalogue:
    catalogue = AssetCatalogue()
    catalogue.to_catalog_string = lambda: text  # type: ignore[method-assign]
    return catalogue


def _relation_catalog(text: str) -> RelationCatalogue:
    catalogue = RelationCatalogue()
    catalogue.to_catalog_string = lambda: text  # type: ignore[method-assign]
    return catalogue


def _task_catalog(text: str) -> TaskCatalogue:
    catalogue = TaskCatalogue()
    catalogue.to_catalog_string = lambda: text  # type: ignore[method-assign]
    return catalogue


@pytest.fixture
def spec_generator():
    """A ``SpecGenerator`` backed by a mocked OpenAI client."""
    client = MagicMock()
    generator = SpecGenerator(QueryBackend(client, "test-model"))
    return generator, client


def _infer(
    generator: SpecGenerator,
    client: MagicMock,
    prompt: str = "p",
    *,
    asset_catalog: AssetCatalogue | None = None,
    relation_catalog: RelationCatalogue | None = None,
    task_catalog: TaskCatalogue | None = None,
    traces: list[str] | None = None,
):
    traces = traces if traces is not None else []
    return generator.infer(
        prompt,
        traces,
        asset_catalog=asset_catalog or _catalog("catalog"),
        relation_catalog=relation_catalog or _relation_catalog("RELATIONS"),
        task_catalog=task_catalog or _task_catalog("TASKS"),
    )


def test_infer_sets_response_format_to_json_schema(spec_generator):
    generator, client = spec_generator
    client.chat.completions.create.return_value = _chat_response(content=json.dumps(_MINIMAL_SPEC))
    _infer(generator, client)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["name"] == "ArenaEnvGraphSpec"
    assert kwargs["response_format"]["json_schema"]["strict"] is True
    assert kwargs["response_format"]["json_schema"]["schema"] is generator._schema


def test_infer_tolerates_unescaped_control_chars(spec_generator):
    generator, client = spec_generator
    payload = dict(_MINIMAL_SPEC)
    payload["env_name"] = "pick\tup"
    raw = json.dumps(payload).replace("\\t", "\t")
    assert "\t" in raw
    client.chat.completions.create.return_value = _chat_response(content=raw)
    spec, _ = _infer(generator, client)
    assert spec is not None
    assert "\t" in spec.env_name


def test_infer_user_message_contains_catalog_and_prompt(spec_generator):
    generator, client = spec_generator
    client.chat.completions.create.return_value = _chat_response(content=json.dumps(_MINIMAL_SPEC))
    _infer(
        generator,
        client,
        "user wants avocado on kitchen",
        asset_catalog=_catalog("<<CATALOG-MARKER>>"),
        relation_catalog=_relation_catalog("<<RELATIONS-MARKER>>"),
        task_catalog=_task_catalog("<<TASKS-MARKER>>"),
    )
    msgs = client.chat.completions.create.call_args.kwargs["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"]
    user_msg = msgs[1]["content"]
    assert "<<CATALOG-MARKER>>" in user_msg
    assert "<<RELATIONS-MARKER>>" in user_msg
    assert "<<TASKS-MARKER>>" in user_msg
    assert "user wants avocado on kitchen" in user_msg


def test_infer_raises_when_response_has_no_choices(spec_generator):
    generator, client = spec_generator
    resp = MagicMock()
    resp.choices = []
    client.chat.completions.create.return_value = resp
    with pytest.raises(RuntimeError, match="failed generate_spec after 4 attempts"):
        _infer(generator, client)
    assert client.chat.completions.create.call_count == 4


def test_infer_retries_after_api_error_then_succeeds(spec_generator):
    generator, client = spec_generator
    client.chat.completions.create.side_effect = [
        ConnectionError("timeout"),
        _chat_response(content=json.dumps(_MINIMAL_SPEC)),
    ]
    spec, _ = _infer(generator, client)
    assert isinstance(spec, ArenaEnvGraphSpec)
    assert spec.background.registry_name == "maple_table_robolab"
    assert client.chat.completions.create.call_count == 2


def test_infer_raises_after_api_errors_exhaust_retries():
    client = MagicMock()
    client.chat.completions.create.side_effect = ConnectionError("timeout")
    generator = SpecGenerator(QueryBackend(client, "test-model", max_retries=1))
    with pytest.raises(RuntimeError, match="failed generate_spec after 2 attempts"):
        _infer(generator, client)
    assert client.chat.completions.create.call_count == 2


def test_infer_returns_none_with_validation_traces_on_invalid_spec(spec_generator):
    generator, client = spec_generator
    invalid = dict(_MINIMAL_SPEC)
    invalid["embodiment"]["registry_name"] = "not_a_real_asset"
    client.chat.completions.create.return_value = _chat_response(content=json.dumps(invalid))
    traces: list[str] = []
    spec, data = _infer(generator, client, traces=traces)
    assert spec is None
    assert data["embodiment"]["registry_name"] == "not_a_real_asset"
    assert traces
    assert any("registry_name" in line for line in traces)
