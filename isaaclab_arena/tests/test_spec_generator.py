# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for pass-1 environment graph spec generation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from isaaclab_arena.agentic_environment_generation.query_backend import QueryBackend
from isaaclab_arena.agentic_environment_generation.spec_generator import SpecGenerator
from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.tests.utils.agentic_environment_generation import catalog as make_catalog
from isaaclab_arena.tests.utils.agentic_environment_generation import chat_response, minimal_spec_dict
from isaaclab_arena.tests.utils.agentic_environment_generation import relation_catalog as make_relation_catalog
from isaaclab_arena.tests.utils.agentic_environment_generation import task_catalog as make_task_catalog


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
    asset_catalog=None,
    relation_catalog=None,
    task_catalog=None,
    traces: list[str] | None = None,
):
    traces = traces if traces is not None else []
    return generator.infer(
        prompt,
        traces,
        asset_catalog=asset_catalog or make_catalog("catalog"),
        relation_catalog=relation_catalog or make_relation_catalog("RELATIONS"),
        task_catalog=task_catalog or make_task_catalog("TASKS"),
    )


def test_infer_sets_response_format_to_json_schema(spec_generator):
    generator, client = spec_generator
    client.chat.completions.create.return_value = chat_response(content=json.dumps(minimal_spec_dict()))
    _infer(generator, client)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["name"] == "ArenaEnvGraphSpec"
    assert kwargs["response_format"]["json_schema"]["strict"] is True
    assert kwargs["response_format"]["json_schema"]["schema"] is generator._schema


def test_infer_tolerates_unescaped_control_chars(spec_generator):
    generator, client = spec_generator
    payload = dict(minimal_spec_dict())
    payload["env_name"] = "pick\tup"
    raw = json.dumps(payload).replace("\\t", "\t")
    assert "\t" in raw
    client.chat.completions.create.return_value = chat_response(content=raw)
    spec, _ = _infer(generator, client)
    assert spec is not None
    assert "\t" in spec.env_name


def test_infer_user_message_contains_catalog_and_prompt(spec_generator):
    generator, client = spec_generator
    client.chat.completions.create.return_value = chat_response(content=json.dumps(minimal_spec_dict()))
    _infer(
        generator,
        client,
        "user wants avocado on kitchen",
        asset_catalog=make_catalog("<<CATALOG-MARKER>>"),
        relation_catalog=make_relation_catalog("<<RELATIONS-MARKER>>"),
        task_catalog=make_task_catalog("<<TASKS-MARKER>>"),
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
        chat_response(content=json.dumps(minimal_spec_dict())),
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
    invalid = dict(minimal_spec_dict())
    invalid["embodiment"]["registry_name"] = "not_a_real_asset"
    client.chat.completions.create.return_value = chat_response(content=json.dumps(invalid))
    traces: list[str] = []
    spec, data = _infer(generator, client, traces=traces)
    assert spec is None
    assert data["embodiment"]["registry_name"] == "not_a_real_asset"
    assert traces
    assert any("registry_name" in line for line in traces)
