# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for environment graph spec inference."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from isaaclab_arena.agentic_environment_generation.prompt_normalization import (
    NormalizedPromptDescriptions,
    format_normalized_prompt_block,
)
from isaaclab_arena.agentic_environment_generation.simready_asset_search import (
    SIMREADY_USD_OBJECT_REGISTRY_NAME,
    SimReadyCandidateCatalogue,
    SimReadyObjectCandidate,
)
from isaaclab_arena.agentic_environment_generation.spec_inference import SpecInference
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.tests.utils.agentic_environment_generation import catalog as make_catalog
from isaaclab_arena.tests.utils.agentic_environment_generation import (
    chat_response,
    inference_backend,
    minimal_normalized_dict,
    minimal_spec_dict,
)
from isaaclab_arena.tests.utils.agentic_environment_generation import relation_catalog as make_relation_catalog
from isaaclab_arena.tests.utils.agentic_environment_generation import task_catalog as make_task_catalog


@pytest.fixture
def spec_inference(stub_openai):
    """A ``SpecInference`` backed by a mocked OpenAI client."""
    _, client = stub_openai
    return SpecInference(inference_backend(stub_openai)), client


def _infer(
    inference: SpecInference,
    client: MagicMock,
    prompt: str = "p",
    *,
    asset_catalog=None,
    relation_catalog=None,
    task_catalog=None,
    normalized_prompt_block: str | None = None,
    simready_candidate_catalog=None,
    traces: list[str] | None = None,
):
    traces = traces if traces is not None else []
    return inference.infer(
        prompt,
        traces,
        asset_catalog=asset_catalog or make_catalog("catalog"),
        relation_catalog=relation_catalog or make_relation_catalog("RELATIONS"),
        task_catalog=task_catalog or make_task_catalog("TASKS"),
        normalized_prompt_block=normalized_prompt_block,
        simready_candidate_catalog=simready_candidate_catalog,
    )


def test_infer_sets_response_format_to_json_schema(spec_inference):
    inference, client = spec_inference
    client.chat.completions.create.return_value = chat_response(content=json.dumps(minimal_spec_dict()))
    _infer(inference, client)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["name"] == "ArenaEnvGraphSpec"
    assert kwargs["response_format"]["json_schema"]["strict"] is True
    assert kwargs["response_format"]["json_schema"]["schema"] is inference._schema


def test_infer_user_message_contains_catalog_and_prompt(spec_inference):
    inference, client = spec_inference
    client.chat.completions.create.return_value = chat_response(content=json.dumps(minimal_spec_dict()))
    _infer(
        inference,
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


def test_infer_retries_after_api_error_then_succeeds(spec_inference):
    inference, client = spec_inference
    client.chat.completions.create.side_effect = [
        ConnectionError("timeout"),
        chat_response(content=json.dumps(minimal_spec_dict())),
    ]
    spec, _ = _infer(inference, client)
    assert isinstance(spec, ArenaEnvGraphSpec)
    assert spec.background.registry_name == "maple_table_robolab"
    assert client.chat.completions.create.call_count == 2


def test_infer_returns_none_with_validation_traces_on_invalid_spec(spec_inference):
    inference, client = spec_inference
    invalid = dict(minimal_spec_dict())
    invalid["embodiment"]["registry_name"] = "not_a_real_asset"
    client.chat.completions.create.return_value = chat_response(content=json.dumps(invalid))
    traces: list[str] = []
    spec, data = _infer(inference, client, traces=traces)
    assert spec is None
    assert data["embodiment"]["registry_name"] == "not_a_real_asset"
    assert traces
    assert any("registry_name" in line for line in traces)


def test_infer_user_message_includes_normalized_and_simready_blocks(spec_inference):
    inference, client = spec_inference
    client.chat.completions.create.return_value = chat_response(content=json.dumps(minimal_spec_dict()))
    normalized = format_normalized_prompt_block(NormalizedPromptDescriptions.model_validate(minimal_normalized_dict()))
    simready_catalog = SimReadyCandidateCatalogue(
        candidates=[
            SimReadyObjectCandidate(
                search_phrase="red hammer",
                usd_path="s3://bucket/red_hammer.usd",
            )
        ]
    )
    _infer(
        inference,
        client,
        normalized_prompt_block=normalized,
        simready_candidate_catalog=simready_catalog,
    )
    user_msg = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    system_msg = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "NORMALIZED PROMPT:" in user_msg
    assert "SIMREADY_OBJECT_CANDIDATES" in user_msg
    assert SIMREADY_USD_OBJECT_REGISTRY_NAME in user_msg
    assert SIMREADY_USD_OBJECT_REGISTRY_NAME in system_msg
