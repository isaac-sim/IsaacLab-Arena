# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for prompt normalization inference."""

from __future__ import annotations

import json

import pytest

from isaaclab_arena.agentic_environment_generation.prompt_normalization import (
    NormalizedPromptDescriptions,
    PromptNormalizationInference,
    format_normalized_prompt_block,
)
from isaaclab_arena.tests.utils.agentic_environment_generation import (
    chat_response,
    inference_backend,
    minimal_normalized_dict,
)


@pytest.fixture
def prompt_normalization(stub_openai):
    """A ``PromptNormalizationInference`` backed by a mocked OpenAI client."""
    _, client = stub_openai
    inference = PromptNormalizationInference(inference_backend(stub_openai))
    return inference, client


def test_infer_sets_response_format_to_json_schema(prompt_normalization):
    inference, client = prompt_normalization
    client.chat.completions.create.return_value = chat_response(content=json.dumps(minimal_normalized_dict()))
    traces: list[str] = []
    result = inference.infer("pick avocado", traces)
    assert isinstance(result, NormalizedPromptDescriptions)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["name"] == "NormalizedPromptDescriptions"
    assert kwargs["response_format"]["json_schema"]["strict"] is True


def test_infer_user_message_contains_prompt(prompt_normalization):
    inference, client = prompt_normalization
    client.chat.completions.create.return_value = chat_response(content=json.dumps(minimal_normalized_dict()))
    inference.infer("user wants avocado on kitchen", [])
    msgs = client.chat.completions.create.call_args.kwargs["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "user wants avocado on kitchen" in msgs[1]["content"]


def test_infer_returns_none_with_validation_traces_on_invalid_output(prompt_normalization):
    inference, client = prompt_normalization
    invalid = dict(minimal_normalized_dict())
    invalid["env_name"] = ""
    client.chat.completions.create.return_value = chat_response(content=json.dumps(invalid))
    traces: list[str] = []
    result = inference.infer("p", traces)
    assert result is None
    assert traces


def test_format_normalized_prompt_block_lists_objects():
    normalized = NormalizedPromptDescriptions.model_validate(minimal_normalized_dict())
    block = format_normalized_prompt_block(normalized)
    assert "NORMALIZED PROMPT:" in block
    assert "avocado" in block
    assert "bowl" in block


def test_system_prompt_treats_maple_table_as_background_not_object_reference():
    prompt = PromptNormalizationInference._system_prompt()
    assert "maple table" in prompt
    assert "background asset itself is" in prompt
    assert "Do not invent a table-surface reference" in prompt
