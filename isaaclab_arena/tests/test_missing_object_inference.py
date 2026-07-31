# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pass that names objects the asset catalog does not cover."""

from __future__ import annotations

import json

import pytest

from isaaclab_arena.agentic_environment_generation.missing_object_inference import (
    MAX_SEARCH_PHRASES,
    MissingObjectInference,
)
from isaaclab_arena.tests.utils.agentic_environment_generation import catalog as make_catalog
from isaaclab_arena.tests.utils.agentic_environment_generation import chat_response, inference_backend


@pytest.fixture
def missing_object_inference(stub_openai):
    """A ``MissingObjectInference`` backed by a mocked OpenAI client."""
    _, client = stub_openai
    return MissingObjectInference(inference_backend(stub_openai)), client


def _phrases_response(*phrases: str):
    return chat_response(content=json.dumps({"search_phrases": list(phrases)}))


def test_infer_returns_the_phrases_the_model_named(missing_object_inference):
    inference, client = missing_object_inference
    client.chat.completions.create.return_value = _phrases_response("green trash can", "chrome watering can")
    traces: list[str] = []
    assert inference.infer("p", make_catalog("catalog"), traces) == ["green trash can", "chrome watering can"]
    assert any("green trash can" in line for line in traces)


def test_infer_returns_nothing_when_the_catalog_covers_the_prompt(missing_object_inference):
    inference, client = missing_object_inference
    client.chat.completions.create.return_value = _phrases_response()
    traces: list[str] = []
    assert inference.infer("p", make_catalog("catalog"), traces) == []
    assert any("nothing to search for" in line for line in traces)


def test_infer_drops_blank_phrases(missing_object_inference):
    inference, client = missing_object_inference
    client.chat.completions.create.return_value = _phrases_response("  green trash can  ", "   ", "")
    assert inference.infer("p", make_catalog("catalog"), []) == ["green trash can"]


def test_infer_caps_how_many_objects_one_prompt_can_search_for(missing_object_inference):
    inference, client = missing_object_inference
    asked = [f"object {index}" for index in range(MAX_SEARCH_PHRASES + 3)]
    client.chat.completions.create.return_value = _phrases_response(*asked)
    traces: list[str] = []
    phrases = inference.infer("p", make_catalog("catalog"), traces)
    assert phrases == asked[:MAX_SEARCH_PHRASES]
    assert any(f"first {MAX_SEARCH_PHRASES} are searched" in line for line in traces)


def test_infer_shows_the_model_the_catalog_and_the_prompt(missing_object_inference):
    inference, client = missing_object_inference
    client.chat.completions.create.return_value = _phrases_response()
    inference.infer("user wants a green trash can", make_catalog("<<CATALOG-MARKER>>"), [])
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"]["json_schema"]["name"] == "MissingObjects"
    user_msg = kwargs["messages"][1]["content"]
    assert "<<CATALOG-MARKER>>" in user_msg
    assert "user wants a green trash can" in user_msg
