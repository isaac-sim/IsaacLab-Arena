# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pass that names objects the asset catalog does not cover."""

from __future__ import annotations

import json
import logging

import pytest

from isaaclab_arena.agentic_environment_generation.inference_backend import InferenceBackend
from isaaclab_arena.agentic_environment_generation.missing_object_inference import (
    MAX_SEARCH_PHRASES,
    MissingObjectInference,
)
from isaaclab_arena.tests.utils.agentic_environment_generation import catalog as make_catalog
from isaaclab_arena.tests.utils.agentic_environment_generation import (
    chat_response,
    inference_backend,
    skip_without_live_endpoint_key,
)

INFERENCE_LOGGER = "isaaclab_arena.agentic_environment_generation.missing_object_inference"
"""Where this pass reports what it found; the agent's traces carry only errors."""


@pytest.fixture
def missing_object_inference(stub_openai):
    """A ``MissingObjectInference`` backed by a mocked OpenAI client."""
    _, client = stub_openai
    return MissingObjectInference(inference_backend(stub_openai)), client


def _phrases_response(*phrases: str):
    return chat_response(content=json.dumps({"search_phrases": list(phrases)}))


def test_infer_returns_the_phrases_the_model_named(missing_object_inference, caplog):
    inference, client = missing_object_inference
    client.chat.completions.create.return_value = _phrases_response("green trash can", "chrome watering can")
    with caplog.at_level(logging.INFO, logger=INFERENCE_LOGGER):
        assert inference.infer("p", make_catalog("catalog")) == ["green trash can", "chrome watering can"]
    assert "green trash can" in caplog.text


def test_infer_returns_nothing_when_the_catalog_covers_the_prompt(missing_object_inference, caplog):
    inference, client = missing_object_inference
    client.chat.completions.create.return_value = _phrases_response()
    with caplog.at_level(logging.INFO, logger=INFERENCE_LOGGER):
        assert inference.infer("p", make_catalog("catalog")) == []
    assert "nothing to search for" in caplog.text


def test_infer_drops_blank_phrases(missing_object_inference):
    inference, client = missing_object_inference
    client.chat.completions.create.return_value = _phrases_response("  green trash can  ", "   ", "")
    assert inference.infer("p", make_catalog("catalog")) == ["green trash can"]


def test_infer_caps_how_many_objects_one_prompt_can_search_for(missing_object_inference, caplog):
    inference, client = missing_object_inference
    asked = [f"object {index}" for index in range(MAX_SEARCH_PHRASES + 3)]
    client.chat.completions.create.return_value = _phrases_response(*asked)
    with caplog.at_level(logging.WARNING, logger=INFERENCE_LOGGER):
        phrases = inference.infer("p", make_catalog("catalog"))
    assert phrases == asked[:MAX_SEARCH_PHRASES]
    assert f"first {MAX_SEARCH_PHRASES} are searched" in caplog.text


def test_infer_shows_the_model_the_catalog_and_the_prompt(missing_object_inference):
    inference, client = missing_object_inference
    client.chat.completions.create.return_value = _phrases_response()
    inference.infer("user wants a green trash can", make_catalog("<<CATALOG-MARKER>>"))
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"]["json_schema"]["name"] == "MissingObjects"
    user_msg = kwargs["messages"][1]["content"]
    assert "<<CATALOG-MARKER>>" in user_msg
    assert "user wants a green trash can" in user_msg


_LIVE_CATALOG = (
    "EMBODIMENTS:\n- droid_abs_joint_pos  tags=[default]\n\n"
    "BACKGROUNDS: maple_table\n\n"
    "OBJECTS:\n"
    "- avocado01_fruits_veggies_robolab  tags=[]\n"
    "- plate_large_vomp_robolab  tags=[]\n"
    "- broccoli  tags=[]"
)
"""An asset catalog with three fruit-and-tableware objects, and nothing a watering can matches."""


# Marked flaky to absorb intermittent wire-level hiccups on the inference endpoint.
@skip_without_live_endpoint_key()
@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_infer_names_the_uncatalogued_object_against_live_endpoint():
    """Live test: the one object the catalog has no match for comes back as a search phrase."""
    inference = MissingObjectInference(InferenceBackend())
    phrases = inference.infer(
        "droid picks up a green watering can from the maple table and places it on the plate",
        make_catalog(_LIVE_CATALOG),
    )
    assert any("watering can" in phrase.lower() for phrase in phrases), f"got {phrases!r}"


@skip_without_live_endpoint_key()
@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_infer_names_nothing_when_the_catalog_covers_the_prompt_against_live_endpoint():
    """Live test: a prompt every object of which is catalogued searches for nothing."""
    inference = MissingObjectInference(InferenceBackend())
    phrases = inference.infer(
        "droid picks up the avocado from the maple table and places it on the plate",
        make_catalog(_LIVE_CATALOG),
    )
    assert phrases == [], f"got {phrases!r}"
