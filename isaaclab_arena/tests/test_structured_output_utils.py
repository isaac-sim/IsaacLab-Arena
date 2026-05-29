# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from isaaclab_arena.environments.agentic_env_gen.env_gen_agent import DEFAULT_BASE_URL, DEFAULT_MODEL
from isaaclab_arena.environments.agentic_env_gen.env_intent_spec import EnvIntentSpec
from isaaclab_arena.environments.agentic_env_gen.structured_output_utils import (
    apply_strict_constraints,
    build_strict_schema,
    check_structured_output_support,
    ping,
)

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _chat_response(content: str | None = None, reasoning_content: str | None = None, finish_reason: str = "stop"):
    """Build a nested mock matching the openai chat-completion response shape."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].finish_reason = finish_reason
    resp.choices[0].message.content = content
    resp.choices[0].message.reasoning_content = reasoning_content
    return resp


# ---------------------------------------------------------------------------
# build_strict_schema / apply_strict_constraints
# ---------------------------------------------------------------------------


class _ToyChild(BaseModel):
    name: str
    optional_value: int | None = None


class _ToyParent(BaseModel):
    title: str
    child: _ToyChild
    children: list[_ToyChild] = []


class TestBuildStrictSchema:
    def test_apply_strict_constraints_is_idempotent(self):
        # Safe to call multiple times — the second pass must be a no-op.
        schema = build_strict_schema(_ToyParent)
        snapshot = json.dumps(schema, sort_keys=True)
        apply_strict_constraints(schema)
        assert json.dumps(schema, sort_keys=True) == snapshot


# ---------------------------------------------------------------------------
# ping
# ---------------------------------------------------------------------------


class TestPing:
    def test_error_paths(self):
        class FakeAuthError(Exception):
            pass

        client = MagicMock()
        # SDK exceptions propagate unchanged.
        client.chat.completions.create.side_effect = FakeAuthError("invalid api key")
        with pytest.raises(FakeAuthError, match="invalid api key"):
            ping(client, "m")

        # Same client, different failure shape: 200 OK with no choices.
        client.chat.completions.create.side_effect = None
        resp = MagicMock()
        resp.choices = []
        client.chat.completions.create.return_value = resp
        with pytest.raises(RuntimeError, match="no choices") as exc_info:
            ping(client, "guardrailed-model")
        # Model name surfaces in the message — most-grepped field when
        # triaging a CI ping failure.
        assert "'guardrailed-model'" in str(exc_info.value)


# ---------------------------------------------------------------------------
# check_structured_output_support (mocked)
# ---------------------------------------------------------------------------


class TestCheckStructuredOutputSupport:
    """Inject five distinct failures and test that the function raises a RuntimeError."""

    @pytest.mark.parametrize(
        "failure_mode",
        [
            # 4xx wire rejection — the most common signal that a model
            # doesn't support ``response_format`` / strict schemas.
            "wire_4xx",
            # 200 OK but the model produced nothing on either channel.
            "empty_envelope",
            # 200 OK with an empty ``choices`` list.
            "empty_choices",
            # Wire returns prose / preamble instead of JSON.
            "invalid_json",
            # JSON parses, but doesn't conform to the spec schema.
            "schema_mismatch",
        ],
    )
    def test_raises_runtime_error_on_failure(self, failure_mode):
        client = MagicMock()
        if failure_mode == "wire_4xx":
            client.chat.completions.create.side_effect = Exception("Error code: 400 - additionalProperties")
        elif failure_mode == "empty_envelope":
            client.chat.completions.create.return_value = _chat_response(content=None, reasoning_content=None)
        elif failure_mode == "empty_choices":
            resp = MagicMock()
            resp.choices = []
            client.chat.completions.create.return_value = resp
        elif failure_mode == "invalid_json":
            client.chat.completions.create.return_value = _chat_response(content="not json")
        elif failure_mode == "schema_mismatch":
            client.chat.completions.create.return_value = _chat_response(content='{"missing": "fields"}')

        with pytest.raises(RuntimeError):
            check_structured_output_support(client, "m", EnvIntentSpec)


# ---------------------------------------------------------------------------
# Live endpoint (network + auth required)
# ---------------------------------------------------------------------------


# TODO(qianl): drop the flaky marker once production-side retry is wired.
@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_default_model_supports_structured_output():
    """The default ``EnvGenAgent`` model must support structured outputs."""
    api_key = os.environ.get("NV_API_KEY")
    assert api_key, "NV_API_KEY env var required to run live tests"

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL)
    assert check_structured_output_support(client, DEFAULT_MODEL, EnvIntentSpec) is True
