# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`isaaclab_arena.environments.agentic_env_gen.structured_output_utils`.

The utility module owns the three concerns that decouple "is this
endpoint compatible with our structured-outputs contract?" from the
agent's higher-level pipeline:

* ``build_strict_schema`` / ``apply_strict_constraints`` — schema
  munging that walks every object node (``$defs``, nested arrays,
  ``anyOf`` arms) and applies OpenAI strict-mode constraints. Locked
  in here so a future pydantic version that changes default schema
  output doesn't silently regress Bedrock compatibility.
* ``extract_response_text`` — the NVIDIA-DeepSeek-vs-OpenAI channel
  fallback (``content`` first, then ``reasoning_content``,
  ``"empty"`` last).
* ``check_structured_output_support`` — the deployment validator's
  diagnostic probe. Tested both with mocks (failure-mode coverage)
  and against the real default model (so we notice the day
  NVIDIA's hosted DeepSeek-v4-flash drops structured-output
  support).
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from isaaclab_arena.environments.agentic_env_gen.env_gen_agent import DEFAULT_BASE_URL, DEFAULT_MODEL
from isaaclab_arena.environments.agentic_env_gen.env_intent_spec import EnvIntentSpec
from isaaclab_arena.environments.agentic_env_gen.structured_output_utils import (
    StructuredOutputSupport,
    apply_strict_constraints,
    build_strict_schema,
    check_structured_output_support,
    extract_response_text,
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


# Minimal EnvIntentSpec payload that satisfies the ``tasks_must_be_non_empty``
# validator — reused across the ``check_structured_output_support`` happy-path
# tests so they exercise the real production schema rather than a toy stub.
_MINIMAL_SPEC: dict = {
    "reasoning": (
        "User wants a pick-and-place: foreground object is 'avocado', "
        "target container is 'bowl', background is the kitchen table."
    ),
    "task_description": "pick up the avocado and place it in the bowl",
    "background": "kitchen",
    "embodiment": "franka_ik",
    "items": [
        {"query": "avocado", "role": "foreground", "category_tags": [], "instance_name": None, "scale": None},
        {"query": "bowl", "role": "foreground", "category_tags": [], "instance_name": None, "scale": None},
    ],
    "initial_scene_graph": [
        {"kind": "on", "subject": "avocado", "target": "kitchen"},
        {"kind": "on", "subject": "bowl", "target": "kitchen"},
    ],
    "tasks": [{
        "kind": "pick_and_place",
        "subject": "avocado",
        "target": "bowl",
        "description": "pick up the avocado and place it in the bowl",
    }],
}


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
    def test_root_object_additional_properties_false(self):
        schema = build_strict_schema(_ToyParent)
        assert schema["additionalProperties"] is False

    def test_root_object_lists_every_property_as_required(self):
        schema = build_strict_schema(_ToyParent)
        assert set(schema["required"]) == {"title", "child", "children"}

    def test_nested_defs_object_also_strict(self):
        # OpenAI strict mode applies the constraint to *every* object node,
        # not just the top level — including ``$defs`` entries that get
        # referenced via ``$ref``. Bedrock in particular rejects the request
        # if any descendant object schema is missing the marker.
        schema = build_strict_schema(_ToyParent)
        defs = schema["$defs"]
        assert defs["_ToyChild"]["additionalProperties"] is False
        assert set(defs["_ToyChild"]["required"]) == {"name", "optional_value"}

    def test_defaults_stripped_everywhere(self):
        # Pydantic emits ``"default": null`` for ``optional_value`` at the
        # property level; strict mode rejects ``default`` since every field
        # is required. Drop the key recursively.
        schema = build_strict_schema(_ToyParent)
        stack = [schema]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                assert "default" not in node, f"unexpected default key in {node!r}"
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)

    def test_munging_does_not_mutate_pydantic_cached_schema(self):
        # Pydantic caches ``model_json_schema()`` results internally; our
        # munger MUST work on a deep copy so the cache stays clean for
        # other callers (e.g. ``model_dump_json()`` consumers).
        before = json.dumps(_ToyParent.model_json_schema(), sort_keys=True)
        build_strict_schema(_ToyParent)
        after = json.dumps(_ToyParent.model_json_schema(), sort_keys=True)
        assert before == after

    def test_apply_strict_constraints_is_idempotent(self):
        # Safe to call multiple times — the second pass must be a no-op.
        # Important because callers may receive an already-munged schema
        # from a cache and re-apply defensively.
        schema = build_strict_schema(_ToyParent)
        snapshot = json.dumps(schema, sort_keys=True)
        apply_strict_constraints(schema)
        assert json.dumps(schema, sort_keys=True) == snapshot

    def test_env_intent_spec_munges_clean(self):
        # The real production schema we ship — confirm every object node
        # has the strict-mode marker so the wire stays compatible with
        # Bedrock and any other strict-mode validator users point at.
        schema = build_strict_schema(EnvIntentSpec)

        def assert_strict(node):
            if isinstance(node, dict):
                if node.get("type") == "object" and "properties" in node:
                    assert node.get("additionalProperties") is False
                    assert set(node["required"]) == set(node["properties"].keys())
                for v in node.values():
                    assert_strict(v)
            elif isinstance(node, list):
                for v in node:
                    assert_strict(v)

        assert_strict(schema)


# ---------------------------------------------------------------------------
# extract_response_text
# ---------------------------------------------------------------------------


class TestExtractResponseText:
    def test_prefers_content_when_both_populated(self):
        msg = MagicMock(content='{"a": 1}', reasoning_content='{"b": 2}')
        text, route = extract_response_text(msg)
        assert text == '{"a": 1}'
        assert route == "content"

    def test_falls_back_to_reasoning_content_when_content_empty(self):
        # NVIDIA DeepSeek-v4-flash routes structured outputs into the
        # provider-specific ``reasoning_content`` channel and leaves
        # ``content`` as ``None``. The agent must transparently read either.
        msg = MagicMock(content=None, reasoning_content='{"b": 2}')
        text, route = extract_response_text(msg)
        assert text == '{"b": 2}'
        assert route == "reasoning_content"

    def test_empty_when_both_channels_blank(self):
        msg = MagicMock(content=None, reasoning_content=None)
        text, route = extract_response_text(msg)
        assert text == ""
        assert route == "empty"

    def test_empty_when_message_has_no_attrs(self):
        # Some mock / stub message objects don't define the channels at all;
        # ``getattr(..., None)`` must still resolve to "empty" rather than
        # raising AttributeError.
        msg = object()  # bare object, no attrs
        text, route = extract_response_text(msg)
        assert text == ""
        assert route == "empty"

    def test_treats_empty_string_as_falsy(self):
        # ``""`` and ``None`` must both route to the fallback (otherwise an
        # empty content with a populated reasoning_content would never
        # reach the reasoning channel).
        msg = MagicMock(content="", reasoning_content='{"b": 2}')
        text, route = extract_response_text(msg)
        assert text == '{"b": 2}'
        assert route == "reasoning_content"


# ---------------------------------------------------------------------------
# ping
# ---------------------------------------------------------------------------


class TestPing:
    def test_returns_response_content(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _chat_response(content="OK")
        assert ping(client, "any-model") == "OK"

    def test_returns_empty_string_when_content_is_none(self):
        # Some providers return ``None`` content alongside a finish_reason — we
        # treat that as a successful round-trip (the wire works) rather than
        # raising, since the caller's contract is "did this raise?".
        client = MagicMock()
        client.chat.completions.create.return_value = _chat_response(content=None)
        assert ping(client, "any-model") == ""

    def test_uses_minimal_request_params(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _chat_response(content="OK")
        ping(client, "model-name")
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "model-name"
        assert kwargs["temperature"] == 0
        assert kwargs["max_tokens"] == 8
        # Single user message — no system prompt / catalog payload. Keeping the
        # request small is the whole point: ping must stay cheap enough to
        # gate every agent construction.
        assert len(kwargs["messages"]) == 1
        assert kwargs["messages"][0]["role"] == "user"
        # ping is a structured-outputs-agnostic liveness check; it must NOT
        # ask the model to honour response_format (otherwise it can't fail
        # gracefully on models that lack structured-output support, which
        # defeats the point of having a cheap probe).
        assert "response_format" not in kwargs

    def test_propagates_client_exceptions(self):
        class FakeAuthError(Exception):
            pass

        client = MagicMock()
        client.chat.completions.create.side_effect = FakeAuthError("invalid api key")
        with pytest.raises(FakeAuthError, match="invalid api key"):
            ping(client, "m")


# ---------------------------------------------------------------------------
# check_structured_output_support (mocked)
# ---------------------------------------------------------------------------


class TestCheckStructuredOutputSupport:
    def test_reports_supported_on_valid_envelope(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _chat_response(content=json.dumps(_MINIMAL_SPEC))
        result = check_structured_output_support(client, "some-model", EnvIntentSpec)
        assert isinstance(result, StructuredOutputSupport)
        assert result.supported is True
        assert result.model == "some-model"
        assert result.response_route == "content"
        assert result.api_error is None
        assert result.parse_error is None
        assert result.sample_payload  # truncated text echoed for diagnostics

    def test_reports_reasoning_content_route(self):
        # NVIDIA DeepSeek envelope — the canonical reason this helper
        # exists. We must report ``supported=True`` AND surface the
        # route so deployment validators can flag the model as "works
        # but uses the non-standard channel".
        client = MagicMock()
        client.chat.completions.create.return_value = _chat_response(
            content=None, reasoning_content=json.dumps(_MINIMAL_SPEC)
        )
        result = check_structured_output_support(client, "deepseek", EnvIntentSpec)
        assert result.supported is True
        assert result.response_route == "reasoning_content"

    def test_reports_api_error_on_4xx(self):
        # The most common "model doesn't support structured outputs"
        # signal at the wire level: a 4xx rejecting the
        # ``response_format`` parameter or the schema. Surface it as
        # ``api_error``, leave ``parse_error`` empty, so callers can
        # attribute correctly.
        class FakeBadRequest(Exception):
            pass

        client = MagicMock()
        client.chat.completions.create.side_effect = FakeBadRequest("Error code: 400 - additionalProperties")
        result = check_structured_output_support(client, "claude", EnvIntentSpec)
        assert result.supported is False
        assert result.api_error is not None
        assert "FakeBadRequest" in result.api_error
        assert "400" in result.api_error
        assert result.parse_error is None
        # On an api_error, no payload is available to echo.
        assert result.sample_payload is None
        assert result.finish_reason is None

    def test_reports_parse_error_on_empty_envelope(self):
        # Wire accepts the request, model produces nothing on either
        # channel. The endpoint silently dropped the structured output
        # — the most insidious failure mode, since ``finish_reason``
        # still reads ``stop``.
        client = MagicMock()
        client.chat.completions.create.return_value = _chat_response(content=None, reasoning_content=None)
        result = check_structured_output_support(client, "broken", EnvIntentSpec)
        assert result.supported is False
        assert result.api_error is None
        assert result.parse_error is not None
        assert "empty envelope" in result.parse_error
        assert result.response_route == "empty"
        assert result.finish_reason == "stop"  # forwarded so callers can correlate

    def test_reports_parse_error_when_choices_list_is_empty(self):
        # Real provider behaviour: HTTP returns 200 OK but ``choices`` is
        # an empty list. Seen on Azure when a content-filter trips, and
        # on Bedrock when a guardrail rejects the response post-hoc.
        # Naive ``resp.choices[0]`` access would IndexError and break
        # the always-return-a-StructuredOutputSupport contract; the
        # function must instead surface it as a ``parse_error`` with
        # ``response_route="empty"`` so callers route it the same way
        # they route an empty envelope.
        resp = MagicMock()
        resp.choices = []
        client = MagicMock()
        client.chat.completions.create.return_value = resp
        result = check_structured_output_support(client, "guardrailed", EnvIntentSpec)
        assert result.supported is False
        assert result.api_error is None
        assert result.parse_error is not None
        assert "no choices" in result.parse_error
        assert result.response_route == "empty"
        assert result.finish_reason is None
        assert result.sample_payload is None

    def test_reports_parse_error_on_invalid_json(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _chat_response(content="not json")
        result = check_structured_output_support(client, "m", EnvIntentSpec)
        assert result.supported is False
        assert result.parse_error is not None
        assert "JSONDecodeError" in result.parse_error
        assert result.sample_payload == "not json"

    def test_reports_parse_error_on_validation_failure(self):
        # JSON parses fine, but doesn't match the schema. The probe
        # exists to detect this exact class of "model returns
        # something, but it's wrong" failure.
        client = MagicMock()
        client.chat.completions.create.return_value = _chat_response(content='{"missing": "fields"}')
        result = check_structured_output_support(client, "m", EnvIntentSpec)
        assert result.supported is False
        assert result.parse_error is not None
        assert "ValidationError" in result.parse_error

    def test_request_shape(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _chat_response(content=json.dumps(_MINIMAL_SPEC))
        check_structured_output_support(client, "model-name", EnvIntentSpec)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "model-name"
        assert kwargs["temperature"] == 0
        assert kwargs["response_format"]["type"] == "json_schema"
        assert kwargs["response_format"]["json_schema"]["name"] == "EnvIntentSpec"
        assert kwargs["response_format"]["json_schema"]["strict"] is True
        # The schema sent on the wire must already be munged for strict mode
        # — otherwise Bedrock rejects with 400. Spot-check the root marker.
        sent_schema = kwargs["response_format"]["json_schema"]["schema"]
        assert sent_schema["additionalProperties"] is False

    def test_accepts_alternative_spec_class(self):
        # Callers can probe with a smaller toy spec for cheap model
        # surveys — the probe shouldn't be hard-wired to EnvIntentSpec.
        class TinySpec(BaseModel):
            ok: bool

        client = MagicMock()
        client.chat.completions.create.return_value = _chat_response(content='{"ok": true}')
        result = check_structured_output_support(client, "m", TinySpec)
        assert result.supported is True
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"]["json_schema"]["name"] == "TinySpec"


# ---------------------------------------------------------------------------
# Live endpoint (opt-in, network + auth required)
# ---------------------------------------------------------------------------


@pytest.mark.agent_remote_e2e
def test_default_model_supports_structured_output():
    """The default ``EnvGenAgent`` model must support structured outputs.

    This is the gating contract of the whole agent: ``generate_spec``
    is structured-outputs-only, so the default
    ``DEFAULT_MODEL`` / ``DEFAULT_BASE_URL`` pair must pass the probe.
    Failing here means production env-gen is broken — usually because
    NVIDIA changed which channel DeepSeek-v4-flash routes structured
    outputs into, or pulled the model from the default-models
    catalogue.

    Asserts only ``supported=True``; the route may be either
    ``content`` (standard OpenAI) or ``reasoning_content`` (NVIDIA
    DeepSeek quirk) — both are handled transparently downstream.
    """
    api_key = os.environ.get("NV_API_KEY")
    assert api_key, "NV_API_KEY env var required to run live tests"

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL)
    result = check_structured_output_support(client, DEFAULT_MODEL, EnvIntentSpec)
    assert result.supported, (
        f"Default model {result.model!r} does not support structured outputs against "
        f"{DEFAULT_BASE_URL!r}: api_error={result.api_error!r} "
        f"parse_error={result.parse_error!r} route={result.response_route!r} "
        f"payload={result.sample_payload!r}"
    )
    assert result.response_route in {"content", "reasoning_content"}
