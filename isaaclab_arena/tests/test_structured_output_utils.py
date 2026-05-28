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


# Minimal EnvIntentSpec payload exercising every required field plus one
# task — reused across the ``check_structured_output_support`` happy-path
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

    def test_raises_runtime_error_when_choices_list_is_empty(self):
        # Real provider behaviour: HTTP returns 200 OK but ``choices`` is an
        # empty list. Seen on Azure when a content-filter trips, on Bedrock
        # when a guardrail rejects post-hoc, and on certain rate-limit
        # responses. Naive ``resp.choices[0]`` would raise ``IndexError`` —
        # an opaque crash that breaks the documented ``Raises`` contract.
        # The function must instead surface a structured ``RuntimeError``
        # so callers (notably ``EnvGenAgent.__init__``) see a diagnosable
        # ping failure with the model name baked in.
        resp = MagicMock()
        resp.choices = []
        client = MagicMock()
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
    """Bool-or-raise contract: returns True on a clean round-trip, raises
    ``RuntimeError`` with a multi-line diagnostic on every failure mode.

    Each failure-mode test pins three things:
      1. ``RuntimeError`` (not the original SDK exception) reaches the
         caller — so callers have a single exception type to catch.
      2. The model name appears in the message (the most-grepped field).
      3. The ``cause`` field carries the upstream classifier
         (``BadRequestError`` vs ``JSONDecodeError`` vs ``ValidationError``)
         so the failure attribution survives the wrapping.

    Where the underlying SDK / parser exception is preserved on
    ``__cause__``, we assert that too — it's what makes
    ``raise ... from exc`` worth doing.
    """

    def test_returns_true_on_valid_envelope(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _chat_response(content=json.dumps(_MINIMAL_SPEC))
        # The whole public contract collapses to: ``True`` or it raises.
        # ``is True`` rather than truthy so a future regression that
        # returns a dict/tuple/etc fails this test.
        assert check_structured_output_support(client, "some-model", EnvIntentSpec) is True

    def test_returns_true_on_reasoning_content_envelope(self):
        # NVIDIA DeepSeek envelope — content empty, structured output
        # on the ``reasoning_content`` channel. Must NOT raise; the
        # ``extract_response_text`` fallback handles this transparently.
        # The previous dataclass surfaced ``response_route`` so callers
        # could distinguish; the new API hides that detail (callers
        # don't need it — both channels are equivalent for our purposes).
        client = MagicMock()
        client.chat.completions.create.return_value = _chat_response(
            content=None, reasoning_content=json.dumps(_MINIMAL_SPEC)
        )
        assert check_structured_output_support(client, "deepseek", EnvIntentSpec) is True

    def test_raises_on_4xx_with_underlying_exception_chained(self):
        # The most common "model doesn't support structured outputs"
        # signal at the wire level: a 4xx rejecting ``response_format``
        # or the schema. The original SDK exception must reach the
        # caller via ``__cause__`` so the traceback retains the HTTP
        # status / body — otherwise debugging "why did construction
        # fail?" requires re-running locally.
        class FakeBadRequest(Exception):
            pass

        client = MagicMock()
        original = FakeBadRequest("Error code: 400 - additionalProperties")
        client.chat.completions.create.side_effect = original
        with pytest.raises(RuntimeError, match="does not support structured outputs") as exc_info:
            check_structured_output_support(client, "claude", EnvIntentSpec)
        msg = str(exc_info.value)
        # Model name surfaces (most-grepped field) and the cause type
        # classifies the failure (4xx wire error, not parse / validation).
        assert "'claude'" in msg
        assert "FakeBadRequest" in msg
        assert "400" in msg
        # On an api_error there's no response payload to echo.
        assert "sample_payload = None" in msg
        # Exception chaining preserves the original for traceback drill-down.
        assert exc_info.value.__cause__ is original

    def test_raises_on_empty_envelope(self):
        # Wire accepts the request, model produces nothing on either
        # channel. The endpoint silently dropped the structured output
        # — the most insidious failure mode, since ``finish_reason``
        # still reads ``stop``. No underlying exception to chain.
        client = MagicMock()
        client.chat.completions.create.return_value = _chat_response(content=None, reasoning_content=None)
        with pytest.raises(RuntimeError, match="does not support structured outputs") as exc_info:
            check_structured_output_support(client, "broken", EnvIntentSpec)
        msg = str(exc_info.value)
        assert "empty envelope" in msg
        # finish_reason forwarded so the operator can correlate with
        # provider logs (was it a content-filter stop, a length cap, etc.).
        assert "finish_reason  = 'stop'" in msg
        # No upstream exception to chain on this branch (the function
        # itself synthesises the failure from a structurally-OK response).
        assert exc_info.value.__cause__ is None

    def test_raises_when_choices_list_is_empty(self):
        # Real provider behaviour: HTTP returns 200 OK but ``choices`` is
        # an empty list. Seen on Azure when a content-filter trips, and
        # on Bedrock when a guardrail rejects the response post-hoc.
        # Naive ``resp.choices[0]`` access would IndexError and break
        # the contract — surface it as a structured RuntimeError with
        # a distinct ``cause`` message that operators can tell apart
        # from the "envelope returned but content empty" case.
        resp = MagicMock()
        resp.choices = []
        client = MagicMock()
        client.chat.completions.create.return_value = resp
        with pytest.raises(RuntimeError, match="does not support structured outputs") as exc_info:
            check_structured_output_support(client, "guardrailed", EnvIntentSpec)
        msg = str(exc_info.value)
        assert "no choices" in msg
        assert "response_route = 'empty'" in msg

    def test_raises_on_invalid_json_with_payload_preview(self):
        # The JSON-decode failure is the case where ``sample_payload``
        # earns its keep — without it the operator sees only
        # "Expecting value: line 1 column 1" and has to re-run locally
        # to discover the model emitted a prose preamble. With the
        # preview in the message the failure is debuggable from CI logs.
        client = MagicMock()
        client.chat.completions.create.return_value = _chat_response(content="not json")
        with pytest.raises(RuntimeError, match="does not support structured outputs") as exc_info:
            check_structured_output_support(client, "m", EnvIntentSpec)
        msg = str(exc_info.value)
        assert "JSONDecodeError" in msg
        assert "'not json'" in msg  # the literal response preview
        # Original JSONDecodeError preserved on ``__cause__``.
        assert exc_info.value.__cause__ is not None
        assert type(exc_info.value.__cause__).__name__ == "JSONDecodeError"

    def test_raises_on_validation_failure_with_payload_preview(self):
        # JSON parses fine, but doesn't match the schema. The probe
        # exists to detect this exact class of "model returns
        # something, but it's wrong" failure. The original
        # ValidationError chains via ``__cause__`` so ``.errors()``
        # is still reachable for callers that want the structured
        # error list.
        from pydantic import ValidationError

        client = MagicMock()
        client.chat.completions.create.return_value = _chat_response(content='{"missing": "fields"}')
        with pytest.raises(RuntimeError, match="does not support structured outputs") as exc_info:
            check_structured_output_support(client, "m", EnvIntentSpec)
        msg = str(exc_info.value)
        assert "ValidationError" in msg
        assert '{"missing": "fields"}' in msg  # payload preview echoed
        assert isinstance(exc_info.value.__cause__, ValidationError)

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
        assert check_structured_output_support(client, "m", TinySpec) is True
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"]["json_schema"]["name"] == "TinySpec"


# ---------------------------------------------------------------------------
# Live endpoint (opt-in, network + auth required)
# ---------------------------------------------------------------------------


# The probe hits a real model on every run. NVIDIA's hosted DeepSeek-v4-flash
# is intermittently quirky under structured outputs (occasional blank
# ``content``, transient 429 / 5xx from the proxy, etc.); a single failed
# attempt does NOT mean the deployment is actually broken. Allow up to 2
# reruns so a transient blip doesn't fail CI. Real breakage will fail all 3.
# TODO(qianl): drop the flaky marker once production-side retry is wired
# into ``check_structured_output_support`` (see TODO in structured_output_utils.py).
@pytest.mark.flaky(max_runs=3, min_passes=1)
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

    The probe's ``RuntimeError`` already carries a multi-line
    diagnostic (model / route / finish_reason / cause /
    sample_payload), so test-failure output is self-describing — no
    extra error-message construction needed here.
    """
    api_key = os.environ.get("NV_API_KEY")
    assert api_key, "NV_API_KEY env var required to run live tests"

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL)
    assert check_structured_output_support(client, DEFAULT_MODEL, EnvIntentSpec) is True
