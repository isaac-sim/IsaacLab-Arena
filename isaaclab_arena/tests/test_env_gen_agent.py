# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`isaaclab_arena.environments.agentic_env_gen.env_gen_agent.EnvGenAgent`.

The agent's behaviour decomposes into three agent-level concerns that
we exercise without ever hitting the wire:

* ``__init__`` argument / env-var precedence, the missing-key guard,
  and the two constructor-time validations (``ping`` then
  ``check_structured_output_support``) that convert late wire /
  capability failures into fail-fast errors.
* ``generate_spec`` — the openai client is replaced with a
  ``MagicMock`` so we assert on the request shape (model, messages,
  ``response_format``, temperature, max_tokens) and the
  error-propagation contract.
* ``_system_prompt`` keeps its cross-cutting guidance intact;
  per-field schema details ride on the wire via
  ``response_format=json_schema`` rather than the prompt text.

Schema munging, the ``ping`` and ``check_structured_output_support``
helpers, and their failure-mode coverage all live in
:mod:`test_structured_output_utils`.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from isaaclab_arena.environments.agentic_env_gen.env_gen_agent import DEFAULT_BASE_URL, DEFAULT_MODEL, EnvGenAgent
from isaaclab_arena.environments.agentic_env_gen.structured_output_utils import apply_strict_constraints

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _chat_response(content: str | None = None, reasoning_content: str | None = None, finish_reason: str = "stop"):
    """Build a nested mock matching the openai chat-completion response shape.

    Models that route structured outputs into ``reasoning_content`` (e.g.
    NVIDIA DeepSeek) leave ``content`` empty — the fixture mirrors that by
    populating either channel independently.
    """
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].finish_reason = finish_reason
    resp.choices[0].message.content = content
    resp.choices[0].message.reasoning_content = reasoning_content
    return resp


@pytest.fixture
def stub_openai():
    """Patch ``openai.OpenAI`` so ``EnvGenAgent()`` never hits the wire.

    The agent does a deferred ``from openai import OpenAI`` inside
    ``__init__`` to avoid pulling the dependency at module import
    time, so we patch the symbol on the ``openai`` module itself.

    The patched client is pre-loaded to satisfy the two constructor
    probes (cheap ``ping`` then full structured-output check):
    ``side_effect`` returns a "OK" ping response then a
    ``_MINIMAL_SPEC`` probe response. Tests that want to assert on a
    failing ``__init__`` reach for ``patch("openai.OpenAI")``
    directly with a custom ``side_effect``.
    """
    with patch("openai.OpenAI") as mock_cls:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _chat_response(content="OK"),
            _chat_response(content=json.dumps(_MINIMAL_SPEC)),
        ]
        mock_cls.return_value = client
        yield mock_cls


@pytest.fixture
def agent(stub_openai):
    """A constructed ``EnvGenAgent`` with a fully mocked openai client.

    ``__init__``'s two calls (ping + structured-output probe) are
    served by ``stub_openai``'s pre-loaded ``side_effect``. After
    construction we *reset the mock* so per-test assertions on
    ``call_args`` / ``call_count`` start from a clean slate; tests
    can then set ``.return_value`` (or a fresh ``.side_effect``) to
    drive whichever method they're exercising.
    """
    a = EnvGenAgent(api_key="test-key")
    a.client.chat.completions.create.side_effect = None
    a.client.chat.completions.create.reset_mock()
    return a


# Minimal EnvIntentSpec payload — exercises every required field plus one
# task. Reused across the generate_spec happy-path tests.
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
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_explicit_api_key_overrides_env(self, monkeypatch, stub_openai):
        monkeypatch.setenv("NV_API_KEY", "env-key")
        a = EnvGenAgent(api_key="explicit-key")
        assert a.api_key == "explicit-key"

    def test_falls_back_to_env_var(self, monkeypatch, stub_openai):
        monkeypatch.setenv("NV_API_KEY", "env-key")
        a = EnvGenAgent()
        assert a.api_key == "env-key"

    def test_raises_when_no_key_anywhere(self, monkeypatch, stub_openai):
        monkeypatch.delenv("NV_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            EnvGenAgent()

    def test_default_model_and_base_url(self, stub_openai):
        a = EnvGenAgent(api_key="k")
        assert a.model == DEFAULT_MODEL
        stub_openai.assert_called_once_with(api_key="k", base_url=DEFAULT_BASE_URL)

    def test_custom_model_and_base_url(self, stub_openai):
        a = EnvGenAgent(api_key="k", model="custom-model", base_url="http://localhost:8000")
        assert a.model == "custom-model"
        stub_openai.assert_called_once_with(api_key="k", base_url="http://localhost:8000")

    def test_init_runs_ping_then_structured_output_probe(self, stub_openai):
        # ``__init__`` is contracted to run TWO wire checks in order:
        # (1) the cheap ``ping`` so a dead endpoint / bad key fails before
        # we spend tokens on (2) the heavier structured-output probe.
        # Asserting the order matters because reversing it would waste a
        # full schema probe on every misconfigured deployment.
        a = EnvGenAgent(api_key="k")
        assert a.client.chat.completions.create.call_count == 2
        first, second = a.client.chat.completions.create.call_args_list
        # First call = ping: small message, no response_format.
        assert first.kwargs["temperature"] == 0
        assert first.kwargs["max_tokens"] == 8
        assert len(first.kwargs["messages"]) == 1
        assert "response_format" not in first.kwargs
        # Second call = structured-output probe: carries the EnvIntentSpec
        # schema, signalling the model has to actually honour
        # ``response_format=json_schema``.
        assert second.kwargs["response_format"]["type"] == "json_schema"
        assert second.kwargs["response_format"]["json_schema"]["name"] == "EnvIntentSpec"

    def test_init_propagates_ping_failure(self):
        # If the openai client raises on the FIRST (ping) call — bad key,
        # unreachable endpoint, etc. — the exception must surface from
        # ``EnvGenAgent()`` itself, not be swallowed into a silently-broken
        # instance that fails later when generate_spec is called. The
        # structured-output probe must NOT be attempted (otherwise we'd
        # waste a schema-carrying request on a dead wire).
        class FakeAuthError(Exception):
            pass

        with patch("openai.OpenAI") as mock_cls:
            client = MagicMock()
            client.chat.completions.create.side_effect = FakeAuthError("bad key")
            mock_cls.return_value = client
            with pytest.raises(FakeAuthError, match="bad key"):
                EnvGenAgent(api_key="k")
            # Exactly one create() call — the ping. The probe never ran.
            assert client.chat.completions.create.call_count == 1

    def test_init_raises_when_structured_output_unsupported(self):
        # The agent is structured-outputs-only — a model that can't honour
        # ``response_format=json_schema`` is fundamentally unusable. The
        # constructor must refuse rather than letting downstream
        # ``generate_spec`` blow up later. ``check_structured_output_support``
        # raises the diagnostic RuntimeError directly, so all the
        # informative fields are baked into the probe's exception — no
        # caller-side message construction. This test just confirms the
        # probe's exception reaches the caller verbatim (no swallow,
        # no rewrap that drops fields).
        with patch("openai.OpenAI") as mock_cls:
            client = MagicMock()
            client.chat.completions.create.side_effect = [
                _chat_response(content="OK"),  # ping passes
                _chat_response(content=None, reasoning_content=None),  # probe empty
            ]
            mock_cls.return_value = client
            with pytest.raises(RuntimeError, match="does not support structured outputs") as exc_info:
                EnvGenAgent(api_key="k")
            msg = str(exc_info.value)
            # Diagnostic fields from the probe must reach the operator —
            # ``sample_payload`` in particular is what turns cryptic JSON /
            # validation errors into debuggable failures.
            assert "response_route" in msg
            assert "finish_reason" in msg
            assert "cause" in msg
            assert "sample_payload" in msg
            # The empty-envelope route signal — keeps callers able to
            # attribute "empty" vs "content" vs "reasoning_content".
            assert "'empty'" in msg

    def test_init_caches_strict_schema(self, stub_openai):
        # The strict schema munging walks ~10 nested object nodes; caching it
        # on the instance avoids redoing the walk on every generate_spec call.
        # The cached schema must already be munged — re-running the munger
        # should be a no-op (idempotent).
        a = EnvGenAgent(api_key="k")
        assert isinstance(a._spec_schema, dict)
        before = json.dumps(a._spec_schema, sort_keys=True)
        apply_strict_constraints(a._spec_schema)
        after = json.dumps(a._spec_schema, sort_keys=True)
        assert before == after


# ---------------------------------------------------------------------------
# generate_spec
# ---------------------------------------------------------------------------


class TestGenerateSpec:
    def test_happy_path_returns_spec_and_raw(self, agent):
        raw = json.dumps(_MINIMAL_SPEC)
        agent.client.chat.completions.create.return_value = _chat_response(content=raw)
        spec, returned_raw = agent.generate_spec("avocado on kitchen", catalog_text="catalog")
        assert spec.embodiment == "franka_ik"
        assert spec.background == "kitchen"
        assert len(spec.tasks) == 1
        assert returned_raw == raw

    def test_reads_from_reasoning_content_channel(self, agent):
        # DeepSeek quirk: when structured outputs are requested, the model
        # puts the JSON in ``reasoning_content`` instead of ``content``.
        raw = json.dumps(_MINIMAL_SPEC)
        agent.client.chat.completions.create.return_value = _chat_response(content=None, reasoning_content=raw)
        spec, returned_raw = agent.generate_spec("p", catalog_text="catalog")
        assert spec.embodiment == "franka_ik"
        assert returned_raw == raw

    def test_request_sets_response_format_to_json_schema(self, agent):
        agent.client.chat.completions.create.return_value = _chat_response(content=json.dumps(_MINIMAL_SPEC))
        agent.generate_spec("p", catalog_text="catalog")
        kwargs = agent.client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"]["type"] == "json_schema"
        assert kwargs["response_format"]["json_schema"]["name"] == "EnvIntentSpec"
        assert kwargs["response_format"]["json_schema"]["strict"] is True
        # The schema sent on the wire is the cached, strict-mode-munged copy.
        assert kwargs["response_format"]["json_schema"]["schema"] is agent._spec_schema

    def test_raises_runtime_error_on_empty_envelope(self, agent):
        # Both channels empty — the endpoint accepted ``response_format`` but
        # the model dropped the structured output (the canonical "endpoint
        # doesn't actually support structured outputs" failure mode).
        agent.client.chat.completions.create.return_value = _chat_response(content=None, reasoning_content=None)
        with pytest.raises(RuntimeError, match="empty structured-outputs envelope"):
            agent.generate_spec("p", catalog_text="catalog")

    def test_tolerates_unescaped_control_chars(self, agent):
        # DeepSeek-v4-flash emits literal tab/newline characters inside JSON
        # strings despite the structured-outputs contract. Python's default
        # ``json.loads`` rejects them; we pass ``strict=False`` to accept.
        payload = dict(_MINIMAL_SPEC)
        payload["task_description"] = "pick up\tthe\tavocado"
        raw = json.dumps(payload).replace("\\t", "\t")
        assert "\t" in raw  # raw payload now has literal tab chars in a string
        agent.client.chat.completions.create.return_value = _chat_response(content=raw)
        spec, _ = agent.generate_spec("p", catalog_text="catalog")
        assert "\t" in spec.task_description

    def test_propagates_validation_error_for_schema_violation(self, agent):
        # Well-formed JSON but missing every required EnvIntentSpec field —
        # pydantic surfaces this as a ``ValidationError`` distinct from a
        # transport or parse error.
        agent.client.chat.completions.create.return_value = _chat_response(content='{"missing": "fields"}')
        with pytest.raises(ValidationError):
            agent.generate_spec("p", catalog_text="catalog")

    def test_request_uses_configured_model(self, agent):
        agent.client.chat.completions.create.return_value = _chat_response(content=json.dumps(_MINIMAL_SPEC))
        agent.generate_spec("p", catalog_text="catalog")
        kwargs = agent.client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == agent.model

    def test_forwards_temperature_and_max_tokens(self, agent):
        agent.client.chat.completions.create.return_value = _chat_response(content=json.dumps(_MINIMAL_SPEC))
        agent.generate_spec("p", catalog_text="catalog", temperature=0.7, max_tokens=500)
        kwargs = agent.client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 500

    def test_user_message_contains_catalog_and_prompt(self, agent):
        agent.client.chat.completions.create.return_value = _chat_response(content=json.dumps(_MINIMAL_SPEC))
        agent.generate_spec("user wants avocado on kitchen", catalog_text="<<CATALOG-MARKER>>")
        msgs = agent.client.chat.completions.create.call_args.kwargs["messages"]
        assert [m["role"] for m in msgs] == ["system", "user"]
        user_msg = msgs[1]["content"]
        assert "<<CATALOG-MARKER>>" in user_msg
        assert "user wants avocado on kitchen" in user_msg
        # Under structured outputs the "emit ONLY JSON" instruction is
        # redundant (and was deliberately dropped) — the wire enforces
        # the envelope.
        assert "Return ONLY" not in user_msg


# ---------------------------------------------------------------------------
# _system_prompt
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_contains_cross_cutting_guidance(self, agent):
        # Under structured outputs the schema (including every Relation /
        # Task literal enum) flows to the model via ``response_format``.
        # The system prompt is reserved for cross-cutting rules that
        # can't be expressed in the schema — articulated-object anchoring,
        # distractor anchoring, anti-hallucination directives. Lock those
        # markers in so a future prompt rewrite can't accidentally drop
        # them.
        prompt = agent._system_prompt()
        for marker in (
            "Articulated objects",
            "Distractor items",
            "Do NOT hallucinate",
            "pick_and_place",
            "open_door",
            "close_door",
        ):
            assert marker in prompt, f"system prompt missing required marker {marker!r}"

    def test_does_not_repeat_response_format_instruction(self, agent):
        # Belt-and-suspenders: ensure the prompt isn't still telling the
        # model "emit ONLY JSON" — that instruction is redundant under
        # structured outputs and the wire enforces it.
        prompt = agent._system_prompt()
        assert "Emit ONLY" not in prompt
        assert "ONLY the JSON object" not in prompt


# ---------------------------------------------------------------------------
# Live endpoint (opt-in, network + auth required)
# ---------------------------------------------------------------------------


@pytest.mark.agent_remote_e2e
def test_generate_spec_against_live_endpoint():
    """End-to-end smoke test against the real OpenAI-compatible endpoint.

    Exercises the full structured-outputs pipeline with default
    ``model`` / ``base_url`` / system prompt:

        auth → HTTPS → response_format=json_schema → channel fallback
        → json.loads(strict=False) → EnvIntentSpec.model_validate

    Two layers gate this from default ``pytest`` runs:

      * ``agent_remote_e2e`` marker — registered in ``pytest.ini`` next to
        ``gr00t_remote_e2e``. Run explicitly with
        ``pytest -m agent_remote_e2e isaaclab_arena/tests/test_env_gen_agent.py``.

    The asset catalog is supplied inline rather than via ``AssetRegistry``
    so the test doesn't depend on Isaac Lab asset registration state — we
    only want to validate the agent wire here, not the catalog builder.

    The structured-outputs *capability* of the default model is
    pinned separately by
    :func:`test_structured_output_utils.test_default_model_supports_structured_output`;
    this test exercises the higher-level ``generate_spec`` pipeline
    end-to-end.

    Assertions are intentionally loose: we check shape (non-empty raw,
    non-empty tasks, populated background/embodiment, populated
    reasoning) rather than exact content, since agent output drifts
    between model versions.
    """
    agent = EnvGenAgent()
    catalog = (
        "EMBODIMENTS: franka_ik\n\n"
        "BACKGROUNDS: maple_table_kitchen\n\n"
        "OBJECTS (2):\n"
        "- avocado_robolab  tags=['vegetable']\n"
        "- bowl_robolab  tags=['container']"
    )
    spec, raw = agent.generate_spec(
        "pick up the avocado and place it in the bowl on the kitchen table",
        catalog_text=catalog,
    )
    assert isinstance(raw, str) and raw, "agent returned empty raw response"
    assert spec.tasks, "EnvIntentSpec must contain at least one task"
    assert spec.background, "EnvIntentSpec.background must be populated"
    assert spec.embodiment, "EnvIntentSpec.embodiment must be populated"
    # Structured outputs guarantee the forced-CoT reasoning field is
    # populated — under the old prose-extraction path it could come
    # back blank if the model wrapped the schema in markdown.
    assert spec.reasoning, "EnvIntentSpec.reasoning must be populated"
