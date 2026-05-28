# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`isaaclab_arena.environments.agentic_env_gen.llm_agent.LLMAgent`.

The agent's behaviour decomposes into four pure-Python concerns that we exercise
without ever hitting the wire:

* ``__init__`` argument / env-var precedence and the missing-key guard.
* ``_extract_json`` parsing of well-behaved, fenced, prosed, and malformed
  LLM responses (including the ``LLMResponseParseError`` → ``ValueError`` MRO
  contract so callers can still ``except ValueError``).
* ``generate_spec`` / ``ping`` — the openai client is replaced with a
  ``MagicMock`` so we assert on the request shape (model, messages,
  temperature, max_tokens) and the error-propagation contract.
* ``_system_prompt`` is asserted to enumerate every ``RelationKind`` /
  ``TaskKind`` literal so prompt and schema cannot drift apart silently.
"""

from __future__ import annotations

import json
from typing import get_args
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from isaaclab_arena.environments.agentic_env_gen.llm_agent import (
    _RAW_RESPONSE_PREVIEW_CHARS,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LLMAgent,
    LLMResponseParseError,
)
from isaaclab_arena.environments.agentic_env_gen.llm_schema import RelationKind, TaskKind

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_openai():
    """Patch the ``openai.OpenAI`` constructor so ``LLMAgent()`` never hits the wire.

    The agent does a deferred ``from openai import OpenAI`` inside
    ``__init__`` to avoid pulling the dependency at module import time, so we
    patch the symbol on the ``openai`` module itself rather than on the
    ``llm_agent`` namespace.
    """
    with patch("openai.OpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        yield mock_cls


@pytest.fixture
def agent(stub_openai):
    """A constructed ``LLMAgent`` with a fully mocked openai client.

    Tests should set ``agent.client.chat.completions.create.return_value`` (or
    ``.side_effect``) to control the simulated LLM response.
    """
    return LLMAgent(api_key="test-key")


def _chat_response(content: str | None):
    """Build the nested mock that mimics the openai chat-completion response shape."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


# Minimal LLMEnvSpec payload — exercises every required field plus one task so
# the ``tasks_must_be_non_empty`` validator passes. Reused across the
# generate_spec happy-path tests.
_MINIMAL_SPEC: dict = {
    "reasoning": (
        "User wants a pick-and-place: foreground object is 'avocado', "
        "target container is 'bowl', background is the kitchen table."
    ),
    "task_description": "pick up the avocado and place it in the bowl",
    "background": "kitchen",
    "embodiment": "franka_ik",
    "items": [
        {"query": "avocado", "role": "foreground", "category_tags": []},
        {"query": "bowl", "role": "foreground", "category_tags": []},
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
        a = LLMAgent(api_key="explicit-key")
        assert a.api_key == "explicit-key"

    def test_falls_back_to_env_var(self, monkeypatch, stub_openai):
        monkeypatch.setenv("NV_API_KEY", "env-key")
        a = LLMAgent()
        assert a.api_key == "env-key"

    def test_raises_when_no_key_anywhere(self, monkeypatch, stub_openai):
        monkeypatch.delenv("NV_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            LLMAgent()

    def test_default_model_and_base_url(self, stub_openai):
        a = LLMAgent(api_key="k")
        assert a.model == DEFAULT_MODEL
        stub_openai.assert_called_once_with(api_key="k", base_url=DEFAULT_BASE_URL)

    def test_custom_model_and_base_url(self, stub_openai):
        a = LLMAgent(api_key="k", model="custom-model", base_url="http://localhost:8000")
        assert a.model == "custom-model"
        stub_openai.assert_called_once_with(api_key="k", base_url="http://localhost:8000")

    def test_init_pings_to_verify_connection(self, stub_openai):
        # ``__init__`` is contracted to run a ping round-trip before returning
        # so a bad key / wrong model / dead endpoint fails at construction time
        # rather than deep inside the first generate_spec. Locking in the
        # request shape (single user message, max_tokens=8, temperature=0)
        # guarantees we don't accidentally inflate the startup cost.
        a = LLMAgent(api_key="k")
        a.client.chat.completions.create.assert_called_once()
        kwargs = a.client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0
        assert kwargs["max_tokens"] == 8
        assert len(kwargs["messages"]) == 1

    def test_init_propagates_ping_failure(self):
        # If the openai client raises during the constructor ping (bad key,
        # unreachable endpoint, ...), the exception must surface from
        # ``LLMAgent()`` itself — not be swallowed into a silently-broken
        # instance that fails later when generate_spec is called.
        class FakeAuthError(Exception):
            pass

        with patch("openai.OpenAI") as mock_cls:
            client = MagicMock()
            client.chat.completions.create.side_effect = FakeAuthError("bad key")
            mock_cls.return_value = client
            with pytest.raises(FakeAuthError, match="bad key"):
                LLMAgent(api_key="k")


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_json_object(self):
        assert LLMAgent._extract_json('{"a": 1}') == {"a": 1}

    def test_strips_fenced_json_block(self):
        assert LLMAgent._extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_strips_bare_triple_backticks(self):
        assert LLMAgent._extract_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_extracts_object_from_prose(self):
        text = 'Sure! Here is the JSON: {"a": 1} -- hope that helps.'
        assert LLMAgent._extract_json(text) == {"a": 1}

    def test_handles_nested_braces(self):
        text = 'prefix {"outer": {"inner": [1, 2, 3]}} suffix'
        assert LLMAgent._extract_json(text) == {"outer": {"inner": [1, 2, 3]}}

    def test_raises_when_no_opening_brace(self):
        with pytest.raises(LLMResponseParseError, match="No JSON object found"):
            LLMAgent._extract_json("plain text with no braces at all")

    def test_raises_on_unbalanced_braces(self):
        with pytest.raises(LLMResponseParseError, match="Unbalanced braces"):
            LLMAgent._extract_json('prefix {"a": 1 with no closing brace')

    def test_parse_error_is_a_value_error(self):
        # MRO contract: LLMResponseParseError subclasses ValueError so existing
        # ``except ValueError`` clauses (e.g. wrapping model_validate) still
        # catch parse failures. Asserting via ``except ValueError`` rather than
        # ``issubclass`` keeps the test grounded in how callers actually use it.
        with pytest.raises(ValueError):
            LLMAgent._extract_json("no braces here")

    def test_truncates_long_raw_response_in_error(self):
        # Confirm the preview cap really clips the embedded raw response —
        # otherwise a megabyte-scale LLM hallucination would bury the
        # stack trace. We allow a small wrapper budget for the surrounding
        # error message (repr quotes + "No JSON object found in ..." prefix).
        huge = "x" * 5000
        with pytest.raises(LLMResponseParseError) as exc_info:
            LLMAgent._extract_json(huge)
        msg = str(exc_info.value)
        wrapper_budget = 200
        assert len(msg) <= _RAW_RESPONSE_PREVIEW_CHARS + wrapper_budget
        # ...and a 4000-char run from deep inside ``huge`` must not have leaked.
        assert "x" * 4000 not in msg


# ---------------------------------------------------------------------------
# generate_spec
# ---------------------------------------------------------------------------


class TestGenerateSpec:
    def test_happy_path_returns_spec_and_raw(self, agent):
        raw = json.dumps(_MINIMAL_SPEC)
        agent.client.chat.completions.create.return_value = _chat_response(raw)
        spec, returned_raw = agent.generate_spec("avocado on kitchen", catalog_text="catalog")
        assert spec.embodiment == "franka_ik"
        assert spec.background == "kitchen"
        assert len(spec.tasks) == 1
        assert returned_raw == raw

    def test_handles_fenced_response(self, agent):
        raw = f"```json\n{json.dumps(_MINIMAL_SPEC)}\n```"
        agent.client.chat.completions.create.return_value = _chat_response(raw)
        spec, _ = agent.generate_spec("p", catalog_text="catalog")
        assert spec.embodiment == "franka_ik"

    def test_propagates_parse_error_for_garbage_response(self, agent):
        agent.client.chat.completions.create.return_value = _chat_response("not json at all")
        with pytest.raises(LLMResponseParseError):
            agent.generate_spec("p", catalog_text="catalog")

    def test_propagates_validation_error_for_schema_violation(self, agent):
        # Well-formed JSON but missing every required LLMEnvSpec field — pydantic
        # surfaces this as a ``ValidationError`` distinct from a parse error.
        agent.client.chat.completions.create.return_value = _chat_response('{"missing": "fields"}')
        with pytest.raises(ValidationError):
            agent.generate_spec("p", catalog_text="catalog")

    def test_request_uses_configured_model(self, agent):
        agent.client.chat.completions.create.return_value = _chat_response(json.dumps(_MINIMAL_SPEC))
        agent.generate_spec("p", catalog_text="catalog")
        kwargs = agent.client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == agent.model

    def test_forwards_temperature_and_max_tokens(self, agent):
        agent.client.chat.completions.create.return_value = _chat_response(json.dumps(_MINIMAL_SPEC))
        agent.generate_spec("p", catalog_text="catalog", temperature=0.7, max_tokens=500)
        kwargs = agent.client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 500

    def test_user_message_contains_catalog_and_prompt(self, agent):
        agent.client.chat.completions.create.return_value = _chat_response(json.dumps(_MINIMAL_SPEC))
        agent.generate_spec("user wants avocado on kitchen", catalog_text="<<CATALOG-MARKER>>")
        msgs = agent.client.chat.completions.create.call_args.kwargs["messages"]
        assert [m["role"] for m in msgs] == ["system", "user"]
        user_msg = msgs[1]["content"]
        assert "<<CATALOG-MARKER>>" in user_msg
        assert "user wants avocado on kitchen" in user_msg
        # The "JSON-only" instruction is the contract that lets _extract_json
        # work — if it disappears the LLM tends to wrap in prose.
        assert "JSON" in user_msg


# ---------------------------------------------------------------------------
# ping
# ---------------------------------------------------------------------------


class TestPing:
    def test_returns_response_content(self, agent):
        agent.client.chat.completions.create.return_value = _chat_response("OK")
        assert agent.ping() == "OK"

    def test_returns_empty_string_when_content_is_none(self, agent):
        # Some providers return ``None`` content alongside a finish_reason — we
        # treat that as a successful round-trip (the wire works) rather than
        # raising, since the caller's contract is "did this raise?".
        agent.client.chat.completions.create.return_value = _chat_response(None)
        assert agent.ping() == ""

    def test_uses_minimal_request_params(self, agent):
        agent.client.chat.completions.create.return_value = _chat_response("OK")
        agent.ping()
        kwargs = agent.client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == agent.model
        assert kwargs["temperature"] == 0
        assert kwargs["max_tokens"] == 8
        # Single user message — no system prompt / catalog payload. Keeping the
        # request small is the whole point: ping must stay cheap enough to run
        # on every CI job startup.
        assert len(kwargs["messages"]) == 1
        assert kwargs["messages"][0]["role"] == "user"

    def test_propagates_client_exceptions(self, agent):
        class FakeAuthError(Exception):
            pass

        agent.client.chat.completions.create.side_effect = FakeAuthError("invalid api key")
        with pytest.raises(FakeAuthError, match="invalid api key"):
            agent.ping()


# ---------------------------------------------------------------------------
# _system_prompt
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_enumerates_every_relation_kind(self, agent):
        # The prompt derives its bullet list from ``get_args(RelationKind)``;
        # this assertion fails the moment someone adds a kind to the literal
        # without rebuilding the prompt, which would silently teach the LLM a
        # vocabulary the resolver doesn't accept.
        prompt = agent._system_prompt()
        for kind in get_args(RelationKind):
            assert kind in prompt, f"relation kind {kind!r} missing from system prompt"

    def test_enumerates_every_task_kind(self, agent):
        # Task kinds are quoted in the prompt (JSON-style) to disambiguate from
        # surrounding prose — keep the quoting in sync with the source.
        prompt = agent._system_prompt()
        for kind in get_args(TaskKind):
            assert f'"{kind}"' in prompt, f"task kind {kind!r} missing from system prompt"

    def test_embeds_llm_env_spec_schema(self, agent):
        # We assert on field names rather than diffing the full JSON schema so
        # the test isn't brittle to pydantic's schema-generation tweaks across
        # versions.
        prompt = agent._system_prompt()
        for field in (
            "reasoning",
            "task_description",
            "background",
            "embodiment",
            "items",
            "initial_scene_graph",
            "tasks",
        ):
            assert field in prompt


# ---------------------------------------------------------------------------
# Live endpoint (opt-in, network + auth required)
# ---------------------------------------------------------------------------


@pytest.mark.llm_remote_e2e
def test_generate_spec_against_live_endpoint():
    """End-to-end smoke test against the real OpenAI-compatible endpoint.

    Exercises the full pipeline with default ``model`` / ``base_url`` /
    system prompt:

        auth → HTTPS → model response → JSON extract → LLMEnvSpec validation

    Two layers gate this from default ``pytest`` runs:

      * ``llm_remote_e2e`` marker — registered in ``pytest.ini`` next to
        ``gr00t_remote_e2e``. Run explicitly with
        ``pytest -m llm_remote_e2e isaaclab_arena/tests/test_llm_agent.py``.

    The asset catalog is supplied inline rather than via ``AssetRegistry``
    so the test doesn't depend on Isaac Lab asset registration state — we
    only want to validate the LLM wire here, not the catalog builder.

    Assertions are intentionally loose: we check shape (non-empty raw,
    non-empty tasks, populated background/embodiment) rather than exact
    content, since LLM output drifts between model versions.
    """
    agent = LLMAgent()
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
    assert isinstance(raw, str) and raw, "LLM returned empty raw response"
    assert spec.tasks, "LLMEnvSpec must contain at least one task"
    assert spec.background, "LLMEnvSpec.background must be populated"
    assert spec.embodiment, "LLMEnvSpec.embodiment must be populated"
