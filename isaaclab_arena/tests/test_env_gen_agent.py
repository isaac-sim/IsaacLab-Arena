# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from isaaclab_arena.environments.agentic_env_gen.env_gen_agent import EnvGenAgent

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
    """Patch ``openai.OpenAI`` so ``EnvGenAgent()`` never hits the wire."""
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
    """A constructed ``EnvGenAgent`` with a fully mocked openai client."""
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
    "initial_state_graph": [
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

    def test_custom_model_and_base_url(self, stub_openai):
        a = EnvGenAgent(api_key="k", model="custom-model", base_url="http://localhost:8000")
        assert a.model == "custom-model"
        stub_openai.assert_called_once_with(api_key="k", base_url="http://localhost:8000")


# ---------------------------------------------------------------------------
# generate_spec
# ---------------------------------------------------------------------------


class TestGenerateSpec:
    def test_request_sets_response_format_to_json_schema(self, agent):
        agent.client.chat.completions.create.return_value = _chat_response(content=json.dumps(_MINIMAL_SPEC))
        agent.generate_spec("p", catalog_text="catalog")
        kwargs = agent.client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"]["type"] == "json_schema"
        assert kwargs["response_format"]["json_schema"]["name"] == "EnvIntentSpec"
        assert kwargs["response_format"]["json_schema"]["strict"] is True
        # The schema sent on the wire is the cached, strict-mode-munged copy.
        assert kwargs["response_format"]["json_schema"]["schema"] is agent._spec_schema

    def test_tolerates_unescaped_control_chars(self, agent):
        # DeepSeek-v4-flash emits literal tab/newline characters inside JSON
        # strings despite the structured-outputs contract.
        payload = dict(_MINIMAL_SPEC)
        payload["task_description"] = "pick up\tthe\tavocado"
        raw = json.dumps(payload).replace("\\t", "\t")
        assert "\t" in raw  # raw payload now has literal tab chars in a string
        agent.client.chat.completions.create.return_value = _chat_response(content=raw)
        spec, _ = agent.generate_spec("p", catalog_text="catalog")
        assert "\t" in spec.task_description

    def test_user_message_contains_catalog_and_prompt(self, agent):
        agent.client.chat.completions.create.return_value = _chat_response(content=json.dumps(_MINIMAL_SPEC))
        agent.generate_spec("user wants avocado on kitchen", catalog_text="<<CATALOG-MARKER>>")
        msgs = agent.client.chat.completions.create.call_args.kwargs["messages"]
        assert [m["role"] for m in msgs] == ["system", "user"]
        user_msg = msgs[1]["content"]
        assert "<<CATALOG-MARKER>>" in user_msg
        assert "user wants avocado on kitchen" in user_msg


# ---------------------------------------------------------------------------
# Live endpoint (network + auth required)
# ---------------------------------------------------------------------------


# Marked flaky to absorb intermittent wire-level hiccups on the inference endpoint.
# TODO(qianl): drop the flaky marker once production-side retry is implemented.
@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_generate_spec_against_live_endpoint():
    """End-to-end smoke test against the real OpenAI-compatible endpoint."""
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
    assert spec.reasoning, "EnvIntentSpec.reasoning must be populated"
