# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from isaaclab_arena.agentic_environment_generation.environment_generation_agent import (
    AssetCatalogue,
    EnvironmentGenerationAgent,
    RelationCatalogue,
    TaskCatalogue,
)

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
    """Patch ``openai.OpenAI`` so ``EnvironmentGenerationAgent()`` never hits the wire."""
    with patch("isaaclab_arena.agentic_environment_generation.environment_generation_agent.OpenAI") as mock_cls:
        client = MagicMock()
        client.chat.completions.create.return_value = _chat_response(content="OK")
        mock_cls.return_value = client
        yield mock_cls


@pytest.fixture
def agent(stub_openai):
    """A constructed ``EnvironmentGenerationAgent`` with a fully mocked openai client."""
    a = EnvironmentGenerationAgent(api_key="test-key")
    a.client.chat.completions.create.side_effect = None
    a.client.chat.completions.create.reset_mock()
    return a


# Minimal EnvironmentIntentSpec payload — exercises every required field plus one
# task. Reused across the generate_spec happy-path tests.
_MINIMAL_SPEC: dict = {
    "reasoning": (
        "User wants a pick-and-place: foreground object is 'avocado', "
        "target container is 'bowl', background is the kitchen table."
    ),
    "background": "kitchen",
    "embodiment": "franka_ik",
    "items": [
        {"query": "avocado", "category_tags": [], "instance_name": None},
        {"query": "bowl", "category_tags": [], "instance_name": None},
    ],
    "initial_state_graph": [
        {"kind": "on", "subject": "avocado", "reference": "kitchen"},
        {"kind": "on", "subject": "bowl", "reference": "kitchen"},
    ],
    "tasks": [{
        "kind": "PickAndPlaceTask",
        "params": {
            "pick_up_object": "avocado",
            "destination_location": "bowl",
            "background_scene": "kitchen",
        },
        "description": "pick up the avocado and place it in the bowl",
    }],
}


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_explicit_api_key_overrides_env(self, monkeypatch, stub_openai):
        monkeypatch.setenv("NV_API_KEY", "env-key")
        a = EnvironmentGenerationAgent(api_key="explicit-key")
        assert a.api_key == "explicit-key"

    def test_falls_back_to_env_var(self, monkeypatch, stub_openai):
        monkeypatch.setenv("NV_API_KEY", "env-key")
        a = EnvironmentGenerationAgent()
        assert a.api_key == "env-key"

    def test_raises_when_no_key_anywhere(self, monkeypatch, stub_openai):
        monkeypatch.delenv("NV_API_KEY", raising=False)
        with pytest.raises(AssertionError, match="API key required"):
            EnvironmentGenerationAgent()

    def test_custom_model_and_base_url(self, stub_openai):
        a = EnvironmentGenerationAgent(api_key="k", model="custom-model", base_url="http://localhost:8000")
        assert a.model == "custom-model"
        stub_openai.assert_called_once_with(api_key="k", base_url="http://localhost:8000")


# ---------------------------------------------------------------------------
# generate_spec
# ---------------------------------------------------------------------------


def _catalog(text: str, relation_text: str = "RELATIONS (1):\n- on (binary): test") -> AssetCatalogue:
    catalogue = AssetCatalogue()
    catalogue.to_catalog_string = lambda: text  # type: ignore[method-assign]
    return catalogue


def _relation_catalog(text: str) -> RelationCatalogue:
    catalogue = RelationCatalogue()
    catalogue.to_catalog_string = lambda: text  # type: ignore[method-assign]
    return catalogue


def _task_catalog(text: str) -> TaskCatalogue:
    catalogue = TaskCatalogue()
    catalogue.to_catalog_string = lambda: text  # type: ignore[method-assign]
    return catalogue


class TestGenerateSpec:
    def test_builds_catalogues_from_singleton_registries_when_none(self, agent):
        agent.client.chat.completions.create.return_value = _chat_response(content=json.dumps(_MINIMAL_SPEC))
        with (
            patch(
                "isaaclab_arena.agentic_environment_generation.environment_generation_agent.build_asset_catalogue",
            ) as mock_build_assets,
            patch(
                "isaaclab_arena.agentic_environment_generation.environment_generation_agent.build_relation_catalogue",
            ) as mock_build_relations,
            patch(
                "isaaclab_arena.agentic_environment_generation.environment_generation_agent.build_task_catalogue",
            ) as mock_build_tasks,
        ):
            mock_build_assets.return_value = _catalog("<<ASSET-CATALOG>>")
            mock_build_relations.return_value = _relation_catalog("<<RELATION-CATALOG>>")
            mock_build_tasks.return_value = _task_catalog("<<TASK-CATALOG>>")
            agent.generate_spec("p")
        mock_build_assets.assert_called_once_with()
        mock_build_relations.assert_called_once_with()
        mock_build_tasks.assert_called_once_with()

    def test_request_sets_response_format_to_json_schema(self, agent):
        agent.client.chat.completions.create.return_value = _chat_response(content=json.dumps(_MINIMAL_SPEC))
        agent.generate_spec(
            "p",
            asset_catalog=_catalog("catalog"),
            relation_catalog=_relation_catalog("RELATIONS"),
            task_catalog=_task_catalog("TASKS"),
        )
        kwargs = agent.client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"]["type"] == "json_schema"
        assert kwargs["response_format"]["json_schema"]["name"] == "EnvironmentIntentSpec"
        assert kwargs["response_format"]["json_schema"]["strict"] is True
        # The schema sent on the wire is the cached, strict-mode-munged copy.
        assert kwargs["response_format"]["json_schema"]["schema"] is agent._spec_schema

    def test_tolerates_unescaped_control_chars(self, agent):
        # DeepSeek-v4-flash emits literal tab/newline characters inside JSON
        # strings despite the structured-outputs contract.
        payload = dict(_MINIMAL_SPEC)
        payload["reasoning"] = "pick up\tthe\tavocado"
        raw = json.dumps(payload).replace("\\t", "\t")
        assert "\t" in raw  # raw payload now has literal tab chars in a string
        agent.client.chat.completions.create.return_value = _chat_response(content=raw)
        spec, _ = agent.generate_spec(
            "p",
            asset_catalog=_catalog("catalog"),
            relation_catalog=_relation_catalog("RELATIONS"),
            task_catalog=_task_catalog("TASKS"),
        )
        assert "\t" in spec.reasoning

    def test_user_message_contains_catalog_and_prompt(self, agent):
        agent.client.chat.completions.create.return_value = _chat_response(content=json.dumps(_MINIMAL_SPEC))
        agent.generate_spec(
            "user wants avocado on kitchen",
            asset_catalog=_catalog("<<CATALOG-MARKER>>"),
            relation_catalog=_relation_catalog("<<RELATIONS-MARKER>>"),
            task_catalog=_task_catalog("<<TASKS-MARKER>>"),
        )
        msgs = agent.client.chat.completions.create.call_args.kwargs["messages"]
        assert [m["role"] for m in msgs] == ["system", "user"]
        user_msg = msgs[1]["content"]
        assert "<<CATALOG-MARKER>>" in user_msg
        assert "<<RELATIONS-MARKER>>" in user_msg
        assert "<<TASKS-MARKER>>" in user_msg
        assert "user wants avocado on kitchen" in user_msg

    def test_raises_when_response_has_no_choices(self, agent):
        resp = MagicMock()
        resp.choices = []
        agent.client.chat.completions.create.return_value = resp
        with pytest.raises(AssertionError, match="no choices"):
            agent.generate_spec(
                "p",
                asset_catalog=_catalog("catalog"),
                relation_catalog=_relation_catalog("RELATIONS"),
                task_catalog=_task_catalog("TASKS"),
            )


# ---------------------------------------------------------------------------
# Live endpoint (network + auth required)
# ---------------------------------------------------------------------------


# Marked flaky to absorb intermittent wire-level hiccups on the inference endpoint.
# TODO(qianl): drop the flaky marker once production-side retry is implemented.
@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_generate_spec_against_live_endpoint():
    """End-to-end smoke test against the real OpenAI-compatible endpoint."""
    agent = EnvironmentGenerationAgent()
    asset_catalog = _catalog(
        "EMBODIMENTS: franka_ik\n\n"
        "BACKGROUNDS: maple_table_kitchen\n\n"
        "OBJECTS (2):\n"
        "- avocado_robolab  tags=['vegetable']\n"
        "- bowl_robolab  tags=['container']"
    )
    task_catalog = _task_catalog(
        "TASKS (1):\n- PickAndPlaceTask (pick_up_object, destination_location, background_scene): Pick-and-place task."
    )
    spec, raw = agent.generate_spec(
        "pick up the avocado and place it in the bowl on the kitchen table",
        asset_catalog=asset_catalog,
        task_catalog=task_catalog,
    )
    assert isinstance(raw, str) and raw, "agent returned empty raw response"
    assert spec.tasks, "EnvironmentIntentSpec must contain at least one task"
    assert spec.background, "EnvironmentIntentSpec.background must be populated"
    assert spec.embodiment, "EnvironmentIntentSpec.embodiment must be populated"
    assert spec.reasoning, "EnvironmentIntentSpec.reasoning must be populated"
