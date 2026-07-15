# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`isaaclab_arena.agentic_environment_generation.inference_backend`."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from isaaclab_arena.agentic_environment_generation.inference_backend import (
    DEFAULT_BASE_URL,
    InferenceBackend,
    StructuredOutputRequest,
)
from isaaclab_arena.tests.utils.agentic_environment_generation import chat_response, inference_backend


def _request() -> StructuredOutputRequest:
    return StructuredOutputRequest(
        schema_name="TestSchema",
        schema={"type": "object", "properties": {}},
        system="system",
        user="user",
        retry_label="test",
    )


class TestInit:
    def test_explicit_api_key_overrides_env(self, monkeypatch, stub_openai):
        mock_cls, _ = stub_openai
        monkeypatch.setenv("NV_API_KEY", "env-key")
        InferenceBackend(api_key="explicit-key")
        mock_cls.assert_called_once_with(api_key="explicit-key", base_url=DEFAULT_BASE_URL)

    def test_falls_back_to_env_var(self, monkeypatch, stub_openai):
        mock_cls, _ = stub_openai
        monkeypatch.setenv("NV_API_KEY", "env-key")
        InferenceBackend()
        mock_cls.assert_called_once_with(api_key="env-key", base_url=DEFAULT_BASE_URL)

    def test_raises_when_no_key_anywhere(self, monkeypatch, stub_openai):
        monkeypatch.delenv("NV_API_KEY", raising=False)
        with pytest.raises(AssertionError, match="API key required"):
            InferenceBackend()

    def test_custom_model_and_base_url(self, stub_openai):
        mock_cls, _ = stub_openai
        backend = InferenceBackend(api_key="k", model="custom-model", base_url="http://localhost:8000")
        assert backend.model == "custom-model"
        mock_cls.assert_called_once_with(api_key="k", base_url="http://localhost:8000")


class TestRunJson:
    def test_tolerates_unescaped_control_chars(self, stub_openai):
        _, client = stub_openai
        backend = inference_backend(stub_openai)
        payload = {"env_name": "pick\tup"}
        raw = json.dumps(payload).replace("\\t", "\t")
        assert "\t" in raw
        client.chat.completions.create.return_value = chat_response(content=raw)
        result = backend.run_json(_request())
        assert "\t" in result["env_name"]

    def test_raises_when_response_has_no_choices(self, stub_openai):
        _, client = stub_openai
        backend = inference_backend(stub_openai)
        resp = MagicMock()
        resp.choices = []
        client.chat.completions.create.return_value = resp
        with pytest.raises(RuntimeError, match="failed test after 4 attempts"):
            backend.run_json(_request())
        assert client.chat.completions.create.call_count == 4

    def test_retries_after_api_error_then_succeeds(self, stub_openai):
        _, client = stub_openai
        backend = inference_backend(stub_openai)
        client.chat.completions.create.side_effect = [
            ConnectionError("timeout"),
            chat_response(content='{"ok": true}'),
        ]
        result = backend.run_json(_request())
        assert result == {"ok": True}
        assert client.chat.completions.create.call_count == 2

    def test_raises_after_api_errors_exhaust_retries(self, stub_openai):
        _, client = stub_openai
        backend = inference_backend(stub_openai, max_retries=1)
        client.chat.completions.create.side_effect = ConnectionError("timeout")
        with pytest.raises(RuntimeError, match="failed test after 2 attempts"):
            backend.run_json(_request())
        assert client.chat.completions.create.call_count == 2
