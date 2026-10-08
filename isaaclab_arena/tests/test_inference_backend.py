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
    INFERENCE_ENDPOINT_ENV_VAR,
    INTERNAL_ENDPOINT,
    OPENAI_ENDPOINT,
    PUBLIC_ENDPOINT,
    InferenceBackend,
    StructuredOutputRequest,
    resolve_inference_endpoint,
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


def _spec_request() -> StructuredOutputRequest:
    """Request whose schema has real fields, so a wrapper key stands out from them."""
    return StructuredOutputRequest(
        schema_name="TestSchema",
        schema={"type": "object", "properties": {"env_name": {"type": "string"}, "objects": {"type": "array"}}},
        system="system",
        user="user",
        retry_label="test",
    )


@pytest.fixture(autouse=True)
def clean_endpoint_env(monkeypatch):
    """Keep the developer's endpoint selection out of endpoint-resolution tests."""
    monkeypatch.delenv(INFERENCE_ENDPOINT_ENV_VAR, raising=False)
    monkeypatch.delenv(INTERNAL_ENDPOINT.api_key_env_var, raising=False)
    monkeypatch.delenv(PUBLIC_ENDPOINT.api_key_env_var, raising=False)
    monkeypatch.delenv(OPENAI_ENDPOINT.api_key_env_var, raising=False)


class TestResolveInferenceEndpoint:
    def test_defaults_to_the_public_endpoint(self):
        assert resolve_inference_endpoint() == PUBLIC_ENDPOINT

    def test_reads_the_environment_variable(self, monkeypatch):
        monkeypatch.setenv(INFERENCE_ENDPOINT_ENV_VAR, INTERNAL_ENDPOINT.name)
        assert resolve_inference_endpoint() == INTERNAL_ENDPOINT

    def test_explicit_name_overrides_the_environment_variable(self, monkeypatch):
        monkeypatch.setenv(INFERENCE_ENDPOINT_ENV_VAR, INTERNAL_ENDPOINT.name)
        assert resolve_inference_endpoint(PUBLIC_ENDPOINT.name) == PUBLIC_ENDPOINT

    def test_raises_on_an_unknown_name(self):
        with pytest.raises(AssertionError, match="Unknown inference endpoint"):
            resolve_inference_endpoint("staging")


class TestInit:
    def test_explicit_api_key_overrides_env(self, monkeypatch, stub_openai):
        mock_cls, _ = stub_openai
        monkeypatch.setenv(PUBLIC_ENDPOINT.api_key_env_var, "env-key")
        InferenceBackend(api_key="explicit-key")
        mock_cls.assert_called_once_with(api_key="explicit-key", base_url=PUBLIC_ENDPOINT.base_url)

    def test_falls_back_to_env_var(self, monkeypatch, stub_openai):
        mock_cls, _ = stub_openai
        monkeypatch.setenv(PUBLIC_ENDPOINT.api_key_env_var, "env-key")
        backend = InferenceBackend()
        assert backend.model == PUBLIC_ENDPOINT.model
        mock_cls.assert_called_once_with(api_key="env-key", base_url=PUBLIC_ENDPOINT.base_url)

    def test_raises_when_no_key_anywhere(self, stub_openai):
        with pytest.raises(AssertionError, match=f"set {PUBLIC_ENDPOINT.api_key_env_var}"):
            InferenceBackend()

    def test_selected_endpoint_sets_base_url_model_and_key(self, monkeypatch, stub_openai):
        mock_cls, _ = stub_openai
        monkeypatch.setenv(INFERENCE_ENDPOINT_ENV_VAR, INTERNAL_ENDPOINT.name)
        monkeypatch.setenv(INTERNAL_ENDPOINT.api_key_env_var, "internal-key")
        backend = InferenceBackend()
        assert backend.endpoint == INTERNAL_ENDPOINT
        assert backend.model == INTERNAL_ENDPOINT.model
        mock_cls.assert_called_once_with(api_key="internal-key", base_url=INTERNAL_ENDPOINT.base_url)

    def test_endpoint_argument_overrides_env_var(self, monkeypatch, stub_openai):
        mock_cls, _ = stub_openai
        monkeypatch.setenv(INFERENCE_ENDPOINT_ENV_VAR, INTERNAL_ENDPOINT.name)
        monkeypatch.setenv(PUBLIC_ENDPOINT.api_key_env_var, "public-key")
        backend = InferenceBackend(endpoint=PUBLIC_ENDPOINT.name)
        assert backend.model == PUBLIC_ENDPOINT.model
        mock_cls.assert_called_once_with(api_key="public-key", base_url=PUBLIC_ENDPOINT.base_url)

    def test_key_of_the_other_endpoint_is_not_used(self, monkeypatch, stub_openai):
        monkeypatch.setenv(PUBLIC_ENDPOINT.api_key_env_var, "public-key")
        with pytest.raises(AssertionError, match=f"set {INTERNAL_ENDPOINT.api_key_env_var}"):
            InferenceBackend(endpoint=INTERNAL_ENDPOINT.name)

    def test_openai_endpoint_sets_base_url_model_and_key(self, monkeypatch, stub_openai):
        mock_cls, client = stub_openai
        monkeypatch.setenv(OPENAI_ENDPOINT.api_key_env_var, "openai-key")
        backend = InferenceBackend(endpoint=OPENAI_ENDPOINT.name)
        assert backend.endpoint == OPENAI_ENDPOINT
        assert backend.model == OPENAI_ENDPOINT.model
        mock_cls.assert_called_once_with(api_key="openai-key", base_url=OPENAI_ENDPOINT.base_url)
        ping_kwargs = client.chat.completions.create.call_args.kwargs
        assert ping_kwargs["max_completion_tokens"] == 32
        assert "max_tokens" not in ping_kwargs
        assert "temperature" not in ping_kwargs

    def test_custom_model_and_base_url(self, stub_openai):
        mock_cls, _ = stub_openai
        backend = InferenceBackend(api_key="k", model="custom-model", base_url="http://localhost:8000")
        assert backend.model == "custom-model"
        mock_cls.assert_called_once_with(api_key="k", base_url="http://localhost:8000")


class TestRunJson:
    def test_public_endpoint_uses_legacy_completion_options(self, stub_openai):
        _, client = stub_openai
        backend = inference_backend(stub_openai)
        client.chat.completions.create.return_value = chat_response(content='{"ok": true}')
        backend.run_json(_request())
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["max_tokens"] == 4096
        assert kwargs["temperature"] == 0.2
        assert "max_completion_tokens" not in kwargs

    def test_openai_endpoint_uses_openai_completion_options(self, stub_openai):
        _, client = stub_openai
        backend = InferenceBackend(api_key="test-key", endpoint=OPENAI_ENDPOINT.name)
        client.chat.completions.create.reset_mock()
        client.chat.completions.create.return_value = chat_response(content='{"ok": true}')
        backend.run_json(_request())
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["max_completion_tokens"] == 4096
        assert "max_tokens" not in kwargs
        assert "temperature" not in kwargs

    def test_internal_gpt_uses_openai_completion_options(self, stub_openai):
        _, client = stub_openai
        backend = InferenceBackend(api_key="test-key", endpoint=INTERNAL_ENDPOINT.name)
        client.chat.completions.create.reset_mock()
        client.chat.completions.create.return_value = chat_response(content='{"ok": true}')
        backend.run_json(_request())
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["max_completion_tokens"] == 4096
        assert "max_tokens" not in kwargs
        assert "temperature" not in kwargs

    def test_tolerates_unescaped_control_chars(self, stub_openai):
        _, client = stub_openai
        backend = inference_backend(stub_openai)
        payload = {"env_name": "pick\tup"}
        raw = json.dumps(payload).replace("\\t", "\t")
        assert "\t" in raw
        client.chat.completions.create.return_value = chat_response(content=raw)
        result = backend.run_json(_request())
        assert "\t" in result["env_name"]

    def test_unwraps_provider_envelope_around_payload(self, stub_openai):
        _, client = stub_openai
        backend = inference_backend(stub_openai)
        spec = {"env_name": "pick_and_place", "objects": []}
        client.chat.completions.create.return_value = chat_response(content=json.dumps({"input": spec}))
        assert backend.run_json(_spec_request()) == spec

    def test_keeps_single_field_response_declared_by_the_schema(self, stub_openai):
        _, client = stub_openai
        backend = inference_backend(stub_openai)
        payload = {"env_name": {"unexpected": "shape"}}
        client.chat.completions.create.return_value = chat_response(content=json.dumps(payload))
        assert backend.run_json(_spec_request()) == payload

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
