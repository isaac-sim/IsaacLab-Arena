# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise shared text/vision inference without making network requests."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, OpenAI

from isaaclab_arena.inference.backend import (
    InferenceBackend,
    InferenceBackendCfg,
    InferenceRequest,
    InferenceResponseError,
    StructuredOutputRequest,
)
from isaaclab_arena.tests.utils.agentic_environment_generation import chat_response


def _backend(stub_openai, **config):
    _, client = stub_openai
    backend = InferenceBackend(api_key="test-key", config=InferenceBackendCfg(**config))
    client.chat.completions.create.return_value = chat_response('{"kind": "wait", "steps": 2}')
    return backend, client


def _request(image=False):
    content = [{"type": "text", "text": "Choose the next command."}]
    if image:
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}})
    return InferenceRequest(
        messages=[{"role": "user", "content": content}],
        response_schema={"type": "object", "properties": {"kind": {"type": "string"}}},
    )


def test_multimodal_request_and_metadata(stub_openai):
    backend, client = _backend(stub_openai, supports_images=True)
    client.chat.completions.create.assert_not_called()
    client.chat.completions.create.return_value.usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    request = _request(image=True)
    original = deepcopy(request)
    response = backend.infer(request)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"] == request.messages
    assert kwargs["response_format"]["json_schema"]["schema"] == request.response_schema
    assert request == original
    assert response.data == {"kind": "wait", "steps": 2}
    assert response.usage["total_tokens"] == 15
    assert response.request_id == "test-request"
    assert response.model == "test-model"
    assert response.attempts == 1 and response.latency_s >= 0
    assert client.with_options.call_args.kwargs["max_retries"] == 0


def test_custom_endpoint_uses_named_credential_and_capabilities(stub_openai, monkeypatch):
    monkeypatch.setenv("ARENA_TEST_VLM_KEY", "custom-key")
    mock_cls, client = stub_openai
    backend = InferenceBackend(
        config=InferenceBackendCfg(
            endpoint="public",
            base_url="https://example.test/v1",
            model="my-vlm",
            api_key_env_var="ARENA_TEST_VLM_KEY",
            supports_temperature=False,
            max_tokens_parameter="max_completion_tokens",
        )
    )
    mock_cls.assert_called_once_with(
        api_key="custom-key",
        base_url="https://example.test/v1",
        max_retries=0,
        timeout=60.0,
    )
    client.chat.completions.create.return_value = chat_response("{}")
    backend.infer(_request())
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "my-vlm"
    assert kwargs["max_completion_tokens"] == 4096
    assert "temperature" not in kwargs and "max_tokens" not in kwargs


def test_images_require_explicit_capability(stub_openai):
    backend, client = _backend(stub_openai)
    with pytest.raises(AssertionError, match="supports_images"):
        backend.infer(_request(image=True))
    client.chat.completions.create.assert_not_called()


def test_json_object_mode_adds_schema_without_mutating_history(stub_openai):
    backend, client = _backend(stub_openai, response_format="json_object")
    request = _request()
    original = deepcopy(request)
    backend.infer(request)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert "Return JSON matching this schema" in kwargs["messages"][-1]["content"]
    assert request == original


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
def test_transient_http_failure_retries(stub_openai, status):
    backend, client = _backend(stub_openai, retry_backoff_s=0)
    error = APIStatusError(
        "temporary", response=httpx.Response(status, request=httpx.Request("POST", "https://example.test")), body=None
    )
    client.chat.completions.create.side_effect = [error, chat_response("{}")]
    assert backend.infer(_request()).attempts == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_http_failure_does_not_retry(stub_openai, status):
    backend, client = _backend(stub_openai)
    error = APIStatusError(
        "permanent", response=httpx.Response(status, request=httpx.Request("POST", "https://example.test")), body=None
    )
    client.chat.completions.create.side_effect = error
    with pytest.raises(APIStatusError):
        backend.infer(_request())
    assert client.chat.completions.create.call_count == 1


def test_connection_failures_exhaust_only_configured_attempts(stub_openai):
    backend, client = _backend(stub_openai, max_retries=1, retry_backoff_s=0)
    client.chat.completions.create.side_effect = APIConnectionError(
        request=httpx.Request("POST", "https://example.test")
    )
    with pytest.raises(APIConnectionError):
        backend.infer(_request())
    assert client.chat.completions.create.call_count == 2


def test_retry_budget_prevents_another_request(stub_openai):
    backend, client = _backend(stub_openai, retry_budget_s=1, retry_backoff_s=2)
    client.chat.completions.create.side_effect = APIConnectionError(
        request=httpx.Request("POST", "https://example.test")
    )
    with patch("isaaclab_arena.inference.backend.time.sleep") as sleep:
        with pytest.raises(TimeoutError, match="budget exhausted"):
            backend.infer(_request())
    sleep.assert_not_called()
    assert client.chat.completions.create.call_count == 1
    assert client.with_options.call_args.kwargs["timeout"] <= 1


@pytest.mark.parametrize("content", ["not JSON", "[]", "null", '{"value": NaN}', '{"value": Infinity}', ""])
def test_invalid_output_returns_to_caller_without_transport_retry(stub_openai, content):
    backend, client = _backend(stub_openai)
    client.chat.completions.create.return_value = chat_response(content)
    with pytest.raises(InferenceResponseError):
        backend.infer(_request())
    assert client.chat.completions.create.call_count == 1


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "tool_calls"])
def test_incomplete_output_is_not_a_command(stub_openai, finish_reason):
    backend, client = _backend(stub_openai)
    client.chat.completions.create.return_value = chat_response("{}", finish_reason=finish_reason)
    with pytest.raises(InferenceResponseError, match="unfinished"):
        backend.infer(_request())


def test_refusal_and_reasoning_are_not_commands(stub_openai):
    backend, client = _backend(stub_openai)
    response = chat_response(None, reasoning_content='{"kind": "wait", "steps": 2}')
    client.chat.completions.create.return_value = response
    with pytest.raises(InferenceResponseError, match="Empty"):
        backend.infer(_request())
    response.choices[0].message.content = "{}"
    response.choices[0].message.refusal = "Refused"
    with pytest.raises(InferenceResponseError, match="refused"):
        backend.infer(_request())


def test_legacy_import_and_text_api_share_the_backend(stub_openai):
    from isaaclab_arena.agentic_environment_generation.inference_backend import InferenceBackend as LegacyBackend

    assert LegacyBackend is InferenceBackend
    backend, client = _backend(stub_openai)
    result = backend.run_json(StructuredOutputRequest("test", {}, "system", "user", "test"))
    assert result == {"kind": "wait", "steps": 2}
    assert client.with_options.call_args.kwargs["max_retries"] == 0


def test_close_is_idempotent_and_prevents_reuse(stub_openai):
    backend, client = _backend(stub_openai)
    backend.close()
    backend.close()
    client.close.assert_called_once()
    with pytest.raises(AssertionError, match="closed"):
        backend.infer(_request())


@pytest.mark.parametrize(
    "config",
    [
        {"request_timeout_s": 0},
        {"retry_budget_s": float("inf")},
        {"retry_backoff_s": -1},
        {"max_retries": 10},
        {"max_tokens": 0},
        {"temperature": float("nan")},
        {"max_tokens_parameter": "unsupported"},
    ],
)
def test_invalid_backend_limits_are_rejected(config):
    with pytest.raises(AssertionError):
        InferenceBackendCfg(**config)


def test_real_sdk_serializes_both_apis_through_one_transport():
    payloads = []

    def handle(request):
        assert request.url == "https://example.test/v1/chat/completions"
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"x-request-id": "wire-request"},
            json={
                "id": "completion-1",
                "object": "chat.completion",
                "created": 0,
                "model": "test-vlm",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"steps": 2}',
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handle))
    sdk_client = OpenAI(api_key="test-key", base_url="https://example.test/v1", http_client=http_client)
    with patch("isaaclab_arena.inference.backend.OpenAI", return_value=sdk_client):
        backend = InferenceBackend(api_key="test-key", config=InferenceBackendCfg(supports_images=True))
        try:
            response = backend.infer(_request(image=True))
            result = backend.run_json(StructuredOutputRequest("test", {}, "system", "user", "test"))
            assert response.data == result == {"steps": 2}
            assert response.request_id == "wire-request"
            assert response.usage["total_tokens"] == 15
            assert payloads[0]["messages"][0]["content"][1]["type"] == "image_url"
            assert payloads[1]["messages"][0] == {"role": "system", "content": "system"}
        finally:
            backend.close()
    assert http_client.is_closed
