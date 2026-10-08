# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise the shared text/vision API without making network requests."""

import json
from copy import deepcopy
from unittest.mock import patch

import httpx
import pytest
from openai import OpenAI

from isaaclab_arena.inference.backend import (
    InferenceBackend,
    InferenceBackendCfg,
    InferenceRequest,
    StructuredOutputRequest,
)
from isaaclab_arena.tests.utils.agentic_environment_generation import chat_response


def _request():
    return InferenceRequest(
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Choose the next command."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ],
        }],
        response_schema={"type": "object", "properties": {"kind": {"type": "string"}}},
    )


def test_multimodal_request_preserves_images_schema_and_history(stub_openai):
    _, client = stub_openai
    backend = InferenceBackend(api_key="test-key", config=InferenceBackendCfg())
    client.chat.completions.create.assert_not_called()
    client.chat.completions.create.return_value = chat_response('{"kind": "wait"}')
    request = _request()
    original = deepcopy(request)
    response = backend.infer(request)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"] == request.messages
    assert kwargs["response_format"]["json_schema"]["schema"] == request.response_schema
    assert request == original
    assert response.data == {"kind": "wait"}


def test_custom_endpoint_uses_named_credential_and_timeout(stub_openai, monkeypatch):
    monkeypatch.setenv("ARENA_TEST_VLM_KEY", "custom-key")
    mock_cls, client = stub_openai
    backend = InferenceBackend(
        config=InferenceBackendCfg(
            endpoint="internal",
            base_url="https://example.test/v1",
            model="my-vlm",
            api_key_env_var="ARENA_TEST_VLM_KEY",
            request_timeout_s=15.0,
        )
    )
    mock_cls.assert_called_once_with(api_key="custom-key", base_url="https://example.test/v1", timeout=15.0)
    client.chat.completions.create.return_value = chat_response("{}")
    backend.infer(_request())
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "my-vlm"
    assert kwargs["max_completion_tokens"] == 4096
    assert "temperature" not in kwargs


def test_legacy_import_and_text_api_share_multimodal_inference(stub_openai):
    from isaaclab_arena.agentic_environment_generation.inference_backend import InferenceBackend as LegacyBackend

    assert LegacyBackend is InferenceBackend
    backend = InferenceBackend(api_key="test-key", config=InferenceBackendCfg())
    _, client = stub_openai
    client.chat.completions.create.return_value = chat_response('{"kind": "wait"}')
    with patch.object(backend, "infer", wraps=backend.infer) as infer:
        result = backend.run_json(StructuredOutputRequest("test", {}, "system", "user", "test-retry"))
    assert result == {"kind": "wait"}
    request = infer.call_args.args[0]
    assert request.messages == [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}]
    assert request.retry_label == "test-retry"


def test_multimodal_inference_retains_existing_json_retries(stub_openai):
    _, client = stub_openai
    backend = InferenceBackend(api_key="test-key", max_retries=1, config=InferenceBackendCfg())
    client.chat.completions.create.side_effect = [chat_response("invalid JSON"), chat_response('{"kind": "wait"}')]
    assert backend.infer(_request()).data == {"kind": "wait"}
    assert client.chat.completions.create.call_count == 2


def test_close_is_idempotent_and_prevents_reuse(stub_openai):
    _, client = stub_openai
    backend = InferenceBackend(api_key="test-key", config=InferenceBackendCfg())
    backend.close()
    backend.close()
    client.close.assert_called_once()
    with pytest.raises(AssertionError, match="closed"):
        backend.infer(_request())
    with pytest.raises(AssertionError, match="closed"):
        backend.run_json(StructuredOutputRequest("test", {}, "system", "user", "test"))


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_request_timeout_is_rejected(timeout):
    with pytest.raises(AssertionError):
        InferenceBackendCfg(request_timeout_s=timeout)


def test_real_sdk_serializes_text_and_images_through_one_transport():
    payloads = []

    def handle(request):
        assert request.url == "https://example.test/v1/chat/completions"
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "completion-1",
                "object": "chat.completion",
                "created": 0,
                "model": "test-vlm",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"steps": 2}'},
                }],
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handle))
    sdk_client = OpenAI(api_key="test-key", base_url="https://example.test/v1", http_client=http_client)
    with patch("isaaclab_arena.inference.backend.OpenAI", return_value=sdk_client):
        backend = InferenceBackend(api_key="test-key", config=InferenceBackendCfg())
        try:
            response = backend.infer(_request())
            result = backend.run_json(StructuredOutputRequest("test", {}, "system", "user", "test"))
            assert response.data == result == {"steps": 2}
            assert payloads[0]["messages"][0]["content"][1]["type"] == "image_url"
            assert payloads[1]["messages"][0] == {"role": "system", "content": "system"}
        finally:
            backend.close()
    assert http_client.is_closed
