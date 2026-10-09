# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise shared text and image inference with mocked and live endpoints."""

import base64
import json
from copy import deepcopy
from io import BytesIO
from unittest.mock import patch

import httpx
import pytest
from openai import OpenAI
from PIL import Image

from isaaclab_arena.inference.backend import InferenceBackend, InferenceRequest, StructuredOutputRequest
from isaaclab_arena.tests.utils.agentic_environment_generation import (
    chat_response,
    inference_backend,
    skip_without_live_endpoint_key,
)


def _request():
    return InferenceRequest(
        messages=[
            {"role": "system", "content": "Choose robot commands."},
            {"role": "user", "content": "Wait until the path is clear."},
            {"role": "assistant", "content": '{"kind": "wait"}'},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Choose the next command."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                ],
            },
        ],
        response_schema={"type": "object", "properties": {"kind": {"type": "string"}}},
    )


def test_multimodal_request_preserves_images_schema_and_history(stub_openai):
    _, client = stub_openai
    backend = inference_backend(stub_openai)
    client.chat.completions.create.return_value = chat_response('{"kind": "wait"}')
    request = _request()
    original = deepcopy(request)

    assert backend.infer(request) == {"kind": "wait"}

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"] == original.messages
    assert kwargs["response_format"]["json_schema"] == {
        "name": request.schema_name,
        "schema": request.response_schema,
    }
    kwargs["messages"][-1]["content"][1]["image_url"]["url"] = "changed"
    assert request == original


@pytest.mark.parametrize("failure", [ConnectionError("timeout"), chat_response("invalid JSON")])
def test_multimodal_inference_retries_without_losing_images(stub_openai, failure):
    _, client = stub_openai
    backend = inference_backend(stub_openai, max_retries=1)
    client.chat.completions.create.side_effect = [failure, chat_response('{"input": {"kind": "wait"}}')]
    request = _request()

    assert backend.infer(request) == {"kind": "wait"}
    assert client.chat.completions.create.call_count == 2
    for call in client.chat.completions.create.call_args_list:
        assert call.kwargs["messages"] == request.messages


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
                    "message": {"role": "assistant", "content": '{"kind": "wait"}'},
                }],
            },
        )

    with OpenAI(
        api_key="test-key",
        base_url="https://example.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as sdk_client:
        with patch("isaaclab_arena.inference.backend.OpenAI", return_value=sdk_client):
            backend = InferenceBackend(api_key="test-key")
        payloads.clear()  # Discard the constructor's connection check.
        request = _request()
        response = backend.infer(request)
        result = backend.run_json(StructuredOutputRequest("test", {}, "system", "user", "test"))

    assert response == result == {"kind": "wait"}
    assert payloads[0]["messages"] == request.messages
    assert payloads[0]["response_format"]["json_schema"]["schema"] == request.response_schema
    assert payloads[1]["messages"] == [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}]


@skip_without_live_endpoint_key()
def test_multimodal_image_against_live_endpoint():
    image_buffer = BytesIO()
    Image.new("RGB", (64, 64), color="red").save(image_buffer, format="PNG")
    image_data = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    request = InferenceRequest(
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the dominant color of this image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
            ],
        }],
        response_schema={
            "type": "object",
            "properties": {"color": {"type": "string", "enum": ["red", "green", "blue"]}},
            "required": ["color"],
            "additionalProperties": False,
        },
        schema_name="image_color",
    )
    backend = InferenceBackend()
    try:
        assert backend.infer(request) == {"color": "red"}
    finally:
        backend.client.close()
