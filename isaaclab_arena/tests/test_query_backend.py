# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared structured-output query backend."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from isaaclab_arena.agentic_environment_generation.query_backend import QueryBackend, StructuredOutputRequest


def _chat_response(content: str | None = None, reasoning_content: str | None = None):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].message.reasoning_content = reasoning_content
    return resp


def test_run_json_parses_content_channel():
    client = MagicMock()
    client.chat.completions.create.return_value = _chat_response(content='{"ok": true}')
    backend = QueryBackend(client, "test-model")
    data = backend.run_json(
        StructuredOutputRequest(
            schema_name="TestSchema",
            schema={"type": "object"},
            system="sys",
            user="usr",
            retry_label="test",
        )
    )
    assert data == {"ok": True}


def test_run_json_parses_reasoning_content_channel():
    client = MagicMock()
    client.chat.completions.create.return_value = _chat_response(
        content=None,
        reasoning_content='{"ok": true}',
    )
    backend = QueryBackend(client, "test-model")
    data = backend.run_json(
        StructuredOutputRequest(
            schema_name="TestSchema",
            schema={"type": "object"},
            system="sys",
            user="usr",
            retry_label="test",
        )
    )
    assert data == {"ok": True}


def test_run_json_tolerates_unescaped_control_chars():
    client = MagicMock()
    client.chat.completions.create.return_value = _chat_response(content='{"name": "pick\tup"}')
    backend = QueryBackend(client, "test-model")
    data = backend.run_json(
        StructuredOutputRequest(
            schema_name="TestSchema",
            schema={"type": "object"},
            system="sys",
            user="usr",
            retry_label="test",
        )
    )
    assert data["name"] == "pick\tup"


def test_run_json_retries_then_succeeds():
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        ConnectionError("timeout"),
        _chat_response(content='{"ok": true}'),
    ]
    backend = QueryBackend(client, "test-model", max_retries=3)
    data = backend.run_json(
        StructuredOutputRequest(
            schema_name="TestSchema",
            schema={"type": "object"},
            system="sys",
            user="usr",
            retry_label="test",
        )
    )
    assert data == {"ok": True}
    assert client.chat.completions.create.call_count == 2


def test_run_json_raises_after_retries_exhausted():
    client = MagicMock()
    client.chat.completions.create.side_effect = ConnectionError("timeout")
    backend = QueryBackend(client, "test-model", max_retries=1)
    with pytest.raises(RuntimeError, match="failed test after 2 attempts"):
        backend.run_json(
            StructuredOutputRequest(
                schema_name="TestSchema",
                schema={"type": "object"},
                system="sys",
                user="usr",
                retry_label="test",
            )
        )
    assert client.chat.completions.create.call_count == 2


def test_run_json_forwards_inference_params_from_constructor():
    client = MagicMock()
    client.chat.completions.create.return_value = _chat_response(content=json.dumps({"ok": True}))
    backend = QueryBackend(client, "test-model", temperature=0.5, max_tokens=123, max_retries=0)
    backend.run_json(
        StructuredOutputRequest(
            schema_name="TestSchema",
            schema={"type": "object"},
            system="sys",
            user="usr",
            retry_label="test",
        )
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_tokens"] == 123
