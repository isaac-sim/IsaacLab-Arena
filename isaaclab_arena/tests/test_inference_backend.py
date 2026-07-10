# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`isaaclab_arena.agentic_environment_generation.inference_backend`."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from isaaclab_arena.agentic_environment_generation.inference_backend import InferenceBackend, StructuredOutputRequest
from isaaclab_arena.tests.utils.agentic_environment_generation import chat_response


def _request() -> StructuredOutputRequest:
    return StructuredOutputRequest(
        schema_name="TestSchema",
        schema={"type": "object", "properties": {}},
        system="system",
        user="user",
        retry_label="test",
    )


def _backend(client: MagicMock, *, max_retries: int = 3) -> InferenceBackend:
    client.chat.completions.create.return_value = chat_response(content="OK")
    backend = InferenceBackend(client, "test-model", max_retries=max_retries)
    client.chat.completions.create.reset_mock()
    return backend


class TestRunJson:
    def test_tolerates_unescaped_control_chars(self):
        client = MagicMock()
        backend = _backend(client)
        payload = {"env_name": "pick\tup"}
        raw = json.dumps(payload).replace("\\t", "\t")
        assert "\t" in raw
        client.chat.completions.create.return_value = chat_response(content=raw)
        result = backend.run_json(_request())
        assert "\t" in result["env_name"]

    def test_raises_when_response_has_no_choices(self):
        client = MagicMock()
        backend = _backend(client)
        resp = MagicMock()
        resp.choices = []
        client.chat.completions.create.return_value = resp
        with pytest.raises(RuntimeError, match="failed test after 4 attempts"):
            backend.run_json(_request())
        assert client.chat.completions.create.call_count == 4

    def test_retries_after_api_error_then_succeeds(self):
        client = MagicMock()
        backend = _backend(client)
        client.chat.completions.create.side_effect = [
            ConnectionError("timeout"),
            chat_response(content='{"ok": true}'),
        ]
        result = backend.run_json(_request())
        assert result == {"ok": True}
        assert client.chat.completions.create.call_count == 2

    def test_raises_after_api_errors_exhaust_retries(self):
        client = MagicMock()
        backend = _backend(client, max_retries=1)
        client.chat.completions.create.side_effect = ConnectionError("timeout")
        with pytest.raises(RuntimeError, match="failed test after 2 attempts"):
            backend.run_json(_request())
        assert client.chat.completions.create.call_count == 2
