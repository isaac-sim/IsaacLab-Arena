# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Utilities for the environment-generation agent (OpenAI-compatible API)."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel


def ping(client: Any, model: str) -> str:
    """Smoke-test the endpoint + API key + model with a minimal request.

    Args:
        client: An OpenAI-compatible client (typically
            ``openai.OpenAI`` or a compatible mock).
        model: Model identifier forwarded to
            ``client.chat.completions.create(model=...)``.

    Returns:
        The model's response text.
    """
    # TODO(qianl): wrap with transient-error retry.
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Respond with exactly: OK"}],
        temperature=0,
        max_tokens=8,
    )
    choices = getattr(resp, "choices", None) or []
    assert choices, (
        f"ping to model {model!r} returned HTTP 200 with no choices "
        "(content filter / guardrail / rate-limit response with empty body)."
    )
    return choices[0].message.content or ""


def build_strict_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Return ``model_cls``'s JSON schema munged for OpenAI strict mode."""
    schema = copy.deepcopy(model_cls.model_json_schema())
    apply_strict_constraints(schema)
    return schema


def apply_strict_constraints(node: Any) -> None:
    """Recursively apply OpenAI strict-mode constraints to a JSON-schema node."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        # Strict mode forbids ``default`` keys (every field is required, so
        # defaults can never apply). Drop them defensively at every level.
        node.pop("default", None)
        for v in node.values():
            apply_strict_constraints(v)
    elif isinstance(node, list):
        for v in node:
            apply_strict_constraints(v)


def extract_response_text(message: Any) -> tuple[str, str]:
    """Pull the agent's structured-output text from the chat-completion message.

    Returns ``(text, route)`` where ``route`` is one of:

      * ``"content"`` — the standard OpenAI-compatible channel.
      * ``"reasoning_content"`` — NVIDIA DeepSeek's provider-specific
        channel; the model emits structured outputs here instead of
        ``content``. We treat it as equivalent.
      * ``"empty"`` — both channels were empty / missing; the caller
        should surface a clear error.
    """
    content = getattr(message, "content", None)
    if content:
        return content, "content"
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        return reasoning, "reasoning_content"
    return "", "empty"
