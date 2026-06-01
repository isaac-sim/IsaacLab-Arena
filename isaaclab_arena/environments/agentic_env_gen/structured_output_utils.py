# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Utilities for agents with OpenAI-compatible structured outputs."""

from __future__ import annotations

import copy
import json
from typing import Any

from pydantic import BaseModel

# Truncate echoed response payloads in diagnostic to keep
# error messages readable.
_RESPONSE_PREVIEW_CHARS = 500


def _format_failure_message(
    *,
    model: str,
    response_route: str,
    finish_reason: str | None,
    cause: str,
    sample_payload: str | None,
) -> str:
    return (
        f"Model {model!r} does not support structured outputs:\n"
        f"  response_route = {response_route!r}\n"
        f"  finish_reason  = {finish_reason!r}\n"
        f"  cause          = {cause}\n"
        f"  sample_payload = {sample_payload!r}"
    )


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
    if not choices:
        raise RuntimeError(
            f"ping to model {model!r} returned HTTP 200 with no choices "
            "(content filter / guardrail / rate-limit response with empty body)."
        )
    return choices[0].message.content or ""


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


def check_structured_output_support(
    client: Any,
    model: str,
    spec_class: type[BaseModel],
) -> bool:
    """Probe whether ``model`` can produce ``spec_class``-shaped structured outputs.

    Args:
        client: An OpenAI-compatible client.
        model: Model identifier as understood by the client's base_url.
        spec_class: The pydantic model whose strict schema will be
            sent to the endpoint.

    Returns:
        ``True`` when the probe round-trips successfully (wire ok,
        schema honoured, pydantic validation passed).
    """
    schema = build_strict_schema(spec_class)
    system = (
        f"Return a valid {spec_class.__name__} JSON object. Every required field must be "
        "populated — use realistic dummy values where the prompt doesn't specify one."
    )
    # TODO(qianl): wrap with transient-error retry.

    # API call returns exception.
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "Generate a minimal valid example."},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": spec_class.__name__, "strict": True, "schema": schema},
            },
            temperature=0,
            max_tokens=2000,
        )
    except Exception as exc:
        raise RuntimeError(
            _format_failure_message(
                model=model,
                response_route="empty",
                finish_reason=None,
                cause=f"{type(exc).__name__}: {str(exc)[:_RESPONSE_PREVIEW_CHARS]}",
                sample_payload=None,
            )
        ) from exc

    # API call returns no choices.
    choices = getattr(resp, "choices", None) or []
    if not choices:
        raise RuntimeError(
            _format_failure_message(
                model=model,
                response_route="empty",
                finish_reason=None,
                cause="Response contained no choices (model emitted zero candidates).",
                sample_payload=None,
            )
        )

    # API call returns empty message envelope.
    finish_reason = choices[0].finish_reason
    text, route = extract_response_text(choices[0].message)
    sample = text[:_RESPONSE_PREVIEW_CHARS] if text else None
    if not text:
        raise RuntimeError(
            _format_failure_message(
                model=model,
                response_route=route,
                finish_reason=finish_reason,
                cause="Model returned an empty envelope on both content and reasoning_content.",
                sample_payload=None,
            )
        )

    # API call returns invalid JSON.
    try:
        data = json.loads(text, strict=False)
        spec_class.model_validate(data)
    except Exception as exc:
        raise RuntimeError(
            _format_failure_message(
                model=model,
                response_route=route,
                finish_reason=finish_reason,
                cause=f"{type(exc).__name__}: {str(exc)[:_RESPONSE_PREVIEW_CHARS]}",
                sample_payload=sample,
            )
        ) from exc

    return True
