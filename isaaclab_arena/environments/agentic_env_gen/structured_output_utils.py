# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Utilities for OpenAI-compatible structured outputs (``response_format=json_schema``).

The functions here are the building blocks the env-gen agent uses to
send strict-mode-compatible schemas, handle provider-specific response
routing (NVIDIA DeepSeek's ``reasoning_content`` quirk), and probe a
candidate model's structured-output capability before deployment.

They are intentionally pydantic-model-agnostic: pass any
``pydantic.BaseModel`` subclass as ``spec_class`` and the utility
adapts. The agent module wires :class:`EnvIntentSpec` in as the
production default.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

# Truncate echoed response payloads in diagnostic results to this many
# characters — long enough to diagnose a failure, short enough to keep
# error messages and probe results readable.
_RESPONSE_PREVIEW_CHARS = 500


@dataclass(frozen=True)
class StructuredOutputSupport:
    """Result of probing a model for structured-outputs capability.

    The probe sends a one-shot request asking the configured model to
    return a payload matching ``spec_class``'s strict schema. The
    result captures every signal a deployment validator needs to
    decide "is this model usable?":

      * ``supported``: True iff a valid ``spec_class`` instance came
        back end-to-end (wire ok, schema honoured, pydantic
        validation passed).
      * ``response_route``: which channel held the structured output
        (``"content"`` for OpenAI-compatible models,
        ``"reasoning_content"`` for NVIDIA DeepSeek, ``"empty"`` when
        the model dropped the request).
      * ``api_error`` / ``parse_error``: filled in (mutually
        exclusively, in that order) when ``supported`` is False so
        the caller can attribute the failure correctly.
    """

    supported: bool
    model: str
    finish_reason: str | None
    response_route: str
    api_error: str | None
    parse_error: str | None
    sample_payload: str | None


def build_strict_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Return ``model_cls``'s JSON schema munged for OpenAI strict mode.

    OpenAI's structured outputs strict mode (and AWS Bedrock's
    Anthropic models, which surface the same constraint) require:

      * ``additionalProperties: false`` on every object schema.
      * Every property listed in ``required`` (use a nullable type
        union — e.g. ``str | None`` — for fields that should be
        emittable as ``null``).
      * No ``default`` keys in the schema (defaults are nonsensical
        when every field is required).

    Pydantic's default ``model_json_schema()`` honours the first
    constraint only. We deep-walk the schema and apply the other two
    so the schema flies past both NVIDIA and Bedrock validation.

    The returned dict is a deep copy — mutating it never leaks back
    into pydantic's internal schema cache.
    """
    schema = copy.deepcopy(model_cls.model_json_schema())
    apply_strict_constraints(schema)
    return schema


def apply_strict_constraints(node: Any) -> None:
    """Recursively apply OpenAI strict-mode constraints to a JSON-schema node.

    Mutates ``node`` in place. Safe to call on an already-munged schema
    (the operation is idempotent).
    """
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

    Sends a one-shot chat completion (no structured outputs) to verify:

      * the API key authenticates,
      * the configured model exists at the client's ``base_url``,
      * the network path is reachable.

    Intended for CI startup probes and constructor-time fail-fast
    checks; the success signal is "we got a response without
    raising". The response *content* is returned for diagnostics but
    intentionally not asserted on — different models phrase the
    acknowledgment differently, and a quirky reply still means the
    wire is working.

    This is the *cheap* probe; pair with
    :func:`check_structured_output_support` for a full deployment
    validation (ping confirms the wire, the probe confirms the
    model can actually produce structured outputs).

    Args:
        client: An OpenAI-compatible client (typically
            ``openai.OpenAI`` or a compatible mock).
        model: Model identifier forwarded to
            ``client.chat.completions.create(model=...)``.

    Returns:
        The model's response text (typically "OK" or similar). Empty
        string if the model returned no content (still a successful
        round-trip).

    Raises:
        Any exception raised by the underlying ``openai`` client.
        Common ones at this layer are ``AuthenticationError``
        (bad key), ``NotFoundError`` (wrong ``model``),
        ``APIConnectionError`` (unreachable endpoint), and
        ``RateLimitError`` (quota exhausted).
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Respond with exactly: OK"}],
        temperature=0,
        max_tokens=8,
    )
    return resp.choices[0].message.content or ""


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
) -> StructuredOutputSupport:
    """Probe whether ``model`` can produce ``spec_class``-shaped structured outputs.

    Sends a single chat-completion against ``client`` with
    ``response_format=json_schema`` carrying ``spec_class``'s strict
    schema and a minimal user prompt asking the model to fabricate a
    valid instance. Reports diagnostics rather than raising so
    deployment validators can decide how to react (warn, fall back,
    abort).

    Two failure modes are reported separately:

      * ``api_error`` — the request was rejected at the wire
        (400/401/etc). The endpoint or its proxy doesn't understand
        ``response_format``, or the schema violates a
        provider-specific constraint (e.g. Bedrock requiring
        ``additionalProperties: false`` everywhere — we munge for
        this, but other constraints can still surface here).
      * ``parse_error`` — the request succeeded and the model
        returned a payload, but it doesn't parse as JSON or doesn't
        validate against the schema.

    Args:
        client: An OpenAI-compatible client (typically
            ``openai.OpenAI`` or a compatible mock).
        model: Model identifier as understood by the client's
            base_url. Forwarded verbatim to
            ``client.chat.completions.create(model=...)``.
        spec_class: The pydantic model whose strict schema will be
            sent to the endpoint.

    Returns:
        A :class:`StructuredOutputSupport` capturing the outcome.
    """
    schema = build_strict_schema(spec_class)
    # The user prompt is deliberately content-free; the schema itself
    # plus the system prompt below carry all the structural
    # information. We just want a valid envelope back.
    system = (
        f"Return a valid {spec_class.__name__} JSON object. Every required field must be "
        "populated — use realistic dummy values where the prompt doesn't specify one."
    )
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
        return StructuredOutputSupport(
            supported=False,
            model=model,
            finish_reason=None,
            response_route="empty",
            api_error=f"{type(exc).__name__}: {str(exc)[:_RESPONSE_PREVIEW_CHARS]}",
            parse_error=None,
            sample_payload=None,
        )

    finish_reason = resp.choices[0].finish_reason
    text, route = extract_response_text(resp.choices[0].message)
    sample = text[:_RESPONSE_PREVIEW_CHARS] if text else None
    if not text:
        return StructuredOutputSupport(
            supported=False,
            model=model,
            finish_reason=finish_reason,
            response_route=route,
            api_error=None,
            parse_error="Model returned an empty envelope on both content and reasoning_content.",
            sample_payload=None,
        )
    try:
        data = json.loads(text, strict=False)
        spec_class.model_validate(data)
    except Exception as exc:
        return StructuredOutputSupport(
            supported=False,
            model=model,
            finish_reason=finish_reason,
            response_route=route,
            api_error=None,
            parse_error=f"{type(exc).__name__}: {str(exc)[:_RESPONSE_PREVIEW_CHARS]}",
            sample_payload=sample,
        )
    return StructuredOutputSupport(
        supported=True,
        model=model,
        finish_reason=finish_reason,
        response_route=route,
        api_error=None,
        parse_error=None,
        sample_payload=sample,
    )
