# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible structured-output inference backend for agent inference steps."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from pydantic import BaseModel

MAX_RETRIES_LIMIT = 10

INTERNAL_BASE_URL = "https://inference-api.nvidia.com"
INTERNAL_MODEL = "azure/anthropic/claude-opus-4-8"
PUBLIC_BASE_URL = "https://integrate.api.nvidia.com/v1"
PUBLIC_MODEL = "openai/gpt-oss-120b"
# If you cannot access the model in your region, you can try the "nvidia/nemotron-3-super-120b-a12b" model.

INFERENCE_ENDPOINT_ENV_VAR = "ARENA_INFERENCE_ENDPOINT"
"""Environment variable naming the inference endpoint every agentic command uses."""


@dataclass(frozen=True)
class InferenceEndpoint:
    """One named inference endpoint: where to call, which model, and which API key to read."""

    name: str
    base_url: str
    model: str
    api_key_env_var: str


INTERNAL_ENDPOINT = InferenceEndpoint("internal", INTERNAL_BASE_URL, INTERNAL_MODEL, "NV_API_KEY")
"""NVIDIA-internal inference endpoint, reached with an internal API key."""

PUBLIC_ENDPOINT = InferenceEndpoint("public", PUBLIC_BASE_URL, PUBLIC_MODEL, "NVIDIA_API_KEY")
"""Publicly reachable build.nvidia.com endpoint, reached with an NGC API key."""

INFERENCE_ENDPOINTS = {endpoint.name: endpoint for endpoint in (INTERNAL_ENDPOINT, PUBLIC_ENDPOINT)}
DEFAULT_ENDPOINT_NAME = PUBLIC_ENDPOINT.name


def resolve_inference_endpoint(name: str | None = None) -> InferenceEndpoint:
    """Return the inference endpoint named by ``name``, the environment, or the default.

    Args:
        name: Endpoint name, or ``None`` to read ``ARENA_INFERENCE_ENDPOINT`` and fall back
            to the public endpoint.

    Returns:
        The selected endpoint preset.
    """
    resolved = name or os.getenv(INFERENCE_ENDPOINT_ENV_VAR) or DEFAULT_ENDPOINT_NAME
    assert resolved in INFERENCE_ENDPOINTS, (
        f"Unknown inference endpoint {resolved!r}: set {INFERENCE_ENDPOINT_ENV_VAR} to one of "
        f"{sorted(INFERENCE_ENDPOINTS)}"
    )
    return INFERENCE_ENDPOINTS[resolved]


@dataclass(frozen=True)
class StructuredOutputRequest:
    """One JSON-schema structured-output chat completion."""

    schema_name: str
    schema: dict[str, Any]
    system: str
    user: str
    retry_label: str


class InferenceBackend:
    """Shared LLM JSON-schema runner with retry and tolerant JSON parsing."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_retries: int = 3,
        endpoint: str | None = None,
    ):
        """Configure an OpenAI-compatible structured-output client.

        Args:
            api_key: API token for the inference endpoint. Falls back to the environment
                variable the selected endpoint reads.
            model: Model identifier passed to the chat completion API. Defaults to the
                selected endpoint's model.
            base_url: OpenAI-compatible inference endpoint. Defaults to the selected
                endpoint's base URL.
            temperature: Sampling temperature for completion requests.
            max_tokens: Maximum tokens in each completion response.
            max_retries: Additional attempts after a recoverable failure; must be in
                ``[0, MAX_RETRIES_LIMIT)``.
            endpoint: Inference endpoint name, ``internal`` or ``public``. Falls back to
                the ``ARENA_INFERENCE_ENDPOINT`` environment variable.
        """
        assert (
            0 <= max_retries < MAX_RETRIES_LIMIT
        ), f"max_retries must be in [0, {MAX_RETRIES_LIMIT}), got {max_retries}"
        inference_endpoint = resolve_inference_endpoint(endpoint)
        resolved_api_key = api_key or os.getenv(inference_endpoint.api_key_env_var)
        assert resolved_api_key, (
            f"API key required for the {inference_endpoint.name!r} inference endpoint: set "
            f"{inference_endpoint.api_key_env_var} or pass api_key. Select another endpoint with "
            f"{INFERENCE_ENDPOINT_ENV_VAR}."
        )
        resolved_base_url = base_url or inference_endpoint.base_url
        resolved_model = model or inference_endpoint.model
        print(
            f"[inference] endpoint {inference_endpoint.name!r} model {resolved_model!r} at {resolved_base_url}",
            flush=True,
        )
        client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
        self._client: OpenAI = client
        self._endpoint = inference_endpoint
        self._model = resolved_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        _ping(client, resolved_model)

    @property
    def endpoint(self) -> InferenceEndpoint:
        """Inference endpoint preset the client was configured from."""
        return self._endpoint

    @property
    def model(self) -> str:
        """Model identifier passed to completion requests."""
        return self._model

    @property
    def client(self) -> OpenAI:
        """OpenAI-compatible client used for completion requests."""
        return self._client

    def run_json(self, request: StructuredOutputRequest) -> dict[str, Any]:
        """Call a JSON-schema structured-output endpoint and parse the response as JSON.

        Args:
            request: System/user prompts, JSON schema metadata, and retry log label.

        Returns:
            Parsed JSON object from the model response.
        """
        messages = [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user},
        ]
        last_exc: Exception | None = None
        for attempt in range(1 + self._max_retries):
            if attempt > 0:
                print(f"[{request.retry_label}] retry {attempt}/{self._max_retries} after: {last_exc}", flush=True)
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": request.schema_name,
                            "strict": True,
                            "schema": request.schema,
                        },
                    },
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                choices = getattr(resp, "choices", None) or []
                assert choices, (
                    f"Model {self._model!r} returned HTTP 200 with no choices "
                    "(content filter / guardrail / rate-limit response with empty body)."
                )
                text = _extract_response_text(choices[0].message)
                assert text, (
                    f"Model {self._model!r} returned an empty structured-outputs envelope. "
                    "Verify the endpoint/model supports response_format=json_schema."
                )
                # ``strict=False`` lets json.loads accept unescaped control characters
                # (e.g. literal tabs) inside JSON strings — DeepSeek-v4-flash is known
                # to emit these.
                # Model response is wrapped in a single-key dictionary, e.g. {"input": {<answer>}} to <answer>.
                # Seen on the default azure/anthropic/claude-opus-4-8, but not DeepSeek.
                # TODO(xinjieyao): check if other models also wrap the response in a single-key dictionary.
                return _unwrap_provider_envelope(json.loads(text, strict=False), request.schema)
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(
            f"Model {self._model!r} failed {request.retry_label} after "
            f"{1 + self._max_retries} attempts. Last error: {last_exc}"
        ) from last_exc


def _unwrap_provider_envelope(data: Any, schema: dict[str, Any]) -> Any:
    """Drop a single-key wrapper some models put around their answer, e.g. from {"input": {<answer>}} to <answer>."""
    if not isinstance(data, dict) or len(data) != 1:
        return data
    ((key, value),) = data.items()
    if key in (schema.get("properties") or {}) or not isinstance(value, dict):
        return data
    print(f"[inference] unwrapped provider envelope key {key!r} around structured output", flush=True)
    return value


def build_strict_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Return ``model_cls``'s JSON schema munged for OpenAI strict mode."""
    schema = copy.deepcopy(model_cls.model_json_schema())
    _apply_strict_constraints(schema)
    return schema


def _ping(client: OpenAI, model: str) -> str:
    """Smoke-test the endpoint + API key + model with a minimal request.

    Args:
        client: An OpenAI-compatible client (typically ``openai.OpenAI``).
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


def _apply_strict_constraints(node: dict | list) -> None:
    """Recursively apply OpenAI strict-mode constraints to a JSON-schema node."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        # Strict mode forbids ``default`` keys (every field is required, so
        # defaults can never apply). Drop them defensively at every level.
        node.pop("default", None)
        for v in node.values():
            _apply_strict_constraints(v)
    elif isinstance(node, list):
        for v in node:
            _apply_strict_constraints(v)


def _extract_response_text(message: ChatCompletionMessage) -> str | None:
    """Pull structured-output text from a chat-completion message."""
    if message.content:
        return message.content
    # ``reasoning_content`` is NVIDIA DeepSeek's provider-specific
    # channel; it is not a declared field on ``ChatCompletionMessage``
    return getattr(message, "reasoning_content", None)
