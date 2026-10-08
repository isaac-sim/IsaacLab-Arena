# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible structured-output inference backend for agent inference steps."""

from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

from openai import APIConnectionError, APIStatusError, OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Inference endpoints
# -----------------------------------------------------------------------------

INFERENCE_ENDPOINT_ENV_VAR = "ARENA_INFERENCE_ENDPOINT"
"""Environment variable naming the inference endpoint every agentic command uses."""


@dataclass(frozen=True)
class InferenceEndpoint:
    """One named inference endpoint: where to call, which model, and which API key to read."""

    name: str
    base_url: str
    model: str
    api_key_env_var: str
    max_tokens_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    supports_temperature: bool = True


INTERNAL_ENDPOINT = InferenceEndpoint(
    name="internal",
    base_url="https://inference-api.nvidia.com",
    model="openai/openai/gpt-5.6-terra",
    api_key_env_var="NV_API_KEY",
    max_tokens_parameter="max_completion_tokens",
    supports_temperature=False,
)
"""NVIDIA-internal inference endpoint, reached with an internal API key."""

PUBLIC_ENDPOINT = InferenceEndpoint(
    name="public",
    base_url="https://integrate.api.nvidia.com/v1",
    model="deepseek-ai/deepseek-v4.1-flash",
    api_key_env_var="NVIDIA_API_KEY",
)
"""Publicly reachable build.nvidia.com endpoint, reached with an NGC API key."""

OPENAI_ENDPOINT = InferenceEndpoint(
    name="openai",
    base_url="https://api.openai.com/v1",
    model="gpt-5.6-terra",
    api_key_env_var="OPENAI_API_KEY",
    max_tokens_parameter="max_completion_tokens",
    supports_temperature=False,
)
"""Direct OpenAI API endpoint, reached with an OpenAI API key."""

INFERENCE_ENDPOINTS = {endpoint.name: endpoint for endpoint in (INTERNAL_ENDPOINT, PUBLIC_ENDPOINT, OPENAI_ENDPOINT)}
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


# -----------------------------------------------------------------------------
# Inference backend
# -----------------------------------------------------------------------------

MAX_RETRIES_LIMIT = 10


@dataclass(frozen=True)
class InferenceBackendCfg:
    """Configure shared inference without embedding credentials in experiment files."""

    endpoint: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key_env_var: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096
    max_retries: int = 3
    request_timeout_s: float = 60.0
    retry_budget_s: float = 120.0
    retry_backoff_s: float = 1.0
    max_tokens_parameter: Literal["max_tokens", "max_completion_tokens"] | None = None
    supports_temperature: bool | None = None
    supports_images: bool = False
    """Opt in only for a model that accepts image content."""

    response_format: Literal["json_schema", "json_object"] = "json_schema"
    """JSON-object mode includes the schema in the prompt; callers still validate commands."""

    def __post_init__(self):
        assert 0 <= self.max_retries < MAX_RETRIES_LIMIT, "Invalid retry count"
        assert self.max_tokens > 0, "max_tokens must be positive"
        assert self.request_timeout_s > 0 and self.retry_budget_s > 0, "Timeouts must be positive"
        assert self.retry_backoff_s >= 0, "Retry backoff must be nonnegative"
        assert all(
            math.isfinite(value)
            for value in (
                self.request_timeout_s,
                self.retry_budget_s,
                self.retry_backoff_s,
                self.temperature,
            )
        ), "Timeouts, backoff, and temperature must be finite"
        assert self.max_tokens_parameter in (None, "max_tokens", "max_completion_tokens"), "Invalid token parameter"
        assert self.response_format in ("json_schema", "json_object"), "Unsupported response format"


@dataclass(frozen=True)
class InferenceRequest:
    """Send chat messages containing text and optional OpenAI-compatible image parts."""

    messages: list[ChatCompletionMessageParam]
    response_schema: dict[str, Any]
    schema_name: str = "agent_command"


@dataclass(frozen=True)
class InferenceResponse:
    """Return parsed output and provider metadata for evaluation artifacts."""

    data: dict[str, Any]
    model: str
    latency_s: float
    attempts: int
    request_id: str | None
    usage: dict[str, int] | None


class InferenceResponseError(ValueError):
    """Report a refusal, truncated completion, or invalid JSON for caller-owned repair."""


@dataclass(frozen=True)
class StructuredOutputRequest:
    """One JSON-schema structured-output chat completion."""

    schema_name: str
    schema: dict[str, Any]
    system: str
    user: str
    retry_label: str


class InferenceBackend:
    """Share OpenAI-compatible text and vision inference across Arena agents."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_retries: int = 3,
        endpoint: str | None = None,
        *,
        config: InferenceBackendCfg | None = None,
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
            endpoint: Inference endpoint name, ``internal``, ``public``, or ``openai``.
                Falls back to the ``ARENA_INFERENCE_ENDPOINT`` environment variable.
            config: Typed configuration for multimodal inference. Overrides legacy
                options except api_key and skips the legacy constructor health check.
        """
        self.config = config or InferenceBackendCfg(
            endpoint=endpoint,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        if config is not None:
            endpoint, base_url, model = config.endpoint, config.base_url, config.model
            temperature, max_tokens, max_retries = config.temperature, config.max_tokens, config.max_retries
        assert (
            0 <= max_retries < MAX_RETRIES_LIMIT
        ), f"max_retries must be in [0, {MAX_RETRIES_LIMIT}), got {max_retries}"
        inference_endpoint = resolve_inference_endpoint(endpoint)
        key_env_var = self.config.api_key_env_var or inference_endpoint.api_key_env_var
        resolved_api_key = api_key or os.getenv(key_env_var)
        assert resolved_api_key, (
            f"API key required for the {inference_endpoint.name!r} inference endpoint: set "
            f"{key_env_var} or pass api_key. Select another endpoint with "
            f"{INFERENCE_ENDPOINT_ENV_VAR}."
        )
        resolved_base_url = base_url or inference_endpoint.base_url
        resolved_model = model or inference_endpoint.model
        print(
            f"[inference] endpoint {inference_endpoint.name!r} model {resolved_model!r} at {resolved_base_url}",
            flush=True,
        )
        client_options = {}
        if config is not None:
            client_options = {"max_retries": 0, "timeout": config.request_timeout_s}
        client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url, **client_options)
        self._client: OpenAI = client
        self._endpoint = inference_endpoint
        self._model = resolved_model
        self._closed = False
        if config is None:
            try:
                _ping(client, inference_endpoint, resolved_model)
            except Exception:
                self.close()
                raise

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Parse one structured prediction, retrying only transient transport failures.

        Args:
            request: Multimodal conversation and expected output schema.

        Returns:
            Parsed JSON object and metadata. Schema and command validation belong
            to the caller. The retry budget prevents starting further attempts;
            an in-flight call is governed by the SDK's request timeout.
        """
        return self._infer(request)

    def _infer(self, request: InferenceRequest, *, legacy_json: bool = False) -> InferenceResponse:
        """Share transport, retry accounting, and metadata across text and vision callers."""
        assert not self._closed, "Inference backend is closed"
        assert request.messages, "At least one message is required"
        messages = copy.deepcopy(request.messages)
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url":
                        assert self.config.supports_images, "Enable supports_images for a vision-capable model"
        if self.config.response_format == "json_schema":
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": request.schema_name, "schema": request.response_schema},
            }
        else:
            response_format = {"type": "json_object"}
            messages.append(
                {"role": "user", "content": "Return JSON matching this schema: " + json.dumps(request.response_schema)}
            )
        token_parameter = self.config.max_tokens_parameter or self._endpoint.max_tokens_parameter
        options = {token_parameter: self.config.max_tokens}
        supports_temperature = self.config.supports_temperature
        if supports_temperature is None:
            supports_temperature = self._endpoint.supports_temperature
        if supports_temperature:
            options["temperature"] = self.config.temperature

        start = time.monotonic()
        deadline = start + self.config.retry_budget_s
        for attempt in range(1 + self.config.max_retries):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Inference retry budget exhausted")
            try:
                # Disable SDK retries even for clients constructed through the legacy API.
                client = self._client.with_options(max_retries=0, timeout=min(remaining, self.config.request_timeout_s))
                response = client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    response_format=response_format,
                    **options,
                )
            except (APIConnectionError, APIStatusError, ConnectionError) as exc:
                transient = (
                    isinstance(exc, (APIConnectionError, ConnectionError))
                    or exc.status_code in (408, 409, 429)
                    or exc.status_code >= 500
                )
                if not transient or attempt == self.config.max_retries:
                    raise
                delay = self.config.retry_backoff_s * (2**attempt)
                if time.monotonic() + delay >= deadline:
                    raise TimeoutError("Inference retry budget exhausted") from exc
                time.sleep(delay)
                continue
            data = _parse_inference_response(response, legacy_json=legacy_json)
            if legacy_json:
                data = _unwrap_provider_envelope(data, request.response_schema)
            usage = None
            if response.usage is not None:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            return InferenceResponse(
                data=data,
                model=response.model,
                latency_s=time.monotonic() - start,
                attempts=attempt + 1,
                request_id=getattr(response, "_request_id", None),
                usage=usage,
            )
        raise RuntimeError("unreachable")

    def close(self) -> None:
        """Release client connections; repeated calls are harmless."""
        if not self._closed:
            self._client.close()
            self._closed = True

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
        """Use shared inference with legacy text parsing and provider-envelope handling.

        Args:
            request: System/user prompts and JSON schema metadata. retry_label is
                retained for source compatibility; transport errors retain their types.

        Returns:
            Parsed JSON object from the model response.
        """
        return self._infer(
            InferenceRequest(
                messages=[
                    {"role": "system", "content": request.system},
                    {"role": "user", "content": request.user},
                ],
                response_schema=request.schema,
                schema_name=request.schema_name,
            ),
            legacy_json=True,
        ).data


def _parse_inference_response(response, *, legacy_json: bool = False) -> dict[str, Any]:
    """Require a complete JSON object without interpreting reasoning as a command."""
    if not response.choices:
        raise InferenceResponseError("No completion choices returned")
    choice = response.choices[0]
    if choice.message.refusal or choice.finish_reason != "stop":
        raise InferenceResponseError(f"Completion refused or unfinished: {choice.finish_reason}")
    content = _extract_response_text(choice.message) if legacy_json else choice.message.content
    if not content:
        raise InferenceResponseError("Empty structured output")
    try:
        data = json.loads(content, strict=not legacy_json, parse_constant=_reject_json_constant)
    except ValueError as exc:
        raise InferenceResponseError("Invalid JSON output") from exc
    if not isinstance(data, dict):
        raise InferenceResponseError("Expected a JSON object")
    return data


def _reject_json_constant(value: str) -> None:
    """Reject nonstandard JSON numbers such as NaN and Infinity."""
    raise ValueError(f"Invalid JSON number: {value}")


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


def _completion_options(endpoint: InferenceEndpoint, max_tokens: int, temperature: float) -> dict[str, int | float]:
    """Return completion arguments supported by the selected endpoint."""
    options: dict[str, int | float] = {endpoint.max_tokens_parameter: max_tokens}
    if endpoint.supports_temperature:
        options["temperature"] = temperature
    return options


def _ping(client: OpenAI, endpoint: InferenceEndpoint, model: str) -> str:
    """Smoke-test the endpoint + API key + model with a minimal request.

    Args:
        client: An OpenAI-compatible client (typically ``openai.OpenAI``).
        endpoint: Endpoint capabilities used to construct the request.
        model: Model identifier forwarded to
            ``client.chat.completions.create(model=...)``.

    Returns:
        The model's response text.
    """
    # TODO(qianl): wrap with transient-error retry.
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Respond with exactly: OK"}],
        **_completion_options(endpoint, 32, 0),
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
