# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible structured-output query backend for agent inference steps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from isaaclab_arena.agentic_environment_generation.agent_utils import extract_response_text


@dataclass(frozen=True)
class StructuredOutputRequest:
    """One JSON-schema structured-output chat completion."""

    schema_name: str
    schema: dict[str, Any]
    system: str
    user: str
    retry_label: str


@dataclass(frozen=True)
class StructuredOutputResult:
    """Parsed structured-output response from the model."""

    data: dict[str, Any]
    raw_text: str


class QueryBackend:
    """Shared LLM JSON-schema runner with retry and tolerant JSON parsing."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ):
        self._client = client
        self._model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def run_json(self, request: StructuredOutputRequest) -> StructuredOutputResult:
        """Call a JSON-schema structured-output endpoint and parse the response as JSON."""
        messages = [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user},
        ]
        last_exc: Exception | None = None
        for attempt in range(1 + self.max_retries):
            if attempt > 0:
                print(f"[{request.retry_label}] retry {attempt}/{self.max_retries} after: {last_exc}", flush=True)
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
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                choices = getattr(resp, "choices", None) or []
                assert choices, (
                    f"Model {self._model!r} returned HTTP 200 with no choices "
                    "(content filter / guardrail / rate-limit response with empty body)."
                )
                text, route = extract_response_text(choices[0].message)
                assert route != "empty", (
                    f"Model {self._model!r} returned an empty structured-outputs envelope. "
                    "Verify the endpoint/model supports response_format=json_schema."
                )
                # ``strict=False`` lets json.loads accept unescaped control characters
                # (e.g. literal tabs) inside JSON strings — DeepSeek-v4-flash is known
                # to emit these.
                return StructuredOutputResult(data=json.loads(text, strict=False), raw_text=text)
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(
            f"Model {self._model!r} failed {request.retry_label} after "
            f"{1 + self.max_retries} attempts. Last error: {last_exc}"
        ) from last_exc
