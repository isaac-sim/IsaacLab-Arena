# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared structured-output LLM call helper for staged generation agents."""

from __future__ import annotations

import json
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from isaaclab_arena.agentic_environment_generation.agent_utils import build_strict_schema, extract_response_text

T = TypeVar("T", bound=BaseModel)


class BaseLLMAgent(Generic[T]):
    """Call an OpenAI-compatible chat endpoint and parse one strict Pydantic output."""

    def __init__(self, client: Any, model: str, output_model: type[T], schema_name: str | None = None) -> None:
        """Configure the shared client/model and cache the strict JSON schema."""
        self.client = client
        self.model = model
        self.output_model = output_model
        self.schema_name = schema_name or output_model.__name__
        self.schema = build_strict_schema(output_model)

    def call(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_retries: int = 3,
        retry_label: str | None = None,
    ) -> tuple[T, str]:
        """Call the model with ``messages`` and return ``(validated_model, raw_text)``."""
        last_exc: Exception | None = None
        label = retry_label or self.__class__.__name__
        for attempt in range(1 + max_retries):
            if attempt > 0:
                print(f"[{label}] retry {attempt}/{max_retries} after: {last_exc}", flush=True)
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": self.schema_name,
                            "strict": True,
                            "schema": self.schema,
                        },
                    },
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                choices = getattr(resp, "choices", None) or []
                assert choices, (
                    f"Model {self.model!r} returned HTTP 200 with no choices "
                    "(content filter / guardrail / rate-limit response with empty body)."
                )
                text, route = extract_response_text(choices[0].message)
                assert route != "empty", (
                    f"Model {self.model!r} returned an empty structured-outputs envelope. "
                    "Verify the endpoint/model supports response_format=json_schema."
                )
                data = json.loads(text, strict=False)
                return self.output_model.model_validate(data), text
            except Exception as exc:
                last_exc = exc

        raise RuntimeError(
            f"Model {self.model!r} failed after {1 + max_retries} attempts. Last error: {last_exc}"
        ) from last_exc
