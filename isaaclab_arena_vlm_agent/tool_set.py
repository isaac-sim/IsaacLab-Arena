# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tools a VLM policy offers the model before it returns an action."""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from isaaclab_arena.inference.backend import build_strict_schema
from isaaclab_arena_vlm_agent.observation_adapter import VLMObservationAdapter


@dataclass
class ToolResult:
    """The answer to one tool call."""

    text: str
    """JSON text returned to the model as the tool message."""

    image_urls: list[str] = field(default_factory=list)
    """Images the answer shows, as data URLs, sent in a user message after the tool messages."""


def make_function_tool(name: str, description: str, arguments_model: type[BaseModel]) -> dict[str, Any]:
    """Return an OpenAI function tool definition whose arguments follow a strict pydantic model."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": build_strict_schema(arguments_model),
            "strict": True,
        },
    }


class VLMToolSet(ABC):
    """Tools the model may call, through the endpoint's tool calling, before it returns an action."""

    max_rounds: int = 5
    """Tool-call rounds allowed per decision; after the last one the model must return an action."""

    @abstractmethod
    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return the OpenAI function tool definitions offered to the model."""

    def get_system_prompt_addition(self) -> str:
        """Return text explaining when and how to use the tools, appended to the system prompt."""
        return ""

    def prepare_request(
        self,
        observation_adapter: VLMObservationAdapter,
        env,
        observation,
        env_id: int,
        images: dict[str, np.ndarray],
        prompt_values: dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Return the images and prompt values to send, and keep what later tool calls in this decision need.

        Called once per decision, before the first inference request. The prompt values may be modified in place.

        Args:
            observation_adapter: Adapter that produced the images, for reading their camera views.
            env: Arena environment, optionally gym-wrapped.
            observation: Current batched observations.
            env_id: Environment index the decision is for.
            images: HWC RGB images indexed by camera name.
            prompt_values: JSON-serializable values for the request.

        Returns:
            Images and prompt values to send.
        """
        return images, prompt_values

    @abstractmethod
    def call_tool(self, name: str, arguments: str, env_id: int) -> ToolResult:
        """Answer one tool call from the environment's current decision.

        Args:
            name: Name of the called tool.
            arguments: Raw JSON text of the call arguments.
            env_id: Environment whose decision the call belongs to.

        Returns:
            The answer, with an error message the model can correct when the call is invalid.
        """

    def reset(self, env_ids: list[int] | None = None) -> None:
        """Discard per-environment state for selected environments, or all environments."""
