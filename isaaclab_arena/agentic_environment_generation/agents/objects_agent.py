# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Foreground-object extraction stage."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from isaaclab_arena.agentic_environment_generation.agents.base_llm_agent import BaseLLMAgent
from isaaclab_arena.agentic_environment_generation.environment_intent_spec import Item


class ObjectsAgentResult(BaseModel):
    """Foreground catalogue objects extracted from a normalized prompt."""

    reasoning: str
    items: list[Item] = Field(default_factory=list)


class ObjectsAgent(BaseLLMAgent[ObjectsAgentResult]):
    """Extract foreground catalogue objects as :class:`Item` entries."""

    def __init__(self, client: Any, model: str) -> None:
        super().__init__(client, model, ObjectsAgentResult)

    def extract(
        self,
        *,
        items_prompt: str,
        asset_catalog: Any,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        max_retries: int = 3,
    ) -> tuple[ObjectsAgentResult, str]:
        """Call the foreground-object extraction prompt."""
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": f"{asset_catalog.to_catalog_string()}\n\nITEMS PROMPT:\n{items_prompt}"},
        ]
        return self.call(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_label="extract_objects",
        )

    @staticmethod
    def system_prompt() -> str:
        return """\
Extract only foreground catalogue objects from the prompt.
Do not include fixtures that are built into the background USD.
Use Item.query and category_tags for fuzzy asset matching; use instance_name only for duplicates.
"""
