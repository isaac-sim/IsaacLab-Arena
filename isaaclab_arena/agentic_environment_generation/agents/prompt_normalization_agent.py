# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Prompt-normalization stage for multi-agent environment generation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from isaaclab_arena.agentic_environment_generation.agents.base_llm_agent import BaseLLMAgent


class TaskParamBinding(BaseModel):
    """Semantic task-param target extracted from a natural-language prompt."""

    param_name: str = Field(min_length=1)
    semantic_target: str = Field(min_length=1)
    target_kind: Literal["background", "item", "fixture", "item_subprim"]


class TaskSketch(BaseModel):
    """Ordered task mention before concrete graph node ids are assigned."""

    kind: str = Field(min_length=1)
    bindings: list[TaskParamBinding] = Field(default_factory=list)
    description: str = Field(min_length=1)


class PromptAnalysisResult(BaseModel):
    """Normalized prompt split into task, object, and fixture concerns."""

    robot_prompt: str | None = None
    background_prompt: str
    items_prompt: str
    fixtures_prompt: str
    tasks: list[TaskSketch] = Field(default_factory=list)


class PromptNormalizationAgent(BaseLLMAgent[PromptAnalysisResult]):
    """Normalize a raw environment-generation prompt into task sketches."""

    def __init__(self, client: Any, model: str) -> None:
        super().__init__(client, model, PromptAnalysisResult)

    def normalize(
        self,
        *,
        prompt: str,
        task_catalog: Any,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        max_retries: int = 3,
    ) -> tuple[PromptAnalysisResult, str]:
        """Call the normalization prompt."""
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": f"{task_catalog.to_catalog_string()}\n\nUSER PROMPT:\n{prompt}"},
        ]
        return self.call(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_label="normalize_prompt",
        )

    @staticmethod
    def system_prompt() -> str:
        return """\
Normalize a robot environment-generation prompt.
Preserve exact names when the user provides them. Otherwise use short semantic phrases.
Split catalogue foreground objects from fixtures that are implied to be inside the background USD.
Return ordered task sketches only for tasks explicitly requested by the prompt.
"""
