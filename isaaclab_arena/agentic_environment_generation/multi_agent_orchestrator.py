# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Orchestrates staged environment-generation agents."""

from __future__ import annotations

from typing import Any

from isaaclab_arena.agentic_environment_generation.agents.intent_agent import IntentDraftAgent
from isaaclab_arena.agentic_environment_generation.environment_intent_spec import EnvironmentIntentSpec


class MultiAgentOrchestrator:
    """Facade over the staged agents used to generate environment intent."""

    def __init__(self, client: Any, model: str) -> None:
        self.intent_agent = IntentDraftAgent(client, model)

    @property
    def spec_schema(self) -> dict[str, Any]:
        """Strict JSON schema used for the main intent call."""
        return self.intent_agent.schema

    def generate_spec(
        self,
        *,
        prompt: str,
        asset_catalog: Any,
        relation_catalog: Any,
        task_catalog: Any,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> tuple[EnvironmentIntentSpec, str]:
        """Generate an intent spec using the Phase 0 one-pass flow."""
        vocabulary = (
            f"{asset_catalog.to_catalog_string()}\n\n"
            f"{relation_catalog.to_catalog_string()}\n\n"
            f"{task_catalog.to_catalog_string()}"
        )
        return self.intent_agent.generate(
            prompt=prompt,
            vocabulary=vocabulary,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
