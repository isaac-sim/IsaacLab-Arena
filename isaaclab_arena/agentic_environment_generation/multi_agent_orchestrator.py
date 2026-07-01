# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Orchestrates staged environment-generation agents."""

from __future__ import annotations

from typing import Any

from isaaclab_arena.agentic_environment_generation.agents.intent_agent import IntentDraftAgent
from isaaclab_arena.agentic_environment_generation.agents.object_reference_agent import ObjectReferenceAgent
from isaaclab_arena.agentic_environment_generation.agents.prompt_normalization_agent import PromptNormalizationAgent
from isaaclab_arena.agentic_environment_generation.asset_matcher import IntentResolutionTraceEvent, match_asset
from isaaclab_arena.agentic_environment_generation.background_object_reference_utils import (
    apply_background_object_reference_inference,
)
from isaaclab_arena.agentic_environment_generation.background_physics_catalog import resolve_background_usd_path
from isaaclab_arena.agentic_environment_generation.environment_intent_spec import EnvironmentIntentSpec
from isaaclab_arena.agentic_environment_generation.usd_prim_index import UsdPrimIndex
from isaaclab_arena.assets.registries import AssetRegistry


class MultiAgentOrchestrator:
    """Facade over the staged agents used to generate environment intent."""

    def __init__(self, client: Any, model: str) -> None:
        self.normalization_agent = PromptNormalizationAgent(client, model)
        self.intent_agent = IntentDraftAgent(client, model)
        self.object_reference_agent = ObjectReferenceAgent(client, model)

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
        """Generate an intent spec using the staged multi-agent flow."""
        normalized, raw_normalized = self.normalization_agent.normalize(
            prompt=prompt,
            task_catalog=task_catalog,
            temperature=temperature,
            max_retries=max_retries,
        )
        vocabulary = (
            f"{asset_catalog.to_catalog_string()}\n\n"
            f"{relation_catalog.to_catalog_string()}\n\n"
            f"{task_catalog.to_catalog_string()}"
        )
        intent, raw_intent = self.intent_agent.generate(
            prompt=normalized.background_prompt
            + "\n\n"
            + normalized.items_prompt
            + "\n\n"
            + "\n".join(task.description for task in normalized.tasks)
            + "\n\n"
            + prompt,
            vocabulary=vocabulary,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        if normalized.fixtures_prompt.strip():
            intent, raw_reference = self._infer_background_references(
                intent,
                fixtures_prompt=normalized.fixtures_prompt,
                asset_registry=AssetRegistry(),
                temperature=temperature,
                max_retries=max_retries,
            )
            raw_response = f"{raw_normalized}\n\n{raw_intent}\n\n{raw_reference}"
        else:
            raw_response = f"{raw_normalized}\n\n{raw_intent}"
        return intent, raw_response

    def _infer_background_references(
        self,
        intent: EnvironmentIntentSpec,
        *,
        fixtures_prompt: str,
        asset_registry: AssetRegistry,
        temperature: float,
        max_retries: int,
    ) -> tuple[EnvironmentIntentSpec, str]:
        """Infer background-scoped references for ``intent`` when the background resolves."""
        trace: list[IntentResolutionTraceEvent] = []
        background_name = match_asset(
            asset_registry, intent.background, "background", trace, required_tags=["background"]
        )
        if background_name is None:
            return intent, ""

        usd_path = resolve_background_usd_path(asset_registry, background_name)
        prim_index = UsdPrimIndex(usd_path=usd_path)
        physics_entries = prim_index.list_entries()
        inference, raw_reference = self.object_reference_agent.infer_references(
            intent=intent,
            scope="background",
            parent_node_id=intent.background,
            parent_asset_name=background_name,
            physics_entries=physics_entries,
            usd_path=prim_index.get_usd_path(),
            reference_prompt=fixtures_prompt,
            temperature=temperature,
            max_retries=max_retries,
        )
        return apply_background_object_reference_inference(intent, inference), raw_reference
