# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Background object-reference inference stage."""

from __future__ import annotations

from typing import Any

from isaaclab_arena.agentic_environment_generation.agents.base_llm_agent import BaseLLMAgent
from isaaclab_arena.agentic_environment_generation.background_object_reference_spec import (
    BackgroundObjectReferenceInferenceSpec,
    validate_background_object_reference_inference,
)
from isaaclab_arena.agentic_environment_generation.background_physics_catalog import (
    PhysicsPrimEntry,
    format_physics_prim_catalog,
)
from isaaclab_arena.agentic_environment_generation.environment_intent_spec import EnvironmentIntentSpec


class ObjectReferenceAgent(BaseLLMAgent[BackgroundObjectReferenceInferenceSpec]):
    """Infer background-scoped object references from a physics prim catalog."""

    def __init__(self, client: Any, model: str) -> None:
        super().__init__(client, model, BackgroundObjectReferenceInferenceSpec)

    def infer_background_references(
        self,
        *,
        intent: EnvironmentIntentSpec,
        physics_entries: list[PhysicsPrimEntry],
        usd_path: str,
        fixtures_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> tuple[BackgroundObjectReferenceInferenceSpec, str]:
        """Call the reference inference prompt and validate the output."""
        user = (
            f"BACKGROUND NODE ID: {intent.background}\n\n"
            f"PHYSICS PRIM CATALOG:\n{format_physics_prim_catalog(physics_entries, usd_path=usd_path)}\n\n"
            f"ENVIRONMENT INTENT SPEC:\n{intent.model_dump_json(indent=2)}\n\n"
            f"FIXTURES PROMPT:\n{fixtures_prompt}"
        )
        result, raw = self.call(
            [
                {"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_label="infer_object_references",
        )
        validate_background_object_reference_inference(intent, result, physics_entries)
        return result, raw

    @staticmethod
    def system_prompt() -> str:
        return """\
Infer object_reference nodes for subprims inside a background USD.
Use only prim paths copied from PHYSICS PRIM CATALOG.
OpenDoorTask.openable_object and CloseDoorTask.openable_object require articulation refs with a revolute joint name.
PickAndPlaceTask.destination_object and destination_location for built-in destination surfaces require rigid refs.
Return remove_item_ids for catalogue items that duplicate built-in fixtures.
"""
