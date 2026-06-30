# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Background object-reference inference stage."""

from __future__ import annotations

from typing import Any, Literal

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
    """Infer object references from a parent asset's physics prim catalog."""

    def __init__(self, client: Any, model: str) -> None:
        super().__init__(client, model, BackgroundObjectReferenceInferenceSpec)

    def infer_references(
        self,
        *,
        intent: EnvironmentIntentSpec,
        scope: Literal["background", "item"],
        parent_node_id: str,
        parent_asset_name: str,
        physics_entries: list[PhysicsPrimEntry],
        usd_path: str,
        reference_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> tuple[BackgroundObjectReferenceInferenceSpec, str]:
        """Call the reference inference prompt for subprims inside one parent asset."""
        user = (
            f"PARENT SCOPE: {scope}\n"
            f"PARENT NODE ID: {parent_node_id}\n"
            f"PARENT ASSET: {parent_asset_name}\n\n"
            f"PHYSICS PRIM CATALOG:\n{format_physics_prim_catalog(physics_entries, usd_path=usd_path)}\n\n"
            f"ENVIRONMENT INTENT SPEC:\n{intent.model_dump_json(indent=2)}\n\n"
            f"REFERENCE PROMPT:\n{reference_prompt}"
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
        for ref in result.object_references:
            assert ref.scope == scope, f"reference {ref.id!r} must set scope={scope!r}"
            if scope == "item":
                assert ref.parent_id == parent_node_id, f"item reference {ref.id!r} must parent to {parent_node_id!r}"
        validate_background_object_reference_inference(intent, result, physics_entries)
        return result, raw

    @staticmethod
    def system_prompt() -> str:
        return """\
Infer object_reference nodes for subprims inside a parent asset USD.
Use only prim paths copied from PHYSICS PRIM CATALOG.
Every object_reference must set scope to PARENT SCOPE.
For item-scoped refs, every object_reference must set parent_id to PARENT NODE ID.
OpenDoorTask.openable_object and CloseDoorTask.openable_object require articulation refs with a revolute joint name.
PickAndPlaceTask.destination_object and destination_location for physical destination surfaces require rigid refs.
For background fixtures, return remove_item_ids for catalogue items that duplicate built-in fixtures.
OpenDoorTask.openable_object and CloseDoorTask.openable_object should remain the parent item unless a child articulation is explicitly requested.
"""
