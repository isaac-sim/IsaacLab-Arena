# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic composition helpers for staged environment generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from isaaclab_arena.agentic_environment_generation.background_object_reference_spec import (
    BackgroundObjectReferenceInferenceSpec,
)
from isaaclab_arena.agentic_environment_generation.background_object_reference_utils import (
    apply_background_object_reference_inference,
)
from isaaclab_arena.agentic_environment_generation.environment_intent_spec import (
    EnvironmentIntentSpec,
    Item,
    ObjectReferenceItem,
)


@dataclass
class SceneNodeRegistry:
    """Track generated node ids before composition freezes task bindings."""

    node_ids: set[str] = field(default_factory=set)
    frozen: bool = False

    def add(self, node_id: str) -> None:
        """Register ``node_id`` before freeze."""
        assert not self.frozen, "SceneNodeRegistry is frozen"
        assert node_id not in self.node_ids, f"duplicate node id {node_id!r}"
        self.node_ids.add(node_id)

    def freeze(self) -> frozenset[str]:
        """Freeze and return the node id set."""
        self.frozen = True
        return frozenset(self.node_ids)


class ComposedIntentBundle(BaseModel):
    """Intent plus sidecar references consumed by the compiler."""

    intent: EnvironmentIntentSpec
    background_object_references: list[ObjectReferenceItem] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)


class IntentComposer:
    """Combine the intent draft and reference inference into a compile-ready bundle."""

    def compose(
        self,
        intent: EnvironmentIntentSpec,
        *,
        reference_inference: BackgroundObjectReferenceInferenceSpec | None = None,
    ) -> ComposedIntentBundle:
        """Return an intent bundle, applying reference inference when present."""
        if reference_inference is None:
            return ComposedIntentBundle(intent=intent, background_object_references=list(intent.object_references))

        merged_intent = apply_background_object_reference_inference(intent, reference_inference)
        diagnostics = self._validate_known_task_refs(merged_intent, reference_inference.object_references)
        return ComposedIntentBundle(
            intent=merged_intent,
            background_object_references=list(merged_intent.object_references),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _validate_known_task_refs(
        intent: EnvironmentIntentSpec,
        references: list[ObjectReferenceItem],
    ) -> list[str]:
        known_ids = {intent.background, intent.embodiment}
        known_ids.update(item.instance_name or item.query for item in intent.items)
        known_ids.update(ref.id for ref in references)
        diagnostics: list[str] = []
        for task_index, task in enumerate(intent.tasks):
            for param_name, param_value in task.params.items():
                if isinstance(param_value, str) and param_value not in known_ids:
                    diagnostics.append(f"task[{task_index}].{param_name} references unknown node {param_value!r}")
        return diagnostics


def item_node_id(item: Item) -> str:
    """Return the graph node id for an intent item."""
    return item.instance_name or item.query
