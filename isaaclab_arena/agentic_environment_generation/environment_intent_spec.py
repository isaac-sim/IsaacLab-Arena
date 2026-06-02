# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Schema the agent must fill in when parsing a natural-language env-generation prompt."""

from __future__ import annotations

import inspect
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from isaaclab_arena.assets.registries import ObjectRelationLibraryRegistry, TaskRegistry

ItemRole = Literal["foreground", "distractor", "anchor"]


class Item(BaseModel):
    """One object the agent wants in the scene."""

    query: str = Field(
        description=(
            "Short human name for the object as it appears in the prompt "
            "(e.g. 'avocado', 'bowl'). The downstream resolver fuzzy-matches "
            "this against the asset catalog — do NOT emit the exact "
            "registered name."
        ),
    )
    role: ItemRole = Field(
        description=(
            "Role the item plays in the env: 'foreground' for objects the "
            "task acts on; 'distractor' for extras mentioned as clutter; "
            "'anchor' for reference surfaces (rare — the background usually "
            "covers this)."
        ),
    )
    category_tags: list[str] = Field(
        default_factory=list,
        description=(
            "Tags that semantically narrow the query, preferring assets with "
            "those tags. PREFERENCE only, not a hard filter — the resolver "
            "falls back to the full catalog if the tag pool is empty or "
            "yields no close match. Err toward emitting useful tags."
        ),
    )
    instance_name: str | None = Field(
        default=None,
        description="Optional explicit instance label for the item; leave null if the prompt does not name one.",
    )


class Relation(BaseModel):
    """A spatial relation between items."""

    kind: str = Field(
        description=(
            "Relation name from the RELATIONS block in the user message "
            "(e.g. 'on', 'next_to', 'is_anchor'). Must match a registered "
            "relation exactly."
        ),
    )
    subject: str = Field(
        description=(
            "Primary endpoint: the object (Item.query) or background this "
            "relation applies to. For binary relations (see RELATIONS arity), "
            "this is the child — e.g. for 'on', the object sitting on the surface."
        ),
    )
    target: str | None = Field(
        default=None,
        description=(
            "Second endpoint for binary relations only: the parent surface or "
            "anchor (Item.query or background name). For 'on', target is what "
            "subject rests on. Leave null for unary relations listed as unary "
            "in RELATIONS (e.g. is_anchor, at_position)."
        ),
    )
    # TODO(qianl): free-form ``dict`` emits ``additionalProperties: true``,
    # which strict-mode structured-outputs endpoints (OpenAI strict /
    # Bedrock-Claude) reject with a 400. The default NVIDIA DeepSeek is
    # lenient and accepts it, so this is a latent portability landmine.
    params: dict = Field(default_factory=dict, description="Optional kind-specific parameters; leave empty by default.")


class Task(BaseModel):
    """One atomic task in the plan that transforms the env state."""

    kind: str = Field(
        description=(
            "Registered task class name from the TASKS block in the user message "
            "(e.g. 'PickAndPlaceTask', 'OpenDoorTask'). Must match "
            "``TaskRegistry.get_task_by_name`` exactly."
        ),
    )
    params: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Constructor kwargs required by this task (listed in TASKS). "
            "Values are Item.query strings for objects or the background "
            "name for scene parameters (e.g. background_scene)."
        ),
    )
    description: str = Field(
        description="Natural-language summary of the task (e.g. 'pick up the avocado and place it in the bowl').",
    )


def required_task_init_param_names(task_cls: type) -> list[str]:
    """Get the list of required parameters from a task class constructor."""
    sig = inspect.signature(task_cls.__init__)
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return required


class EnvironmentIntentSpec(BaseModel):
    """Agent output — a structured "env intent" (blueprint) for the env and a list of tasks."""

    # Forced chain-of-thought field, listed FIRST so the agent emits its
    # analysis before committing to any structured field.
    reasoning: str = Field(
        description=(
            "Step-by-step analysis of the user prompt, written BEFORE the "
            "structured fields below. Identify (1) the task / intent, (2) "
            "the foreground objects the task acts on, (3) the background "
            "surface or scene, (4) any distractors. For each object, "
            "briefly justify the catalog query and tags you will pick. "
            "Resolve any ambiguity here before filling the structured "
            "fields — do not restate this analysis in ``task_description``."
        ),
    )
    task_description: str = Field(
        description="One-sentence natural-language summary of what the env exercises overall.",
    )
    background: str = Field(
        description="Background asset name from the BACKGROUNDS catalog (e.g. 'maple_table_kitchen').",
    )
    embodiment: str = Field(
        default="franka_ik",
        description=(
            "Robot embodiment to control. Use a bare family name ('franka', "
            "'droid', 'g1', 'gr1') when the prompt does not specify a "
            "control mode — the resolver defaults each to its IK variant. "
            "Use a full registered name (e.g. 'franka_joint_pos') only when "
            "the prompt explicitly requests joint control."
        ),
    )
    items: list[Item] = Field(description="Objects to place in the env.")
    initial_state_graph: list[Relation] = Field(
        description=(
            "FULL snapshot of all spatial relations in the starting state. "
            "Use only relation names from the RELATIONS block. Every "
            "persistent placement (e.g. bowl on table, distractors on table) "
            "must appear here in its starting form."
        ),
    )
    # TODO(v0.4+): Add support for composite tasks (parallel/unordered execution)
    # Currently v0.3 only supports sequential task chains.
    tasks: list[Task] = Field(
        description=(
            "Tasks to execute in sequence, using only kinds from the TASKS block. "
            "The task sequence implicitly defines intermediate state graphs. "
            "An empty list is valid for a static scene — prefer empty over "
            "inventing a placeholder task."
        ),
    )

    @model_validator(mode="after")
    def _validate_catalogue_kinds_are_registered(self) -> EnvironmentIntentSpec:
        # Validate relation kinds are in registry.
        allowed_relations = frozenset(ObjectRelationLibraryRegistry().get_all_keys())
        for relation in self.initial_state_graph:
            if relation.kind not in allowed_relations:
                raise ValueError(
                    f"Relation kind {relation.kind!r} is not registered. Allowed: {sorted(allowed_relations)}"
                )

        # Validate task kinds are in registry and agent-ready.
        task_registry = TaskRegistry()
        allowed_tasks = frozenset(
            name
            for name in task_registry.get_all_keys()
            if getattr(task_registry.get_task_by_name(name), "agent_ready", False)
        )
        for task in self.tasks:
            if task.kind not in allowed_tasks:
                raise ValueError(f"Task {task.kind!r} is not agent-ready. Allowed: {sorted(allowed_tasks)}")
            task_cls = task_registry.get_task_by_name(task.kind)
            # Validate task has all required parameters.
            required_params = required_task_init_param_names(task_cls)
            missing = [name for name in required_params if name not in task.params]
            if missing:
                raise ValueError(
                    f"Task {task.kind!r} is missing required params {missing}. Required: {required_params}"
                )
            empty = [
                name
                for name in required_params
                if not isinstance(task.params.get(name), str) or not task.params[name].strip()
            ]
            if empty:
                raise ValueError(f"Task {task.kind!r} has empty required params: {empty}")
        return self
