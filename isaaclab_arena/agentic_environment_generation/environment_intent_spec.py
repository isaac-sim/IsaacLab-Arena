# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Schema the agent must fill in when parsing a natural-language env-generation prompt."""

from __future__ import annotations

import inspect

from pydantic import BaseModel, Field, model_validator

from isaaclab_arena.assets.registries import TaskRegistry
from isaaclab_arena.environments.arena_env_graph_types import SpatialRelationSpec, TaskSpec


# Item differs from ArenaEnvGraphNodeSpec. Item does not name a specify asset
# in the AssetRegistry. Instead, it is a query string that the agent proposes
# to match against the asset catalog.
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
            "Resolve any ambiguity here before filling the structured fields below."
        ),
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
    initial_state_graph: list[SpatialRelationSpec] = Field(
        description=(
            "FULL snapshot of all spatial relations in the starting state. "
            "Use only relation names from the RELATIONS block. Every "
            "persistent placement (e.g. bowl on table, distractors on table) "
            "must appear here in its starting form."
        ),
    )
    # TODO(v0.4+): Add support for composite tasks (parallel/unordered execution)
    # Currently v0.3 only supports sequential task chains.
    tasks: list[TaskSpec] = Field(
        description=(
            "Tasks to execute in sequence, using only kinds from the TASKS block. "
            "Only include tasks the user prompt explicitly requests. "
            "Return [] when the prompt describes a static scene with no "
            "robot action. Do not invent placeholder tasks."
        ),
    )

    # Intent-only checks: nested SpatialRelationSpec / TaskSpec already validate registry membership.
    @model_validator(mode="after")
    def _validate_agent_intent_tasks(self) -> EnvironmentIntentSpec:
        task_registry = TaskRegistry()
        for task in self.tasks:
            # Check required task is agent-ready
            task_cls = task_registry.get_task_by_name(task.kind)
            assert getattr(task_cls, "agent_ready", False), f"Task {task.kind!r} is not agent-ready"
            assert task.description and task.description.strip(), f"Task {task.kind!r} requires a non-empty description"
            # Check required task class constructor parameters are present
            for required_param in required_task_init_param_names(task_cls):
                assert required_param in task.params, f"Task {task.kind!r} is missing required param {required_param}"
                value = task.params[required_param]
                assert (
                    isinstance(value, str) and value.strip()
                ), f"Task {task.kind!r} required param {required_param!r} must be a non-empty string"
        return self
