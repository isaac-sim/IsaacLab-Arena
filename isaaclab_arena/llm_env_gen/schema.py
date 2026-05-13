# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Schema the LLM must fill in when parsing a natural-language scene prompt.

The LLM sees a list of the *available* asset tags / embodiment names pulled
from the registries at call time, and must return a SceneSpec that only uses
those vocabularies. Concrete asset names are resolved by the Resolver in a
second, deterministic step — the LLM never invents USD paths.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Relation kinds currently surfaced to the LLM. Mirror the subset of
# isaaclab_arena.relations.relations that makes sense for tabletop prompts.
# "in" has no In class in isaaclab_arena.relations.relations yet — see the
# TODO there. The scene builder materializes goal-state "in" relations as
# the task's success predicate.
RelationKind = Literal["on", "in", "next_to", "at_position", "is_anchor", "open", "closed"]

ItemRole = Literal["foreground", "distractor", "anchor"]

# Task kinds the LLM can propose as atomic actions in a plan.
TaskKind = Literal["pick_and_place", "open_door", "close_door"]


class Item(BaseModel):
    """One object the LLM wants in the scene.

    `query` is the short human name from the prompt ("avocado", "bowl"). The
    resolver maps it to a registered asset. `category_tags` narrow the search
    and act as a fallback when the exact name does not resolve — e.g. a
    distractor "vegetable" resolves to any asset tagged "vegetable".
    """

    query: str
    role: ItemRole
    category_tags: list[str] = Field(default_factory=list)
    instance_name: str | None = None
    # Uniform spawn scale. ``None`` (the default) lets the placement
    # proposer auto-fit the asset against the tabletop bbox; an explicit
    # positive float overrides the auto-fit.
    scale: float | None = None


class Relation(BaseModel):
    """A spatial / structural relation between two items (or on one item)."""

    kind: RelationKind
    subject: str
    target: str | None = None
    params: dict = Field(default_factory=dict)

    def identity(self) -> tuple[str, str, str | None]:
        """Hashable identity for diffing scene graphs — ignores params."""
        return (self.kind, self.subject, self.target)


class Task(BaseModel):
    """One atomic task in the plan that transforms the scene state.

    A task specifies what action to perform (kind), what object it acts on
    (subject), and optionally where it goes (target). The description provides
    natural-language context for the task.
    """

    kind: TaskKind
    subject: str  # object instance name (e.g. 'avocado', 'microwave')
    target: str | None = None  # target object/location (e.g. 'bowl', 'background')
    description: str  # natural-language task description


class SceneSpec(BaseModel):
    """LLM output — a structured plan for the scene and a list of tasks.

    The language prompt is decomposed into:

      * ``initial_scene_graph`` — every relation that holds at env reset.
        This configures where objects spawn. This is a FULL snapshot
        including all relations that persist throughout all tasks.
      * ``tasks`` — a list of atomic actions to execute in sequence. Each
        task specifies what to do (kind), what object(s) it acts on
        (subject/target), and a natural-language description. The task
        sequence implicitly defines the intermediate scene graphs by applying
        each task's transformations in order.
    """

    task_description: str
    background: str
    embodiment: str = "franka_ik"
    items: list[Item]
    initial_scene_graph: list[Relation]
    tasks: list[Task]

    @model_validator(mode="after")
    def _tasks_must_be_non_empty(self) -> SceneSpec:
        if not self.tasks:
            raise ValueError(
                "tasks list is empty — at least one task must be specified to define the scene transformation."
            )
        return self
