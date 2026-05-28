# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Schema the LLM must fill in when parsing a natural-language env-generation prompt.

The LLM sees a list of the *available* asset tags / embodiment names pulled
from the registries at call time, and must return a LLMEnvSpec that only uses
those vocabularies. Concrete asset names are resolved in a second, deterministic
step — the LLM never invents USD paths.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Relation kinds currently surfaced to the LLM. Mirror the subset of
# ``ArenaEnvGraphSpatialConstraintType`` that makes sense for tabletop
# prompts; values must match the enum's values one-to-one because the
# resolver looks the constraint type up via
# ``ArenaEnvGraphSpatialConstraintType(kind)`` rather than maintaining a
# parallel dict. Solver-internal kinds (``position_limits``,
# ``random_around_solution``, ``rotate_around_solution``) are intentionally
# omitted — they describe how the placement solver explores poses and are
# not natural for an LLM to emit.
# "in" has no In class in isaaclab_arena.relations.relations yet — see the
# TODO there. The downstream env builder materializes goal-state "in"
# relations as the task's success predicate.
RelationKind = Literal["on", "in", "next_to", "at_position", "at_pose", "is_anchor"]

ItemRole = Literal["foreground", "distractor", "anchor"]

# Task kinds the LLM can propose as an atomic task.
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
    # Spawn scale. ``None`` (the default) lets the placement proposer
    # auto-fit the asset; an explicit positive float overrides the
    # auto-fit.
    scale: float | None = None


class Relation(BaseModel):
    """A spatial / structural relation between items.

    Binary kinds (``on``, ``in``, ``next_to``, ...) must set ``target`` to the
    other item — semantics is "subject is in relation to target". Unary kinds
    (``is_anchor``, ``at_position``, ...) describe an intrinsic property of
    ``subject`` alone and must leave ``target`` as ``None``.
    """

    kind: RelationKind
    subject: str
    target: str | None = None
    params: dict = Field(default_factory=dict)

    def identity(self) -> tuple[str, str, str | None]:
        """Hashable identity for diffing scene graphs — ignores params."""
        return (self.kind, self.subject, self.target)


class Task(BaseModel):
    """One atomic task in the plan that transforms the env state.

    A task specifies what action to perform (kind), what object it acts on
    (subject), and optionally where it goes (target). The description provides
    natural-language context for the task.
    """

    kind: TaskKind
    subject: str  # object instance name (e.g. 'avocado', 'microwave')
    target: str | None = None  # target object/location (e.g. 'bowl', 'background')
    description: str  # natural-language task description


class LLMEnvSpec(BaseModel):
    """LLM output — a structured plan for the env and a list of tasks.

    The language prompt is decomposed into:

      * ``initial_scene_graph`` — every relation that holds at env reset.
        This configures where objects spawn. This is a FULL snapshot
        including all relations that persist throughout all tasks. (Field
        name kept as ``initial_scene_graph`` even though the class is now
        ``LLMEnvSpec`` — renaming the field would change the JSON schema
        the LLM is prompted against and is out of scope here.)
      * ``tasks`` — a list of atomic tasks to execute in sequence. Each
        task specifies what to do (kind), what object(s) it acts on
        (subject/target), and a natural-language description. The task
        sequence implicitly defines the intermediate env graphs by applying
        each task's transformations in order.
    """

    task_description: str
    background: str
    embodiment: str = "franka_ik"
    items: list[Item]
    initial_scene_graph: list[Relation]
    tasks: list[Task]

    @model_validator(mode="after")
    def _tasks_must_be_non_empty(self) -> LLMEnvSpec:
        if not self.tasks:
            raise ValueError(
                "tasks list is empty — at least one task must be specified to define the env transformation."
            )
        return self
