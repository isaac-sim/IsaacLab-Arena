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
RelationKind = Literal["on", "in", "next_to", "at_position", "is_anchor"]

ItemRole = Literal["foreground", "distractor", "anchor"]


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


class Relation(BaseModel):
    """A spatial / structural relation between two items (or on one item)."""

    kind: RelationKind
    subject: str
    target: str | None = None
    params: dict = Field(default_factory=dict)

    def identity(self) -> tuple[str, str, str | None]:
        """Hashable identity for diffing scene graphs — ignores params."""
        return (self.kind, self.subject, self.target)


class SceneSpec(BaseModel):
    """LLM output — a structured plan for the scene.

    The language prompt is decomposed into two full scene graphs:

      * ``initial_scene_graph`` — every relation that holds at env reset.
        This configures where objects spawn.
      * ``final_scene_graph`` — every relation that must hold for the task
        to be considered complete. This is a FULL snapshot, not a diff:
        relations that are unchanged between initial and final must appear
        in both lists (e.g. the bowl stays on the table). Relations that
        are invalidated by the task (the avocado is no longer on the
        table because it is now in the bowl) must be omitted from the
        final graph.
    """

    task_description: str
    background: str
    embodiment: str = "franka_ik"
    items: list[Item]
    initial_scene_graph: list[Relation]
    final_scene_graph: list[Relation]

    @model_validator(mode="after")
    def _graphs_must_differ(self) -> "SceneSpec":
        if not self.goal_added() and not self.goal_removed():
            raise ValueError(
                "initial_scene_graph and final_scene_graph are identical — the task "
                "is trivially solved at reset. At least one relation must differ."
            )
        return self

    def goal_added(self) -> list[Relation]:
        """Relations that must become true to solve the task (final − initial)."""
        initial = {r.identity() for r in self.initial_scene_graph}
        return [r for r in self.final_scene_graph if r.identity() not in initial]

    def goal_removed(self) -> list[Relation]:
        """Relations that must become false to solve the task (initial − final)."""
        final = {r.identity() for r in self.final_scene_graph}
        return [r for r in self.initial_scene_graph if r.identity() not in final]
