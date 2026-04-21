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

from pydantic import BaseModel, Field

# Relation kinds currently surfaced to the LLM. Mirror the subset of
# isaaclab_arena.relations.relations that makes sense for tabletop prompts.
# "in" has no In class in isaaclab_arena.relations.relations yet — see the
# TODO there. Scene builders should skip initial-phase "in" relations and
# materialize goal-phase "in" as the task's success predicate only.
RelationKind = Literal["on", "in", "next_to", "at_position", "is_anchor"]

# "initial" relations configure the starting scene. "goal" relations are the
# task's success condition and must NOT affect initial placement (e.g. the
# avocado starts on the table; it is "in" the bowl only at goal time).
RelationPhase = Literal["initial", "goal"]

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
    """A spatial / structural relation between two items (or on one item).

    `phase` distinguishes the starting scene from the task's success condition:
      * "initial" — placement at env reset. Affects where the object spawns.
      * "goal" — must hold for the task to be considered complete. Does NOT
        affect initial placement; the builder feeds it to the task as the
        success predicate.
    """

    kind: RelationKind
    subject: str
    target: str | None = None
    phase: RelationPhase = "initial"
    params: dict = Field(default_factory=dict)


class SceneSpec(BaseModel):
    """LLM output — a structured plan for the scene.

    Must be fully resolvable against AssetRegistry + the relation set. Nothing
    here is executed directly; the Resolver turns it into concrete asset
    classes + relation instances.
    """

    task_description: str
    background: str
    embodiment: str = "franka_ik"
    items: list[Item]
    relations: list[Relation]
