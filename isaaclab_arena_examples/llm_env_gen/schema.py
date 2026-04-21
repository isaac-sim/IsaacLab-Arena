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
# Extend as more relations are validated end-to-end.
RelationKind = Literal["on", "next_to", "at_position", "is_anchor"]

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
