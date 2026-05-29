# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Schema the agent must fill in when parsing a natural-language env-generation prompt.

The agent sees a list of the *available* asset tags / embodiment names pulled
from the registries at call time, and must return an EnvIntentSpec that only uses
those vocabularies. Concrete asset names are resolved in a second, deterministic
step — the agent never invents USD paths.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Relation kinds currently surfaced to the agent.
# Should be a subset of ``ArenaEnvGraphSpatialConstraintType``.
RelationKind = Literal["on", "in", "next_to", "at_position", "at_pose", "is_anchor"]

ItemRole = Literal["foreground", "distractor", "anchor"]

# Task kinds the agent can propose as an atomic task.
# Should be a subset of ``ArenaEnvGraphTaskConstraintType``.
TaskKind = Literal["pick_and_place", "open_door", "close_door"]


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
    scale: float | None = Field(
        default=None,
        description=(
            "Spawn scale. Leave null (the default) so the placement proposer "
            "auto-fits the asset; only set a positive float when the prompt "
            "explicitly demands a size override."
        ),
    )


class Relation(BaseModel):
    """A spatial relation between items.

    Binary kinds (``on``, ``in``, ``next_to``, ...) must set ``target`` to the
    other item — semantics is "subject is in relation to target". Unary kinds
    (``is_anchor``, ``at_position``, ...) describe an intrinsic property of
    ``subject`` alone and must leave ``target`` as ``None``.
    """

    kind: RelationKind = Field(
        description=(
            "Spatial relation only — articulated-state changes (open/close) are expressed via tasks, not via relations."
        ),
    )
    subject: str = Field(
        description="Item the relation applies to, named by its Item.query string or the background name.",
    )
    target: str | None = Field(
        default=None,
        description=(
            "The other item the relation is anchored on for binary kinds "
            "(on / in / next_to / at_position / at_pose); leave null for "
            "unary kinds (is_anchor)."
        ),
    )
    # TODO(qianl): free-form ``dict`` emits ``additionalProperties: true``,
    # which strict-mode structured-outputs endpoints (OpenAI strict /
    # Bedrock-Claude) reject with a 400. The default NVIDIA DeepSeek is
    # lenient and accepts it, so this is a latent portability landmine.
    params: dict = Field(
        default_factory=dict,
        description="Optional kind-specific parameters; leave empty by default.",
    )

    def identity(self) -> tuple[str, str, str | None]:
        """Hashable identity for diffing scene graphs — ignores params."""
        return (self.kind, self.subject, self.target)


class Task(BaseModel):
    """One atomic task in the plan that transforms the env state."""

    kind: TaskKind = Field(description="The action to perform.")
    subject: str = Field(
        description=(
            "The primary object the task acts on, named by its Item.query string (e.g. 'avocado', 'microwave')."
        ),
    )
    target: str | None = Field(
        default=None,
        description=(
            "The secondary object or location, named by its Item.query "
            "string or the background name. Leave null for unary tasks "
            "(open_door / close_door)."
        ),
    )
    description: str = Field(
        description="Natural-language summary of the task (e.g. 'pick up the avocado and place it in the bowl').",
    )


class EnvIntentSpec(BaseModel):
    """Agent output — a structured "env intent" (blueprint) for the env and a list of tasks.

    Field-level guidance lives on the individual ``Field(description=...)``
    entries below and is surfaced to the agent via ``model_json_schema()``;
    only cross-cutting rules and few-shot examples are kept in the
    prompt text (see ``EnvGenAgent._system_prompt``).
    """

    # Forced chain-of-thought field, listed FIRST so the agent emits its
    # analysis before committing to any structured field. Instruction-tuned
    # models respect schema field order, and writing reasoning before
    # answers measurably improves structured-output quality (the
    # "think step by step then commit" pattern). Bonus debuggability:
    # when a downstream resolver step fails, the reasoning trace shows
    # which step the model got wrong (e.g. it picked "tomato" because
    # it misidentified the foreground object as a vegetable) — without
    # this, the only signal is the malformed spec itself.
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
        description="One-sentence natural-language summary of what the env exercises overall."
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
            "FULL snapshot of all relations in the starting state. Every "
            "persistent relation (e.g. bowl on table, distractors present) "
            "must appear here. Relations that change via tasks are still "
            "listed here in their starting form."
        ),
    )
    tasks: list[Task] = Field(
        description=(
            "Tasks to execute in sequence. The task sequence implicitly "
            "defines the intermediate env graphs by applying each task's "
            "transformations in order. An empty list is valid and means "
            "the env has no task — at the arena layer this maps to the "
            "``NoTask`` null object (e.g. a static playground / sandbox "
            "env). Prefer an empty list over inventing a placeholder "
            "task when the user prompt genuinely describes a task-less "
            "scene."
        ),
    )
