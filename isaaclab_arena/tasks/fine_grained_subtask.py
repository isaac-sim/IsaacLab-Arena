# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Declarative unit of fine-grained progress inside a task.

A ``FineGrainedSubtask`` is a named program for the per-step state machine. It
declares one or more *groups* (e.g. one per object) where each group is a
sequential chain of predicates that must succeed in order. Across groups,
``logical`` picks the aggregation rule ("all" / "any" / "choose K").

This is intentionally a thin data carrier: validation + normalization happens
in ``__post_init__``; everything else (per-step ticking, transition events,
publishing to ``env.extras``) lives in ``fine_grained_state_machine.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Union


SubtaskConditions = Union[
    Callable,
    list[Callable],
    list[tuple[Callable, float]],
    dict[str, Callable],
    dict[str, list[Callable]],
    dict[str, list[tuple[Callable, float]]],
]


DEFAULT_GROUP_NAME = "default"


def sanitize_conditions(conditions: SubtaskConditions) -> dict[str, list[tuple[Callable, float]]]:
    """Normalize the many accepted ``conditions`` shapes to the canonical form.

    Canonical form: ``dict[group_name -> list[(callable, score)]]`` with one
    group per logical-aggregation unit and within-group entries treated as a
    sequential chain. Equal scores are assigned where the input did not specify
    them; scores are *not* normalized here — see ``normalize_scores``.

    Accepted input shapes:
      1. ``func`` (single callable)                — one group with one predicate.
      2. ``[func, func, ...]``                     — one group, sequential chain.
      3. ``[(func, score), ...]``                  — same, weighted.
      4. ``{group: func}``                         — multiple groups, one predicate each.
      5. ``{group: [func, ...]}``                  — multiple groups, sequential chains, equal scores.
      6. ``{group: [(func, score), ...]}``         — multiple groups, weighted.
    """
    if callable(conditions):
        return {DEFAULT_GROUP_NAME: [(conditions, 1.0)]}

    if isinstance(conditions, list):
        if len(conditions) == 0:
            raise ValueError("FineGrainedSubtask.conditions list cannot be empty")
        return {DEFAULT_GROUP_NAME: _sanitize_group_chain(conditions, group_name=DEFAULT_GROUP_NAME)}

    if isinstance(conditions, dict):
        if len(conditions) == 0:
            raise ValueError("FineGrainedSubtask.conditions dict cannot be empty")
        return {
            group_name: _sanitize_group_chain(value, group_name=group_name)
            for group_name, value in conditions.items()
        }

    raise TypeError(
        f"FineGrainedSubtask.conditions must be a callable, list, or dict; got {type(conditions).__name__}"
    )


def _sanitize_group_chain(value, group_name: str) -> list[tuple[Callable, float]]:
    if callable(value):
        return [(value, 1.0)]
    if not isinstance(value, list):
        raise TypeError(
            f"Conditions for group '{group_name}' must be a callable or a list; got {type(value).__name__}"
        )
    if len(value) == 0:
        raise ValueError(f"Conditions for group '{group_name}' cannot be empty")

    first = value[0]
    if isinstance(first, tuple):
        chain = []
        for i, item in enumerate(value):
            if not (isinstance(item, tuple) and len(item) == 2):
                raise TypeError(
                    f"Group '{group_name}' index {i}: expected (callable, score) tuple, got {item!r}"
                )
            fn, score = item
            if not callable(fn):
                raise TypeError(
                    f"Group '{group_name}' index {i}: first tuple element must be callable"
                )
            if not isinstance(score, (int, float)):
                raise TypeError(
                    f"Group '{group_name}' index {i}: score must be a number"
                )
            chain.append((fn, float(score)))
        return chain

    if callable(first):
        equal = 1.0 / len(value)
        chain = []
        for i, fn in enumerate(value):
            if not callable(fn):
                raise TypeError(
                    f"Group '{group_name}' index {i}: expected callable, got {type(fn).__name__}"
                )
            chain.append((fn, equal))
        return chain

    raise TypeError(
        f"Group '{group_name}' elements must be callables or (callable, score) tuples; got {type(first).__name__}"
    )


def normalize_scores(
    conditions: dict[str, list[tuple[Callable, float]]],
) -> dict[str, list[tuple[Callable, float]]]:
    """Scale each group's scores to sum to 1.0. Zero/negative-sum groups are left untouched."""
    out: dict[str, list[tuple[Callable, float]]] = {}
    for group, chain in conditions.items():
        total = sum(score for _, score in chain)
        if total <= 0:
            out[group] = list(chain)
            continue
        out[group] = [(fn, score / total) for fn, score in chain]
    return out


@dataclass
class FineGrainedSubtask:
    name: str
    conditions: SubtaskConditions
    score: float = 1.0
    logical: Literal["all", "any", "choose"] = "all"
    K: int | None = None
    description: str | None = None
    # Index of the parent composite-task subtask this recipe belongs to. Set
    # automatically by ``CompositeTaskBase.get_fine_grained_subtasks`` during
    # concatenation; ``None`` for top-level recipes (atomic tasks or
    # composite-level recipes added via ``get_own_fine_grained_subtasks``).
    # When non-None *and* the env exposes ``_current_subtask_idx`` (i.e. the
    # parent is a ``SequentialTaskBase``), the state machine only advances
    # this recipe for envs whose current parent-subtask index matches.
    parent_subtask_idx: int | None = None
    canonical_conditions: dict[str, list[tuple[Callable, float]]] = field(init=False, repr=False)

    def __post_init__(self):
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"FineGrainedSubtask '{self.name}': score must be in [0, 1], got {self.score}")
        if self.logical not in ("all", "any", "choose"):
            raise ValueError(
                f"FineGrainedSubtask '{self.name}': logical must be 'all'/'any'/'choose', got {self.logical!r}"
            )

        sanitized = sanitize_conditions(self.conditions)
        normalized = normalize_scores(sanitized)
        self.canonical_conditions = normalized

        num_groups = len(self.canonical_conditions)
        if self.logical == "choose":
            if self.K is None:
                raise ValueError(
                    f"FineGrainedSubtask '{self.name}': K is required when logical='choose'"
                )
            if not (1 <= self.K <= num_groups):
                raise ValueError(
                    f"FineGrainedSubtask '{self.name}': K={self.K} must be in [1, {num_groups}]"
                )

    @property
    def group_names(self) -> list[str]:
        return list(self.canonical_conditions.keys())

    def get_chain(self, group_name: str) -> list[tuple[Callable, float]]:
        return self.canonical_conditions[group_name]
