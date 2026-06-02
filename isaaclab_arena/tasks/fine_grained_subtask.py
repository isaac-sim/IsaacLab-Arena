# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Union

PredicateGroups = Union[
    Callable,
    list[Callable],
    list[tuple[Callable, float]],
    dict[str, Callable],
    dict[str, list[Callable]],
    dict[str, list[tuple[Callable, float]]],
]


DEFAULT_GROUP_NAME = "default"


def sanitize_predicate_groups(predicate_groups: PredicateGroups) -> dict[str, list[tuple[Callable, float]]]:
    """Normalize predicate_groups to the canonical form.

    Canonical form: ``dict[group_name: list[(callable, score)]]``.

    Accepted input shapes:
      1. func (single callable)                — one group with one predicate
      2. [func, func, ...]                     — one group, sequential chain
      3. [(func, score), ...]                  — one group, sequential chain, weighted
      4. {group: func}                         — multiple groups, one predicate each
      5. {group: [func, ...]}                  — multiple groups, sequential chains
      6. {group: [(func, score), ...]}         — multiple groups, sequential chains, weighted
    """
    if callable(predicate_groups):
        return {DEFAULT_GROUP_NAME: [(predicate_groups, 1.0)]}

    if isinstance(predicate_groups, list):
        if len(predicate_groups) == 0:
            raise ValueError("FineGrainedSubtask.predicate_groups list cannot be empty")
        return {DEFAULT_GROUP_NAME: _sanitize_group_chain(predicate_groups, group_name=DEFAULT_GROUP_NAME)}

    if isinstance(predicate_groups, dict):
        if len(predicate_groups) == 0:
            raise ValueError("FineGrainedSubtask.predicate_groups dict cannot be empty")
        return {
            group_name: _sanitize_group_chain(value, group_name=group_name)
            for group_name, value in predicate_groups.items()
        }

    raise TypeError(
        f"FineGrainedSubtask.predicate_groups must be a callable, list, or dict; got {type(predicate_groups).__name__}"
    )


def _sanitize_group_chain(value, group_name: str) -> list[tuple[Callable, float]]:
    if callable(value):
        return [(value, 1.0)]
    if not isinstance(value, list):
        raise TypeError(
            f"Predicate chain for group '{group_name}' must be a callable or a list; got {type(value).__name__}"
        )
    if len(value) == 0:
        raise ValueError(f"Predicate chain for group '{group_name}' cannot be empty")

    first = value[0]
    if isinstance(first, tuple):
        chain = []
        for i, item in enumerate(value):
            if not (isinstance(item, tuple) and len(item) == 2):
                raise TypeError(f"Group '{group_name}' index {i}: expected (callable, score) tuple, got {item!r}")
            fn, score = item
            if not callable(fn):
                raise TypeError(f"Group '{group_name}' index {i}: first tuple element must be callable")
            if not isinstance(score, (int, float)):
                raise TypeError(f"Group '{group_name}' index {i}: score must be a number")
            chain.append((fn, float(score)))
        return chain

    if callable(first):
        equal = 1.0 / len(value)
        chain = []
        for i, fn in enumerate(value):
            if not callable(fn):
                raise TypeError(f"Group '{group_name}' index {i}: expected callable, got {type(fn).__name__}")
            chain.append((fn, equal))
        return chain

    raise TypeError(
        f"Group '{group_name}' elements must be callables or (callable, score) tuples; got {type(first).__name__}"
    )


def normalize_scores(
    predicate_groups: dict[str, list[tuple[Callable, float]]],
) -> dict[str, list[tuple[Callable, float]]]:
    """Scale each group's scores to sum to 1.0. Zero/negative-sum groups are left untouched."""
    out: dict[str, list[tuple[Callable, float]]] = {}
    for group, chain in predicate_groups.items():
        total = sum(score for _, score in chain)
        if total <= 0:
            out[group] = list(chain)
            continue
        out[group] = [(fn, score / total) for fn, score in chain]
    return out


@dataclass
class FineGrainedSubtask:
    """Declarative recipe for one tracked progression inside a task.

    A FineGrainedSubtask declares what the predicate state machine should track.
    Each FineGrainedSubtask holds one or more sequential predicate chains (groups).
    Within a group, predicates run in order. Across groups, predicates run in parallel.

    Args:
        name: Identifies the FineGrainedSubtask within the TaskBase.
        predicate_groups: The sequential predicate chains that define the FineGrainedSubtask.
        score: Weight of the FineGrainedSubtask in the TaskBase-level overall_score.
        logical: How completed groups combine to determine if the FineGrainedSubtask is complete.
            Can be "all", "any", or "choose"
        K: Required when logical == "choose". Specifies the number of groups that must be completed
            to consider the FineGrainedSubtask complete.
        description: An optional description of the FineGrainedSubtask.
    """

    name: str
    predicate_groups: PredicateGroups
    score: float = 1.0
    logical: Literal["all", "any", "choose"] = "all"
    K: int | None = None
    description: str | None = None

    canonical_predicate_groups: dict[str, list[tuple[Callable, float]]] = field(init=False, repr=False)

    # Index of the parent TaskBase this recipe belongs to. Set automatically when used with composite tasks.
    parent_subtask_idx: int | None = None

    def __post_init__(self):
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"FineGrainedSubtask '{self.name}': score must be in [0, 1], got {self.score}")
        if self.logical not in ("all", "any", "choose"):
            raise ValueError(
                f"FineGrainedSubtask '{self.name}': logical must be 'all'/'any'/'choose', got {self.logical!r}"
            )

        sanitized = sanitize_predicate_groups(self.predicate_groups)
        normalized = normalize_scores(sanitized)
        self.canonical_predicate_groups = normalized

        num_groups = len(self.canonical_predicate_groups)
        if self.logical == "choose":
            if self.K is None:
                raise ValueError(f"FineGrainedSubtask '{self.name}': K is required when logical='choose'")
            if not (1 <= self.K <= num_groups):
                raise ValueError(f"FineGrainedSubtask '{self.name}': K={self.K} but must be in [1, {num_groups}]")

    @property
    def group_names(self) -> list[str]:
        """Returns the names of the groups in the FineGrainedSubtask."""
        return list(self.canonical_predicate_groups.keys())

    def get_chain(self, group_name: str) -> list[tuple[Callable, float]]:
        """Returns the chain of predicates for a given group."""
        return self.canonical_predicate_groups[group_name]
