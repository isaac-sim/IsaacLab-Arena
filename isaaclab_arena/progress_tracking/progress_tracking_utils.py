# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Predicate-group helpers for progress tracking: canonicalize input shapes and render predicates."""

from __future__ import annotations

import functools
from collections.abc import Callable

PredicateSequence = list[Callable] | list[tuple[Callable, float]]
PredicateGroups = dict[str, PredicateSequence]


DEFAULT_GROUP_NAME = "default_group"


def _predicate_repr(pred: Callable) -> str:
    """Generate human-readable string representation for a predicate."""

    if isinstance(pred, functools.partial):
        fn, args, kwargs = pred.func, pred.args, (pred.keywords or {})
    else:
        fn, args, kwargs = pred, (), {}
    # fn may be a nameless callable (e.g. a callable object), so fall back to repr.
    name = getattr(fn, "__name__", repr(fn))
    parts = [repr(a) for a in args]
    parts += [f"{key}={value!r}" for key, value in kwargs.items() if isinstance(value, (str, int, float, bool))]
    return f"{name}({', '.join(parts)})" if parts else name


def _format_predicate_groups(predicate_groups: PredicateGroups) -> dict[str, list[tuple[Callable, float]]]:
    """Convert named predicate sequences to lists of predicate-and-score pairs.

    Args:
        predicate_groups: Named, nonempty lists of callables or weighted callable tuples.

    Returns:
        A dict mapping each group name to an ordered list of (predicate, score) pairs.
    """

    assert isinstance(predicate_groups, dict), "ProgressObjective.predicate_groups must be a dictionary of sequences."
    assert predicate_groups, "ProgressObjective.predicate_groups cannot be empty."
    assert all(isinstance(group_name, str) for group_name in predicate_groups), "Predicate group names must be strings."
    return {
        group_name: _format_group_chain(sequence, group_name=group_name)
        for group_name, sequence in predicate_groups.items()
    }


def _format_group_chain(sequence: PredicateSequence, group_name: str) -> list[tuple[Callable, float]]:
    """Format one group's value into an ordered list of (predicate, score) pairs.

    Args:
        sequence: A nonempty list of callables or (callable, score) tuples.
        group_name: Name of the group.

    Returns:
        The group's ordered list of (predicate, score) pairs.
    """

    assert isinstance(
        sequence, list
    ), f"Predicate sequence for group '{group_name}' must be a list; got {type(sequence).__name__}"
    assert sequence, f"Predicate sequence for group '{group_name}' cannot be empty"

    if isinstance(sequence[0], tuple):
        chain = []
        for predicate_index, item in enumerate(sequence):
            assert (
                isinstance(item, tuple) and len(item) == 2
            ), f"Group '{group_name}' index {predicate_index}: expected (callable, score) tuple, got {item!r}"
            predicate, score = item
            assert callable(predicate), f"Group '{group_name}' index {predicate_index}: predicate must be callable"
            assert isinstance(
                score, (int, float)
            ), f"Group '{group_name}' index {predicate_index}: score must be a number"
            chain.append((predicate, float(score)))
        return chain

    chain = []
    for predicate_index, predicate in enumerate(sequence):
        assert callable(
            predicate
        ), f"Group '{group_name}' index {predicate_index}: expected callable, got {type(predicate).__name__}"
        chain.append((predicate, 1.0))
    return chain


def _normalize_scores(
    predicate_groups: dict[str, list[tuple[Callable, float]]],
) -> dict[str, list[tuple[Callable, float]]]:
    """Scale each group's scores to sum to 1.0. Zero and negative-sum groups are left untouched."""

    out: dict[str, list[tuple[Callable, float]]] = {}
    for group, chain in predicate_groups.items():
        total = sum(score for _, score in chain)
        if total <= 0:
            out[group] = list(chain)
            continue
        out[group] = [(fn, score / total) for fn, score in chain]
    return out
