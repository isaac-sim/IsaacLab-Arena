# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Predicate-group helpers for progress tracking: canonicalize input shapes and render predicates."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Union

PredicateGroups = Union[
    Callable,
    list[Callable],
    list[tuple[Callable, float]],
    dict[str, Callable],
    dict[str, list[Callable]],
    dict[str, list[tuple[Callable, float]]],
]


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
    """Normalize any accepted predicate_groups shape into the canonical form.

    The canonical form is a dict keyed by group name, whose values are the group's ordered
    chain of (predicate, score) pairs. A group is a sequence of predicates that are evaluated in order.

    Accepted input shapes:
      1. func (single callable)                one group with one predicate
      2. [func, func, ...]                     one group, sequential chain
      3. [(func, score), ...]                  one group, sequential chain, weighted
      4. {group: func}                         multiple groups, one predicate each
      5. {group: [func, ...]}                  multiple groups, sequential chains
      6. {group: [(func, score), ...]}         multiple groups, sequential chains, weighted

    Note: #6 is the canonical form.

    Args:
        predicate_groups: The predicates to track, in any of the accepted input shapes above.

    Returns:
        A dict mapping each group name to an ordered list of (predicate, score) pairs.
    """

    if callable(predicate_groups):
        return {DEFAULT_GROUP_NAME: [(predicate_groups, 1.0)]}

    if isinstance(predicate_groups, list):
        assert len(predicate_groups) > 0, "ProgressObjective.predicate_groups list cannot be empty"
        return {DEFAULT_GROUP_NAME: _format_group_chain(predicate_groups, group_name=DEFAULT_GROUP_NAME)}

    if isinstance(predicate_groups, dict):
        assert len(predicate_groups) > 0, "ProgressObjective.predicate_groups dict cannot be empty"
        return {
            group_name: _format_group_chain(value, group_name=group_name)
            for group_name, value in predicate_groups.items()
        }

    raise TypeError(
        f"ProgressObjective.predicate_groups must be a callable, list, or dict; got {type(predicate_groups).__name__}"
    )


def _format_group_chain(value, group_name: str) -> list[tuple[Callable, float]]:
    """Format one group's value into an ordered list of (predicate, score) pairs.

    Accepts a single callable, a list of callables, or a list of (callable, score) tuples. A single
    callable or an unweighted list gets an equal score of 1.0 / number-of-predicates per entry.

    Args:
        value: One group's predicates, as a callable, a list of callables, or a list of
            (callable, score) tuples.
        group_name: Name of the group.

    Returns:
        The group's ordered list of (predicate, score) pairs.
    """

    if callable(value):
        return [(value, 1.0)]
    assert isinstance(
        value, list
    ), f"Predicate chain for group '{group_name}' must be a callable or a list; got {type(value).__name__}"
    assert len(value) > 0, f"Predicate chain for group '{group_name}' cannot be empty"

    first = value[0]
    if isinstance(first, tuple):
        chain = []
        for i, item in enumerate(value):
            assert (
                isinstance(item, tuple) and len(item) == 2
            ), f"Group '{group_name}' index {i}: expected (callable, score) tuple, got {item!r}"
            fn, score = item
            assert callable(fn), f"Group '{group_name}' index {i}: first tuple element must be callable"
            assert isinstance(score, (int, float)), f"Group '{group_name}' index {i}: score must be a number"
            chain.append((fn, float(score)))
        return chain

    if callable(first):
        equal = 1.0 / len(value)
        chain = []
        for i, fn in enumerate(value):
            assert callable(fn), f"Group '{group_name}' index {i}: expected callable, got {type(fn).__name__}"
            chain.append((fn, equal))
        return chain

    raise TypeError(
        f"Group '{group_name}' elements must be callables or (callable, score) tuples; got {type(first).__name__}"
    )


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
