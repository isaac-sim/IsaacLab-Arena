# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Predicate-group helpers for progress tracking: canonicalize input shapes and render predicates."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Union

from isaaclab.managers import TerminationTermCfg

Predicate = Callable | TerminationTermCfg
PredicateGroups = Union[
    Predicate,
    list[Predicate],
    list[tuple[Predicate, float]],
    dict[str, Predicate],
    dict[str, list[Predicate]],
    dict[str, list[tuple[Predicate, float]]],
]


DEFAULT_GROUP_NAME = "default_group"


def _predicate_repr(pred: Predicate) -> str:
    """Generate human-readable string representation for a predicate."""

    if isinstance(pred, TerminationTermCfg):
        pred = functools.partial(pred.func, **pred.params)
    if isinstance(pred, functools.partial):
        fn, args, kwargs = pred.func, pred.args, (pred.keywords or {})
    else:
        fn, args, kwargs = pred, (), {}
    # fn may be a nameless callable (e.g. a callable object), so fall back to repr.
    name = getattr(fn, "__name__", type(fn).__name__)
    parts = [repr(a) for a in args]
    parts += [f"{key}={value!r}" for key, value in kwargs.items() if isinstance(value, (str, int, float, bool))]
    return f"{name}({', '.join(parts)})" if parts else name


def _format_predicate_groups(predicate_groups: PredicateGroups) -> dict[str, list[tuple[Predicate, float]]]:
    """Normalize any accepted predicate_groups shape into the canonical form.

    The canonical form is a dict keyed by group name, whose values are the group's ordered
    chain of (predicate, score) pairs. A group is a sequence of predicates that are evaluated in order.

    Accepted input shapes:
      1. predicate                             one group with one predicate
      2. [predicate, predicate, ...]            one group, sequential chain
      3. [(predicate, score), ...]              one group, sequential chain, weighted
      4. {group: predicate}                    multiple groups, one predicate each
      5. {group: [predicate, ...]}             multiple groups, sequential chains
      6. {group: [(predicate, score), ...]}    multiple groups, sequential chains, weighted

    A predicate may be a callable or a ``TerminationTermCfg`` for a managed predicate.

    Note: #6 is the canonical form.

    Args:
        predicate_groups: The predicates to track, in any of the accepted input shapes above.

    Returns:
        A dict mapping each group name to an ordered list of (predicate, score) pairs.
    """

    if _is_predicate(predicate_groups):
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
        "ProgressObjective.predicate_groups must be a callable, managed term config, list, or dict; got"
        f" {type(predicate_groups).__name__}"
    )


def _is_predicate(value) -> bool:
    """Return whether ``value`` is a supported predicate specification."""

    return callable(value) or isinstance(value, TerminationTermCfg)


def _format_group_chain(value, group_name: str) -> list[tuple[Predicate, float]]:
    """Format one group's value into an ordered list of (predicate, score) pairs.

    Accepts a single predicate, a list of predicates, or a list of (predicate, score) tuples. A
    single predicate or an unweighted list gets an equal score of 1.0 / number-of-predicates per
    entry.

    Args:
        value: One group's predicates, as a predicate, list of predicates, or list of
            (predicate, score) tuples. A predicate is a callable or managed term config.
        group_name: Name of the group.

    Returns:
        The group's ordered list of (predicate, score) pairs.
    """

    if _is_predicate(value):
        return [(value, 1.0)]
    assert isinstance(
        value, list
    ), f"Predicate chain for group '{group_name}' must be a predicate or a list; got {type(value).__name__}"
    assert len(value) > 0, f"Predicate chain for group '{group_name}' cannot be empty"

    first = value[0]
    if isinstance(first, tuple):
        chain = []
        for i, item in enumerate(value):
            assert (
                isinstance(item, tuple) and len(item) == 2
            ), f"Group '{group_name}' index {i}: expected (callable, score) tuple, got {item!r}"
            fn, score = item
            assert _is_predicate(
                fn
            ), f"Group '{group_name}' index {i}: first tuple element must be callable or a managed term config"
            assert isinstance(score, (int, float)), f"Group '{group_name}' index {i}: score must be a number"
            chain.append((fn, float(score)))
        return chain

    if _is_predicate(first):
        equal = 1.0 / len(value)
        chain = []
        for i, fn in enumerate(value):
            assert _is_predicate(
                fn
            ), f"Group '{group_name}' index {i}: expected callable or managed term config, got {type(fn).__name__}"
            chain.append((fn, equal))
        return chain

    raise TypeError(
        f"Group '{group_name}' elements must be predicates or (predicate, score) tuples; got {type(first).__name__}"
    )


def _normalize_scores(
    predicate_groups: dict[str, list[tuple[Predicate, float]]],
) -> dict[str, list[tuple[Predicate, float]]]:
    """Scale each group's scores to sum to 1.0. Zero and negative-sum groups are left untouched."""

    out: dict[str, list[tuple[Predicate, float]]] = {}
    for group, chain in predicate_groups.items():
        total = sum(score for _, score in chain)
        if total <= 0:
            out[group] = list(chain)
            continue
        out[group] = [(fn, score / total) for fn, score in chain]
    return out
