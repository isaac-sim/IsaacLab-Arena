# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Normalize predicate sequences and render predicates for progress tracking."""

from __future__ import annotations

import functools
from collections.abc import Callable

from isaaclab.managers import TerminationTermCfg

Predicate = Callable | TerminationTermCfg
PredicateSequence = list[Predicate] | list[tuple[Predicate, float]]
PredicateSequences = dict[str, PredicateSequence]


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


def _format_predicate_sequences(
    predicate_sequences: PredicateSequences,
) -> dict[str, list[tuple[Predicate, float]]]:
    """Convert named predicate sequences to weighted sequences.

    Args:
        predicate_sequences: A nonempty dictionary of named predicate lists.
            Each list contains predicates or (predicate, score) pairs.

    Returns:
        Named lists of (predicate, score) pairs.
    """

    assert isinstance(predicate_sequences, dict), "predicate_sequences must map names to predicate sequences."
    assert predicate_sequences, "ProgressObjective.predicate_sequences cannot be empty."
    assert all(
        isinstance(sequence_name, str) for sequence_name in predicate_sequences
    ), "Predicate sequence names must be strings."
    return {
        sequence_name: _format_predicate_sequence(sequence, sequence_name=sequence_name)
        for sequence_name, sequence in predicate_sequences.items()
    }


def _is_predicate(value) -> bool:
    """Return whether a value is a callable or a managed predicate config."""
    return callable(value) or isinstance(value, TerminationTermCfg)


def _format_predicate_sequence(sequence: PredicateSequence, sequence_name: str) -> list[tuple[Predicate, float]]:
    """Format one sequence into an ordered list of (predicate, score) pairs.

    Args:
        sequence: A nonempty list of predicates or (predicate, score) tuples.
        sequence_name: Name of the sequence.

    Returns:
        The sequence's ordered list of (predicate, score) pairs.
    """

    assert isinstance(
        sequence, list
    ), f"Predicate sequence '{sequence_name}' must be a list; got {type(sequence).__name__}"
    assert sequence, f"Predicate sequence '{sequence_name}' cannot be empty"

    if isinstance(sequence[0], tuple):
        chain = []
        for predicate_index, item in enumerate(sequence):
            assert (
                isinstance(item, tuple) and len(item) == 2
            ), f"Sequence '{sequence_name}' index {predicate_index}: expected (callable, score) tuple, got {item!r}"
            predicate, score = item
            assert _is_predicate(
                predicate
            ), f"Sequence '{sequence_name}' index {predicate_index}: expected a callable or TerminationTermCfg"
            assert isinstance(
                score, (int, float)
            ), f"Sequence '{sequence_name}' index {predicate_index}: score must be a number"
            chain.append((predicate, float(score)))
        return chain

    chain = []
    for predicate_index, predicate in enumerate(sequence):
        assert _is_predicate(
            predicate
        ), f"Sequence '{sequence_name}' index {predicate_index}: expected a callable or TerminationTermCfg"
        chain.append((predicate, 1.0))
    return chain


def _normalize_scores(
    predicate_sequences: dict[str, list[tuple[Predicate, float]]],
) -> dict[str, list[tuple[Predicate, float]]]:
    """Scale each sequence's scores to sum to 1.0. Leave zero and negative-sum sequences untouched."""

    normalized_sequences: dict[str, list[tuple[Predicate, float]]] = {}
    for sequence_name, sequence in predicate_sequences.items():
        total = sum(score for _, score in sequence)
        if total <= 0:
            normalized_sequences[sequence_name] = list(sequence)
            continue
        normalized_sequences[sequence_name] = [(predicate, score / total) for predicate, score in sequence]
    return normalized_sequences
