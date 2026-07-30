# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Three-valued truth operations used by evaluation requirements."""

from collections.abc import Iterable
from enum import StrEnum


class TruthValue(StrEnum):
    """Truth state of a predicate whose evidence may be unavailable."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


def all_truth(values: Iterable[TruthValue]) -> TruthValue:
    """Combine truth values using three-valued conjunction.

    Args:
        values: Truth values to combine.
    """
    saw_unknown = False
    for value in values:
        if not isinstance(value, TruthValue):
            raise TypeError("all_truth values must be TruthValue instances")
        if value is TruthValue.FALSE:
            return TruthValue.FALSE
        if value is TruthValue.UNKNOWN:
            saw_unknown = True
    return TruthValue.UNKNOWN if saw_unknown else TruthValue.TRUE


def any_truth(values: Iterable[TruthValue]) -> TruthValue:
    """Combine truth values using three-valued disjunction.

    Args:
        values: Truth values to combine.
    """
    saw_unknown = False
    for value in values:
        if not isinstance(value, TruthValue):
            raise TypeError("any_truth values must be TruthValue instances")
        if value is TruthValue.TRUE:
            return TruthValue.TRUE
        if value is TruthValue.UNKNOWN:
            saw_unknown = True
    return TruthValue.UNKNOWN if saw_unknown else TruthValue.FALSE
