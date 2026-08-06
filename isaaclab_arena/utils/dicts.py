# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for working with plain dictionaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")


def invert_dict(mapping: Mapping[KeyT, ValueT]) -> dict[ValueT, KeyT]:
    """Return ``mapping`` with its keys and values swapped; the values must be unique.

    Args:
        mapping: Mapping whose values are hashable and pairwise distinct.

    Returns:
        A new dict mapping each value to the key it came from.
    """
    inverted = {value: key for key, value in mapping.items()}
    assert len(inverted) == len(mapping), f"Cannot invert a mapping with duplicate values: {mapping}"
    return inverted
