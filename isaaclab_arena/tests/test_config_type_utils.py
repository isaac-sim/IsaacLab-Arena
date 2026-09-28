# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for structured-config annotation utilities."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import pytest

from isaaclab_arena.hydra.config_type_utils import annotation_accepts_value, convert_override_value, field_annotation


class _Mode(Enum):
    FIRST = "first"
    SECOND = "second"


@dataclass
class _BaseCfg:
    count: int = 1


@dataclass
class _Cfg(_BaseCfg):
    ratio: float = 0.5


def test_field_annotation_resolves_inherited_fields():
    assert field_annotation(_Cfg, "count") is int
    assert field_annotation(_Cfg, "ratio") is float


@pytest.mark.parametrize(
    ("annotation", "value", "expected"),
    [
        (_Mode, "second", _Mode.SECOND),
        (float, 1, 1.0),
        (set[str], ["a", "b"], {"a", "b"}),
        (tuple[float, float], [1, 2.5], (1.0, 2.5)),
    ],
)
def test_convert_override_value(annotation, value, expected):
    converted = convert_override_value(annotation, value)

    assert converted == expected
    assert annotation_accepts_value(annotation, converted)


@pytest.mark.parametrize(
    ("annotation", "value"),
    [
        (float, "fast"),
        (float, True),
        (int, True),
        (set[str], [1]),
        (_Mode, "missing"),
        (Literal["first", "second"], "missing"),
    ],
)
def test_incompatible_override_value_is_rejected(annotation, value):
    converted = convert_override_value(annotation, value)

    assert not annotation_accepts_value(annotation, converted)
