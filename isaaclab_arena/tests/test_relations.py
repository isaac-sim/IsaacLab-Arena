# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.relations import NextTo, NotNextTo, Side, coerce_side
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("left", Side.NEGATIVE_X),
        ("right", Side.POSITIVE_X),
        ("front", Side.NEGATIVE_Y),
        ("back", Side.POSITIVE_Y),
        ("positive_x", Side.POSITIVE_X),
        ("NEGATIVE_Y", Side.NEGATIVE_Y),
        (Side.POSITIVE_X, Side.POSITIVE_X),
    ],
)
def test_coerce_side_accepts_aliases_and_enum_values(raw, expected):
    assert coerce_side(raw) == expected


def test_next_to_and_not_next_to_coerce_side_alias():
    parent = DummyObject(
        name="parent",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.1)),
    )

    next_to = NextTo(parent, side="left")
    not_next_to = NotNextTo(parent, side="right")

    assert next_to.side == Side.NEGATIVE_X
    assert not_next_to.side == Side.POSITIVE_X
