# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Evaluation protocol's three-valued truth model."""

import pytest

from isaaclab_arena.evaluation.protocol.truth import TruthValue, all_truth, any_truth


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], TruthValue.TRUE),
        ([TruthValue.TRUE, TruthValue.TRUE], TruthValue.TRUE),
        ([TruthValue.TRUE, TruthValue.UNKNOWN], TruthValue.UNKNOWN),
        ([TruthValue.UNKNOWN, TruthValue.UNKNOWN], TruthValue.UNKNOWN),
        ([TruthValue.TRUE, TruthValue.FALSE, TruthValue.UNKNOWN], TruthValue.FALSE),
    ],
)
def test_all_truth_uses_three_valued_conjunction(values: list[TruthValue], expected: TruthValue):
    assert all_truth(values) is expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], TruthValue.FALSE),
        ([TruthValue.FALSE, TruthValue.FALSE], TruthValue.FALSE),
        ([TruthValue.FALSE, TruthValue.UNKNOWN], TruthValue.UNKNOWN),
        ([TruthValue.UNKNOWN, TruthValue.UNKNOWN], TruthValue.UNKNOWN),
        ([TruthValue.FALSE, TruthValue.TRUE, TruthValue.UNKNOWN], TruthValue.TRUE),
    ],
)
def test_any_truth_uses_three_valued_disjunction(values: list[TruthValue], expected: TruthValue):
    assert any_truth(values) is expected


@pytest.mark.parametrize("reducer", [all_truth, any_truth])
def test_truth_reducers_reject_boolean_values(reducer):
    with pytest.raises(TypeError):
        reducer([True])
