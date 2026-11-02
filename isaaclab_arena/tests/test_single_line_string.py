# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the single-line string check. No Isaac Sim or GPU required."""

import pytest

from isaaclab_arena.evaluation.single_line_string import is_single_line_nonempty_string


@pytest.mark.parametrize("value", ["benchmark", "my benchmark", " ", "1e3"])
def test_accepts_non_empty_strings_without_line_breaks(value):
    assert is_single_line_nonempty_string(value)


@pytest.mark.parametrize("value", ["", "two\nlines", "carriage\rreturn", None, 5, True])
def test_rejects_empty_multi_line_and_non_string_values(value):
    assert not is_single_line_nonempty_string(value)
