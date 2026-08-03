# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the plain-dictionary helpers."""

import pytest

from isaaclab_arena.utils.dicts import invert_dict


def test_invert_dict_swaps_keys_and_values():
    """Every value becomes the key it came from."""
    assert invert_dict({"pi0_config": "pi0", "pi05_config": "pi05"}) == {"pi0": "pi0_config", "pi05": "pi05_config"}
    assert invert_dict({}) == {}


def test_invert_dict_rejects_duplicate_values():
    """A mapping that is not one-to-one cannot be inverted."""
    with pytest.raises(AssertionError, match="duplicate values"):
        invert_dict({"pi0_config": "pi0", "pi0_config_copy": "pi0"})
