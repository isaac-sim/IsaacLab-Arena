# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Axis enum for single-axis placement relations."""

from enum import Enum


class Axis(str, Enum):
    """World axis for single-axis relations."""

    X = "x"
    Y = "y"
    Z = "z"

    @property
    def index(self) -> int:
        """Column of this axis in a position tensor."""
        return "xyz".index(self.value)
