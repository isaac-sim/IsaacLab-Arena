# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task modes that select the GR00T policy data loading and architecture configuration."""

from enum import Enum


# Policy data loader and architecture configuration depend on which task to choose
class TaskMode(Enum):
    """Task the GR00T policy is configured for, which selects its data loader and architecture configuration."""

    G1_LOCOMANIPULATION = "g1_locomanipulation"
    GR1_TABLETOP_MANIPULATION = "gr1_tabletop_manipulation"
    DROID_MANIPULATION = "droid_manipulation"
