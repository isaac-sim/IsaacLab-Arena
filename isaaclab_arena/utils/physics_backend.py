# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Physics backend identifiers shared across Arena layers."""

from enum import Enum


class PhysicsBackend(str, Enum):
    """Physics backends supported by Arena."""

    PHYSX = "physx"
    NEWTON = "newton"

    def __str__(self) -> str:
        return self.value
