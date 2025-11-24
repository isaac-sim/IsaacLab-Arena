# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod


class AffordanceBase(ABC):
    """Base class for affordances."""

    @property
    @abstractmethod
    def name(self) -> str:
        # NOTE(alexmillane, 2025.09.19) Affordances always have be combined with
        # an Asset which has a "name" property. By declaring this property
        # abstract here, we enforce this.
        pass
