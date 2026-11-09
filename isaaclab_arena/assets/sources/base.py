# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Interface for acquiring USD assets."""

from __future__ import annotations

from abc import ABC, abstractmethod


class UsdSource(ABC):
    """Resolve a USD asset without coupling consumers to its acquisition mechanism."""

    @abstractmethod
    def resolve(self) -> str:
        """Return a usable local or remote USD path."""
