# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Interface for acquiring USD assets."""

from __future__ import annotations

from abc import ABC, abstractmethod


class UsdSource(ABC):
    """Acquire a local USD asset without coupling consumers to its source."""

    @abstractmethod
    def to_local_usd_path(self) -> str:
        """Return the local path to the acquired USD asset."""
