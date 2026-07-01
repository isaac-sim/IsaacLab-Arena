# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lazy, injectable physics prim index for any USD-backed asset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from isaaclab_arena.agentic_environment_generation.background_physics_catalog import (
    PhysicsPrimEntry,
    list_physics_prim_entries,
)


@dataclass
class UsdPrimIndex:
    """Cache physics prim listings for one USD file."""

    usd_path: str | Path
    entries: list[PhysicsPrimEntry] | None = None

    def get_usd_path(self) -> str:
        """Return the indexed USD path as a string."""
        return str(self.usd_path)

    def list_entries(self) -> list[PhysicsPrimEntry]:
        """Return physics prim entries, listing the USD once."""
        if self.entries is None:
            self.entries = list_physics_prim_entries(self.get_usd_path())
        return self.entries
