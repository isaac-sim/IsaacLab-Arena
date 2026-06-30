# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lazy, injectable background physics prim index."""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab_arena.agentic_environment_generation.background_physics_catalog import (
    PhysicsPrimEntry,
    list_physics_prim_entries,
    resolve_background_usd_path,
)
from isaaclab_arena.assets.registries import AssetRegistry


@dataclass
class BackgroundPrimIndex:
    """Lazily list physics prims for a resolved background asset."""

    background_name: str
    registry: AssetRegistry
    entries: list[PhysicsPrimEntry] | None = None
    usd_path: str | None = None

    def get_usd_path(self) -> str:
        """Return the background USD path, resolving it once."""
        if self.usd_path is None:
            self.usd_path = resolve_background_usd_path(self.registry, self.background_name)
        return self.usd_path

    def list_entries(self) -> list[PhysicsPrimEntry]:
        """Return physics prim entries, listing the USD once."""
        if self.entries is None:
            self.entries = list_physics_prim_entries(self.get_usd_path())
        return self.entries
