# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Core contracts for registered Arena environment factories."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class ArenaEnvironmentCfg:
    """Mark a typed Arena environment configuration."""


ArenaEnvironmentCfgT = TypeVar("ArenaEnvironmentCfgT", bound=ArenaEnvironmentCfg)


class ArenaEnvironmentFactory(ABC, Generic[ArenaEnvironmentCfgT]):
    """Build a registered Arena environment from a typed configuration."""

    name: str | None = None
    """Registry name of the environment factory."""

    def __init__(self):
        from isaaclab_arena.assets.registries import AssetRegistry, DeviceRegistry, HDRImageRegistry

        self.asset_registry = AssetRegistry()
        self.device_registry = DeviceRegistry()
        self.hdr_registry = HDRImageRegistry()

    @abstractmethod
    def build(self, cfg: ArenaEnvironmentCfgT) -> IsaacLabArenaEnvironment:
        """Build an Arena environment from its typed configuration."""
        pass
