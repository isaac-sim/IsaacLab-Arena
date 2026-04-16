# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


class ExampleEnvironmentBase(ABC):

    name: str | None = None

    def __init__(self):
        from isaaclab_arena.assets.asset_registry import AssetRegistry, DeviceRegistry, HDRImageRegistry

        self.asset_registry = AssetRegistry()
        self.device_registry = DeviceRegistry()
        self.hdr_registry = HDRImageRegistry()

    @abstractmethod
    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        pass

    @abstractmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        pass
