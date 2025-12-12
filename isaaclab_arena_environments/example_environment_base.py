# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
from abc import ABC, abstractmethod

# NOTE(alexmillane, 2025.09.04): There is an issue with type annotation in this file.
# We cannot annotate types which require the simulation app to be started in order to
# import, because this file is used to retrieve CLI arguments, so it must be imported
# before the simulation app is started.
# TODO(alexmillane, 2025.09.04): Fix this.


class ExampleEnvironmentBase(ABC):

    name: str | None = None

    def __init__(self):
        from isaaclab_arena.assets.asset_registry import AssetRegistry, DeviceRegistry

        self.asset_registry = AssetRegistry()
        self.device_registry = DeviceRegistry()

    @abstractmethod
    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        pass

    @abstractmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        pass
