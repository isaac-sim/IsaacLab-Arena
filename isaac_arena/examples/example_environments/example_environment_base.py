# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# if TYPE_CHECKING:
#     from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment


def add_argument_if_missing(parser: argparse.ArgumentParser, flag: str, **kwargs):
    """Add a --flag argument only if it's not already defined."""
    assert flag.startswith("--"), "Only long flags are supported"
    # strip leading dashes to get the dest
    dest = kwargs.get("dest", flag.lstrip("-").replace("-", "_"))
    for action in parser._actions:
        if action.dest == dest:
            # if argument already exists then return it
            return action
    return parser.add_argument(flag, **kwargs)


class ExampleEnvironmentBase(ABC):

    name: str | None = None

    def __init__(self):
        from isaac_arena.assets.asset_registry import AssetRegistry, DeviceRegistry

        self.asset_registry = AssetRegistry()
        self.device_registry = DeviceRegistry()

    @abstractmethod
    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacArenaEnvironment:
        pass

    @abstractmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        pass
