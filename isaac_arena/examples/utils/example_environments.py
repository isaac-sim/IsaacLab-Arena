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

from isaaclab.envs import ManagerBasedRLEnvCfg

from isaac_arena.assets.asset_registry import AssetRegistry, DeviceRegistry
from isaac_arena.embodiments.gr1t2.gr1t2 import GR1T2Embodiment
from isaac_arena.environments.compile_env import ArenaEnvBuilder
from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
from isaac_arena.geometry.pose import Pose
from isaac_arena.scene.scene import Scene
from isaac_arena.tasks.open_door_task import OpenDoorTask


def add_argument_if_missing(parser: argparse.ArgumentParser, flag: str, **kwargs):
    """Add a --flag argument only if it's not already defined."""
    assert flag.startswith("--"), "Only long flags are supported"
    # strip leading dashes to get the dest
    dest = kwargs.get("dest", flag.lstrip("-").replace("-", "_"))
    for action in parser._actions:
        if action.dest == dest:
            return action  # already exists → return it
    return parser.add_argument(flag, **kwargs)


class ExampleEnvironment(ABC):

    name: str | None = None

    def __init__(self):
        self.asset_registry = AssetRegistry()
        self.device_registry = DeviceRegistry()

    @abstractmethod
    def get_env(self, args_cli: argparse.Namespace) -> IsaacArenaEnvironment:
        pass

    @abstractmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        pass


class Gr1OpenMicrowaveEnvironment(ExampleEnvironment):

    name: str = "gr1_open_microwave"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacArenaEnvironment:

        background = self.asset_registry.get_asset_by_name("packing_table_pick_and_place")()
        microwave = self.asset_registry.get_asset_by_name("microwave")()
        teleop_device = self.device_registry.get_device_by_name("avp")()
        assets = [background, microwave]

        # Put the microwave on the packing table.
        microwave_pose = Pose(
            position_xyz=(0.7, -0.00586, 0.22773),
            rotation_wxyz=(0.7071068, 0, 0, -0.7071068),
        )
        microwave.set_initial_pose(microwave_pose)

        # Optionally add another object
        if args_cli.object is not None:
            object = self.asset_registry.get_asset_by_name(args_cli.object)()
            object_pose = Pose(
                position_xyz=(0.7, -0.00586 + 0.5, 0.22773),
                rotation_wxyz=(0.7071068, 0, 0, -0.7071068),
            )
            object.set_initial_pose(object_pose)
            assets.append(object)

        # Compose the scene
        scene = Scene(assets=assets)

        isaac_arena_environment = IsaacArenaEnvironment(
            name="open_door",
            embodiment=GR1T2Embodiment(),
            scene=scene,
            task=OpenDoorTask(microwave),
            teleop_device=teleop_device,
        )

        return isaac_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        # parser.add_argument("--object", type=str, default=None)
        add_argument_if_missing(parser, "--object", type=str, default=None)


ExampleEnvironments = {
    Gr1OpenMicrowaveEnvironment.name: Gr1OpenMicrowaveEnvironment(),
}


def add_example_environments_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--example_environment", type=str, default=None)
    for example_environment in ExampleEnvironments.values():
        example_environment.add_cli_args(parser)
    return parser


def get_env_cfg_from_cli(args_cli: argparse.Namespace) -> tuple[ManagerBasedRLEnvCfg, str]:
    # Get the example environment
    assert hasattr(args_cli, "example_environment"), "Example environment must be specified"
    assert (
        args_cli.example_environment in ExampleEnvironments
    ), f"Example environment type {args_cli.example_environment} not supported"
    example_env = ExampleEnvironments[args_cli.example_environment]

    # Compile the environment
    env_builder = ArenaEnvBuilder(example_env.get_env(args_cli), args_cli)
    name, cfg = env_builder.build_registered()
    return cfg, name
