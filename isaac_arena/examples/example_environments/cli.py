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

from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
from isaac_arena.examples.example_environments.galileo_pick_and_place_environment import GalileoPickAndPlaceEnvironment
from isaac_arena.examples.example_environments.gr1_open_microwave_environment import Gr1OpenMicrowaveEnvironment
from isaac_arena.examples.example_environments.kitchen_pick_and_place_environment import KitchenPickAndPlaceEnvironment
from isaac_arena.examples.example_environments.lightwheel_kitchen_pot_pick_and_place import (
    LightwheelKitchenPotPickAndPlaceEnvironment,
)

# NOTE(alexmillane, 2025.09.04): There is an issue with type annotation in this file.
# We cannot annotate types which require the simulation app to be started in order to
# import, because this file is used to retrieve CLI arguments, so it must be imported
# before the simulation app is started.
# TODO(alexmillane, 2025.09.04): Fix this.


# Collection of the available example environments
ExampleEnvironments = {
    Gr1OpenMicrowaveEnvironment.name: Gr1OpenMicrowaveEnvironment,
    KitchenPickAndPlaceEnvironment.name: KitchenPickAndPlaceEnvironment,
    GalileoPickAndPlaceEnvironment.name: GalileoPickAndPlaceEnvironment,
    LightwheelKitchenPotPickAndPlaceEnvironment.name: LightwheelKitchenPotPickAndPlaceEnvironment,
}


def add_example_environments_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    subparsers = parser.add_subparsers(dest="example_environment", required=True, help="Example environment to run")
    for example_environment in ExampleEnvironments.values():
        subparser = subparsers.add_parser(example_environment.name)
        example_environment.add_cli_args(subparser)

    return parser


def get_isaac_arena_example_environment_cli_parser() -> argparse.ArgumentParser:
    parser = get_isaac_arena_cli_parser()
    # NOTE(alexmillane, 2025.09.04): This command adds subparsers for each example environment.
    # So it has to be added last, because the subparser flags are parsed after the others.
    add_example_environments_cli_args(parser)
    return parser


def get_arena_builder_from_cli(args_cli: argparse.Namespace):  # -> tuple[ManagerBasedRLEnvCfg, str]:
    from isaac_arena.environments.compile_env import ArenaEnvBuilder

    # Get the example environment
    assert hasattr(args_cli, "example_environment"), "Example environment must be specified"
    assert (
        args_cli.example_environment in ExampleEnvironments
    ), f"Example environment type {args_cli.example_environment} not supported"
    example_env = ExampleEnvironments[args_cli.example_environment]()

    # Compile the environment
    env_builder = ArenaEnvBuilder(example_env.get_env(args_cli), args_cli)
    return env_builder
