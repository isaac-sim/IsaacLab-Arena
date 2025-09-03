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
from typing import TYPE_CHECKING

from isaac_arena.examples.example_environments.gr1_open_microwave_environment import Gr1OpenMicrowaveEnvironment
from isaac_arena.examples.example_environments.pick_and_place_environment import PickAndPlaceEnvironment

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg

# Collection of the available example environments
ExampleEnvironments = {
    Gr1OpenMicrowaveEnvironment.name: Gr1OpenMicrowaveEnvironment,
    PickAndPlaceEnvironment.name: PickAndPlaceEnvironment,
}


def add_example_environments_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--example_environment", type=str, default=None)
    for example_environment in ExampleEnvironments.values():
        example_environment.add_cli_args(parser)
    return parser


def get_env_cfg_from_cli(args_cli: argparse.Namespace):  # -> tuple[ManagerBasedRLEnvCfg, str]:
    from isaac_arena.environments.compile_env import ArenaEnvBuilder

    # Get the example environment
    assert hasattr(args_cli, "example_environment"), "Example environment must be specified"
    assert (
        args_cli.example_environment in ExampleEnvironments
    ), f"Example environment type {args_cli.example_environment} not supported"
    example_env = ExampleEnvironments[args_cli.example_environment]()

    # Compile the environment
    env_builder = ArenaEnvBuilder(example_env.get_env(args_cli), args_cli)
    name, cfg = env_builder.build_registered()
    return cfg, name
