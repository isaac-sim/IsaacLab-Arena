# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

import argparse
import torch
import tqdm

from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
from isaac_arena.isaaclab_utils.simulation_app import SimulationAppContext
from isaac_arena.scene.asset_registry import AssetRegistry


def get_environment_configuration_from_args(args_cli: argparse.Namespace):
    from isaac_arena.embodiments.franka import FrankaEmbodiment
    from isaac_arena.embodiments.gr1t2 import GR1T2Embodiment

    asset_registry = AssetRegistry()
    if args_cli.background:
        background = asset_registry.get_asset_by_name(args_cli.background)
    else:
        background = asset_registry.get_random_asset_by_tag("background")
    if args_cli.object:
        pick_up_object = asset_registry.get_asset_by_name(args_cli.object)
    else:
        pick_up_object = asset_registry.get_random_asset_by_tag("object")

    # Embodiment
    embodiments = {
        "gr1": GR1T2Embodiment,
        "franka": FrankaEmbodiment,
    }
    embodiment = embodiments[args_cli.embodiment]()

    environment_configuration = {
        "background": background,
        "object": pick_up_object,
        "embodiment": embodiment,
    }
    return environment_configuration


def main():
    """Script to run an Isaac Arena environment with a zero-action agent."""

    # Launch Isaac Sim.
    args_parser = get_isaac_arena_cli_parser()
    args_parser.add_argument_group("Zero Action Runner", "Arguments for the zero action runner")
    args_parser.add_argument(
        "--num_steps", type=int, default=100, help="Number of steps to run the policy for. Default to run until "
    )
    args_parser.add_argument(
        "--embodiment",
        type=str,
        default="franka",
        choices=["gr1", "franka"],
        help="Embodiment to use. Default to franka.",
    )

    # Args
    args_cli = args_parser.parse_args()

    # Start the simulation app
    with SimulationAppContext(args_cli):

        # Imports have to follow simulation startup.
        from isaac_arena.environments.compile_env import compile_environment
        from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
        from isaac_arena.scene.pick_and_place_scene import PickAndPlaceScene
        from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTask

        # Scene variation
        environment_configuration = get_environment_configuration_from_args(args_cli)

        # Arena Environment
        isaac_arena_environment = IsaacArenaEnvironment(
            name="kitchen_pick_and_place",
            embodiment=environment_configuration["embodiment"],
            scene=PickAndPlaceScene(
                environment_configuration["background"],
                environment_configuration["object"],
            ),
            task=PickAndPlaceTask(),
        )

        # Compile an IsaacLab compatible arena environment configuration
        env = compile_environment(isaac_arena_environment, args_cli)

        # Run some zero actions.
        for _ in tqdm.tqdm(range(args_cli.num_steps)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

        # Close the environment.
        env.close()


if __name__ == "__main__":
    main()
