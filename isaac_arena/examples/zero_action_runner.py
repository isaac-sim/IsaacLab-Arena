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
from isaac_arena.scene.scene_registry import ObjectRegistry


def get_environment_configuration_from_args(args_cli: argparse.Namespace):
    from isaac_arena.embodiments.franka.franka_embodiment import FrankaEmbodiment
    from isaac_arena.embodiments.gr1t2.gr1t2_embodiment import GR1T2Embodiment

    object_registry = ObjectRegistry()
    if args_cli.background:
        background = object_registry.get_object_by_name(args_cli.background)
    else:
        background = object_registry.get_random_object_by_tag("background")
    if args_cli.pick_up_object:
        pick_up_object = object_registry.get_object_by_name(args_cli.pick_up_object)
    else:
        pick_up_object = object_registry.get_random_object_by_tag("pick_up_object")

    # Embodiment
    embodiments = {
        "gr1": GR1T2Embodiment,
        "franka": FrankaEmbodiment,
    }
    embodiment = embodiments[args_cli.embodiment]()

    scene_configuration = {
        "background": background,
        "pick_up_object": pick_up_object,
        "embodiment": embodiment,
    }
    return scene_configuration


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
        from isaac_arena.environments.compile_env import run_environment
        from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
        from isaac_arena.scene.pick_and_place_scene import PickAndPlaceScene
        from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTaskCfg

        # Scene variation
        scene_configuration = get_environment_configuration_from_args(args_cli)

        # Arena Environment
        isaac_arena_environment = IsaacArenaEnvironment(
            name="kitchen_pick_and_place",
            embodiment=scene_configuration["embodiment"],
            scene=PickAndPlaceScene(
                scene_configuration["background"],
                scene_configuration["pick_up_object"],
            ),
            task=PickAndPlaceTaskCfg(),
        )

        # Compile an IsaacLab compatible arena environment configuration
        env = run_environment(isaac_arena_environment, args_cli)

        # Run some zero actions.
        for _ in tqdm.tqdm(range(args_cli.num_steps)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

        # Close the environment.
        env.close()


if __name__ == "__main__":
    main()
