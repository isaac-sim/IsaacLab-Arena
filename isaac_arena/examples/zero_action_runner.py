# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

import gymnasium as gym
import torch
import tqdm

from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser


def main():
    """Script to run an Isaac Arena environment with a zero-action agent."""

    # Launch Isaac Sim.
    args_parser = get_isaac_arena_cli_parser()
    args_parser.add_argument_group("Zero Action Runner", "Arguments for the zero action runner")
    args_parser.add_argument(
        "--num_steps", type=int, default=100, help="Number of steps to run the policy for. Default to run until "
    )

    # Args
    args_cli = args_parser.parse_args()

    # Start the simulation app
    with SimulationAppContext(args_cli):

        # Imports have to follow simulation startup.
        from isaac_arena.environments.compile_env import compile_arena_env_cfg

        from isaaclab_tasks.utils import parse_env_cfg

    # Build the environment configuration in gym.
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    env.reset()
    # while simulation_app.is_running():
    for _ in tqdm.tqdm(range(args_cli.num_steps)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

        env.reset()
        for _ in tqdm.tqdm(range(args_cli.num_steps)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

        assert False, "Fail!"

    # Close the environment.
    env.close()


if __name__ == "__main__":
    main()
