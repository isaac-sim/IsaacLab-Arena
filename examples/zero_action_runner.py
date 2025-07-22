# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
from isaac_arena.isaaclab_utils.simulation_app import SimulationAppContext


def main():
    """Script to run an Isaac Arena environment with a zero-action agent."""

    # Args
    args_parser = get_isaac_arena_cli_parser()
    args_cli = args_parser.parse_args()

    # Start the simulation app
    with SimulationAppContext(args_cli) as simulation_app:

        # Post simulation app launch imports
        import gymnasium as gym
        import torch

        from isaac_arena.environments.compile_env import compile_arena_env_cfg

        from isaaclab_tasks.utils import parse_env_cfg

        # Compile an isaac arena environment configuration from existing isaac arena registry.
        arena_env_cfg = compile_arena_env_cfg(args_cli)
        gym.register(
            id=args_cli.task,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={
                "env_cfg_entry_point": arena_env_cfg,
            },
            disable_env_checker=True,
        )

        # Build the environment configuration in gym.
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        env = gym.make(args_cli.task, cfg=env_cfg)

        env.reset()
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

        env.close()


if __name__ == "__main__":
    main()
