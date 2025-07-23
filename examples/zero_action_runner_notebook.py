# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

# %%
import argparse
import gymnasium as gym
import torch

# Global simulation app, initialized only once in notebook
simulation_app = None
first_run = True


def run_loop(env, num_steps):
    # Main control loop
    for x in range(num_steps):
        print(f"Running loop {x}")
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)


# Only launch sim app once
if simulation_app is None:
    from isaaclab.app import AppLauncher

    print("Launching simulation app once in notebook")
    simulation_app = AppLauncher()

# %%
print("Running in notebook mode")
args_parser = argparse.ArgumentParser(description="Isaac Arena CLI parser.")
args = args_parser.parse_args([])
args.task = "Isaac-Arena-Kitchen-v0"
args.embodiment = "franka"
args.arena_task = "pick_and_place"
args.scene = "kitchen"
args.device = "cuda:0"
args.num_envs = 1
args.disable_fabric = True
args.num_steps = 10
if first_run:
    # Post simulation app launch imports
    from isaac_arena.environments.compile_env import compile_arena_env_cfg

    from isaaclab_tasks.utils import parse_env_cfg

    # Compile an isaac arena environment configuration from existing isaac arena registry.
    arena_env_cfg = compile_arena_env_cfg(args)
    gym.register(
        id=args.task,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": arena_env_cfg,
        },
        disable_env_checker=True,
    )

    # Build the environment configuration in gym.
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
    env = gym.make(args.task, cfg=env_cfg)

    env.reset()
    first_run = False

run_loop(env, args.num_steps)
