# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the local ``gr1_open_microwave`` environment through the generic Gym API.
使用通用的 Gym API 运行本地 ``gr1_open_microwave`` 环境。

This is a minimal example that:
1. registers the local Arena env with Gym,
2. calls ``gym.make("gr1_open_microwave")`` in the usual Gym way,
3. steps it for 100 simulation steps,
4. renders frames during execution.

Example:
    python examples/example04_run_exist_env0.py --viz kit --steps 1000
    python examples/example04_run_exist_env0.py --viz kit --steps 1000 --enable_cameras --camera_names camera_rgb --camera_resolution 640 480
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser


def parse_args() -> argparse.Namespace:
    parser = get_isaaclab_arena_environments_cli_parser()
    parser.add_argument("--steps", type=int, default=3)
    parser.set_defaults(num_envs=1)
    return parser.parse_args()


def main() -> None:
    args_cli = parse_args()

    # 1) Launch Isaac Sim app.
    app = AppLauncher(args_cli).app

    # 2) Build the already-registered local env.
    env_name = "gr1_open_microwave"
    env_args = get_isaaclab_arena_environments_cli_parser().parse_args([
        env_name,
        "--object",
        "cracker_box",
        "--embodiment",
        "gr1_pink",
    ])
    name, cfg, env_kwargs = get_arena_builder_from_cli(env_args).build_registered()

    # 3) Create the gym env using the repo's canonical API.
    env = gym.make(name, cfg=cfg, render_mode="rgb_array", disable_env_checker=True, **env_kwargs)

    # 4) Reset and inspect.
    obs, info = env.reset()
    print(f"Reset: obs_shape={obs.shape if hasattr(obs, 'shape') else type(obs)}, info_keys={list(info)[:10]}")
    print(f"Action space: {env.action_space}")

    # 5) Step a few times.
    for step_idx in range(args_cli.steps):
        action = torch.zeros(env.action_space.shape, device=env.unwrapped.device, dtype=torch.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"step {step_idx + 1}/{args_cli.steps}: reward={reward}, terminated={terminated}, truncated={truncated}")

        if terminated or truncated:
            print("Episode ended; resetting...")
            env.reset()

    print(f"Completed {args_cli.steps} steps.")

    # 6) Close cleanly.
    env.close()
    app.close()


if __name__ == "__main__":
    main()
