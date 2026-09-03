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
    # 可视化器模式运行100步
    python examples/example04_run_exist_env.py --viz kit --steps 100
    # 观测中加入相继图像
    python examples/example04_run_exist_env.py --enable_cameras --steps 10
"""

from __future__ import annotations

import argparse
import sys

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser


def _shape_of(value):
    if isinstance(value, dict):
        return {key: _shape_of(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shape_of(v) for v in value]
    if hasattr(value, "shape"):
        return tuple(value.shape)
    if hasattr(value, "size"):
        return tuple(value.size())
    return type(value).__name__


def print_obs_shapes(obs, prefix="obs"):
    if isinstance(obs, dict):
        for key, value in obs.items():
            new_prefix = f"{prefix}.{key}"
            if isinstance(value, dict):
                print_obs_shapes(value, new_prefix)
            else:
                shape = _shape_of(value)
                print(f"{new_prefix}: {shape}")
    else:
        print(f"{prefix}: {_shape_of(obs)}")


def parse_args() -> argparse.Namespace:
    parser = get_isaaclab_arena_environments_cli_parser()
    parser.add_argument("--steps", type=int, default=1, help="Number of environment steps to run")
    env_cli_args = [
        *sys.argv[1:],
        "gr1_open_microwave",
        "--object",
        "cracker_box",
        "--embodiment",
        "gr1_pink",
    ]
    return parser.parse_args(env_cli_args)


def main() -> None:
    args_cli = parse_args()
    app = AppLauncher(args_cli).app

    builder = get_arena_builder_from_cli(args_cli)
    name, cfg, env_kwargs = builder.build_registered()

    env = gym.make(name, cfg=cfg, render_mode="rgb_array", disable_env_checker=True, **env_kwargs)
    obs, info = env.reset()

    print("\n=== obs shapes ===")
    print_obs_shapes(obs)
    print("=== end obs shapes ===\n")

    for i in range(args_cli.steps):
        action = torch.zeros(env.action_space.shape, device=env.unwrapped.device, dtype=torch.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"step {i + 1}/{args_cli.steps}: reward={reward}, terminated={terminated}, truncated={truncated}")
        if terminated or truncated:
            env.reset()

    env.close()
    app.close()


if __name__ == "__main__":
    main()
