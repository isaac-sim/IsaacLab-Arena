# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Franka Kitchen 场景构建示例：演示如何组合场景并在 Kit 中可视化。
使用方法：
   python examples/example01_buildscene.py --viz kit --num_steps 500 --keep_open

"""

import argparse
import contextlib
import time

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser


def parse_args() -> argparse.Namespace:
    """解析命令行参数，默认启用 Kit 可视化窗口。"""
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument(
        "--num_steps",
        type=int,
        default=500,
        help="最大仿真步数。",
    )
    parser.add_argument(
        "--keep_open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="演示结束后保持 Kit 窗口打开。",
    )
    parser.set_defaults(num_envs=1, visualizer=["kit"])
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    """构建场景并在 Kit 可视化窗口中运行仿真循环。"""
    import torch

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    asset_registry = AssetRegistry()

    # 选择场景构建块
    background = asset_registry.get_asset_by_name("kitchen")()
    embodiment = asset_registry.get_asset_by_name("franka_ik")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()

    # 组合场景
    scene = Scene(assets=[background, cracker_box, tomato_soup_can])
    env_cfg = IsaacLabArenaEnvironment(
        name="franka_kitchen_example",
        embodiment=embodiment,
        scene=scene,
    )

    env_builder = ArenaEnvBuilder(env_cfg, arena_env_builder_cfg_from_argparse(args_cli))
    env = env_builder.make_registered()
    env.reset()
    simulation_app.update()

    try:
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for step in range(args_cli.num_steps):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                _, _, _, _, _ = env.step(actions)
            simulation_app.update()
            time.sleep(env.unwrapped.step_dt)

        if args_cli.keep_open:
            print("保持 Kit 窗口打开。关闭窗口或按 Ctrl+C 退出。")
            while simulation_app.is_running():
                simulation_app.update()
                time.sleep(1.0 / 60.0)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        with contextlib.suppress(KeyboardInterrupt):
            main()
    finally:
        simulation_app.close()
