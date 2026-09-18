# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""在 Arena 中使用 Newton 物理仿真的示例。

这个脚本展示如何用 Arena 的环境构建器组合一个包含桌面和物体的场景，并使用
Newton 物理后端。

使用方法：

    .venv/bin/python examples/newton/example05_ABB.py
    .venv/bin/python examples/newton/example05_ABB.py --viz kit
    .venv/bin/python examples/newton/example05_ABB.py --viz none

注意：使用 `--viz kit` 可打开 Isaac Sim 可视化窗口。
"""

import contextlib
import math
import time
from typing import Any, cast

import torch
from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser


def parse_args():
    """Parse the demo CLI arguments."""
    parser = get_isaaclab_arena_cli_parser()
    parser.set_defaults(
        num_envs=1,
        presets="newton",
        visualizer=["newton"],
        device="cuda:0",
    )
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    """Create a Arena scene and step it with zero actions."""
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose, PoseRange

    asset_registry = AssetRegistry()
    background = cast(Any, asset_registry.get_asset_by_name("procedural_table"))()
    background.set_initial_pose(Pose(position_xyz=(-0.55, 0.0, 0.235)))
    manip_object = cast(Any, asset_registry.get_asset_by_name("cracker_box"))()
    manip_object.set_initial_pose(
        PoseRange(
            position_xyz_min=(-0.75, -0.1, 1.35),
            position_xyz_max=(-0.35, 0.3, 1.75),
            rpy_min=(-math.pi, -math.pi, -math.pi),
            rpy_max=(math.pi, math.pi, math.pi),
        )
    )
    ground_plane = cast(Any, asset_registry.get_asset_by_name("ground_plane"))()
    light = cast(Any, asset_registry.get_asset_by_name("light"))()

    scene = Scene(assets=[background, manip_object, ground_plane, light])
    arena_env = IsaacLabArenaEnvironment(
        name="arena_demo",
        embodiment=NoEmbodiment(),
        scene=scene,
    )

    env_builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli))
    env = env_builder.make_registered()
    env.reset()
    simulation_app.update()

    runtime_env = cast(Any, env)
    action = torch.zeros(runtime_env.action_space.shape, dtype=torch.float32, device=runtime_env.unwrapped.device)
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                runtime_env.step(action)
            simulation_app.update()
            time.sleep(runtime_env.unwrapped.step_dt)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        with contextlib.suppress(KeyboardInterrupt):
            main()
    finally:
        simulation_app.close()
