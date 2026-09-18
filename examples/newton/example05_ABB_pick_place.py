# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena 中使用本地 USD 资产的 pick-and-place 场景。

Franka 机械臂从地平面上的起始位置抓取水壶，并将其放入目标容器。两个对象均从
本地 USD 文件加载；容器以厘米为单位制作，因此在生成时缩放到米单位。

使用方法：

    .venv/bin/python examples/newton/example05_ABB_pick_place.py
    .venv/bin/python examples/newton/example05_ABB_pick_place.py --viz kit
    .venv/bin/python examples/newton/example05_ABB_pick_place.py --viz none

注意：使用 `--viz kit` 可打开 Isaac Sim 可视化窗口。
"""

import contextlib
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
        presets="physx",
        visualizer=["newton"],
        device="cuda:0",
    )
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    """Build and run a Franka pick-and-place task with local USD assets."""
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.relations import IsAnchor
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    ground_plane = cast(Any, asset_registry.get_asset_by_name("ground_plane"))()
    # PickAndPlaceTask uses this height to identify an object that has fallen out of the workspace.
    ground_plane.object_min_z = -0.2

    jug = Object(
        name="fstyle_jug",
        usd_path="/home/magengyu/IsaacLab-Arena/scene/fstylejug_a01/fstylejug_a01_inst_physx.usd",
        object_type=ObjectType.RIGID,
        initial_pose=Pose(position_xyz=(0.45, -0.25, 0.01)),
    )

    container = Object(
        name="target_container",
        usd_path=(
            "/home/magengyu/IsaacLab-Arena/scene/Container_B04_40x30x12cm/"
            "Container_B04_40x30x12cm_PR_V_NVD_01.usd"
        ),
        object_type=ObjectType.RIGID,
        scale=(0.01, 0.01, 0.01),
        initial_pose=Pose(position_xyz=(0.45, 0.22, 0.0)),
    )
    container.add_relation(IsAnchor())
    destination = ObjectReference(
        name="container_drop_zone",
        parent_asset=container,
        prim_path="{ENV_REGEX_NS}/target_container",
        object_type=ObjectType.RIGID,
    )

    light = cast(Any, asset_registry.get_asset_by_name("light"))()

    scene = Scene(assets=[ground_plane, jug, container, light])
    arena_env = IsaacLabArenaEnvironment(
        name="franka_jug_pick_and_place",
        embodiment=FrankaIKEmbodiment(),
        scene=scene,
        task=PickAndPlaceTask(
            pick_up_object=jug,
            destination_location=destination,
            background_scene=ground_plane,
            task_description="抓取水壶并将其放入目标容器。",
        ),
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
