# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Visualize a cube falling onto the microwave tray in Omniverse Kit.

示例目的（中文说明）:
- 演示如何在 Isaac Lab Arena 中构建一个简单场景: 厨房背景 + 微波炉 + 立方体，
    将立方体瞬移到微波炉转盘上方，打开微波炉门，使立方体落到转盘上，
    并通过任务终止条件检查是否成功（触发 success termination）。

主要步骤:
1. 使用 `AssetRegistry` 加载资产（kitchen / microwave / dex_cube）。
2. 设置微波炉的初始 pose，并构建 `ObjectReference` 指向转盘的刚体 prim。
3. 使用 `ArenaEnvBuilder` 和 `IsaacLabArenaEnvironment` 创建并注册环境。
4. 将立方体瞬移到转盘上方，打开微波炉门，执行若干 simulation steps。
5. 检查任务是否触发 success termination 并正确终止；可选择保持 Kit 窗口打开以观察。

# 在仓库根目录运行示例
python examples/example01_object_on_microwave_tray.py
"""

import argparse
import contextlib
import time

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and enable the Kit visualizer by default."""
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument(
        "--num_steps",
        type=int,
        default=120,
        help="Maximum number of simulation steps to wait for the cube to reach the tray.",
    )
    parser.add_argument(
        "--keep_open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the Kit window open after the demonstration finishes.",
    )
    parser.set_defaults(num_envs=1, visualizer=["kit"])
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    """Run the microwave-tray contact demonstration."""
    import torch

    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen")()
    microwave = asset_registry.get_asset_by_name("microwave")()
    dex_cube = asset_registry.get_asset_by_name("dex_cube")()

    # 将微波炉放到场景中的指定位置与朝向，数据来自场景标定/经验值
    microwave.set_initial_pose(
        Pose(position_xyz=(0.4, -0.00586, 0.22773), rotation_xyzw=(0.0, 0.0, -0.7071068, 0.7071068))
    )

    # 目标引用：指向微波炉内部的转盘刚体（用于任务判断目标是否放置到该刚体上）
    destination_ref = ObjectReference(
        name="microwave_disc",
        parent_asset=microwave,
        prim_path="{ENV_REGEX_NS}/microwave/Microwave039_Disc001",
        object_type=ObjectType.RIGID,
    )

    # 构建场景并创建环境，任务为将 dex_cube 放到 destination_ref（转盘）上
    scene = Scene(assets=[background, microwave, dex_cube])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="microwave_tray",
        embodiment=FrankaIKEmbodiment(),
        scene=scene,
        task=PickAndPlaceTask(dex_cube, destination_ref, background),
    )

    # 使用命令行参数配置构建器并注册环境（Gym 风格包装器）
    env = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
    env.reset()

    try:
        # 将立方体瞬移到转盘上方并清除速度，使其自由下落（模拟放开/掉落）
        # 目标位置为微波炉的世界坐标（x/y 与微波炉一致），z 比转盘高 0.06m
        cube_asset = env.unwrapped.scene[dex_cube.name]
        target_pos = torch.tensor([0.4, -0.00586, 0.28773], device=env.unwrapped.device)
        root_pose = torch.zeros((1, 7), device=env.unwrapped.device)
        root_pose[0, :3] = target_pos
        root_pose[0, 3] = 1.0  # identity quaternion (w, x, y, z)
        # 写入模拟器索引，直接设置根位姿与速度（瞬移）
        cube_asset.write_root_pose_to_sim_index(root_pose=root_pose)
        cube_asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=env.unwrapped.device))

        # 打开微波炉门，使立方体能够落入转盘而不是被门挡住
        microwave.open(env, env_ids=None)

        # 执行若干仿真步（发送零动作），检查任务是否触发 success 终止条件
        succeeded = False
        terminated = False
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for _ in range(args_cli.num_steps):
            if not simulation_app.is_running():
                break
            # 使用 inference_mode 减少开销（不计算梯度）
            with torch.inference_mode():
                _, _, terminated_tensor, _, _ = env.step(actions)
                # 读取成功终止标志与整体终止标志
                succeeded = succeeded or env.unwrapped.termination_manager.get_term("success").item()
                terminated = terminated or terminated_tensor.item()
            # 等待一个仿真步长（与 env.unwrapped.step_dt 保持一致）以便渲染和物理推进
            time.sleep(env.unwrapped.step_dt)

        # 验证任务成功与环境终止
        assert succeeded, "Cube on the tray never fired the success termination"
        assert terminated, "The task was not terminated"
        print("Cube landed on the microwave tray and fired the success termination.")

        # 根据命令行参数决定是否保持 Kit 窗口以便观察
        if args_cli.keep_open:
            print("Keeping the Kit window open. Close it or press Ctrl+C to exit.")
            while simulation_app.is_running():
                simulation_app.update()
                time.sleep(1.0 / 60.0)
    finally:
        # 确保环境资源被正确释放
        env.close()


if __name__ == "__main__":
    try:
        with contextlib.suppress(KeyboardInterrupt):
            main()
    finally:
        simulation_app.close()
