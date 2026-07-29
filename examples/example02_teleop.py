"""Franka Kitchen Pick-and-Place 示例：在 Kit 可视化窗口中运行抓取放置任务。"""

import argparse
import math
import time
import traceback

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
    parser.set_defaults(num_envs=1, visualizer=["kit"], enable_cameras=True)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    """构建场景并在 Kit 可视化窗口中运行仿真循环。"""
    import torch

    from isaaclab.devices.teleop_device_factory import create_teleop_device
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry, DeviceRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import PoseRange

    builder_cfg = arena_env_builder_cfg_from_argparse(args_cli)
    asset_registry = AssetRegistry()
    device_registry = DeviceRegistry()

    # 创建资产
    embodiment = asset_registry.get_asset_by_name("franka_ik")(enable_cameras=True)
    background = asset_registry.get_asset_by_name("kitchen")()
    tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()
    tomato_soup_can.set_initial_pose(
        PoseRange(
            position_xyz_min=(0.375, -0.125, 0.9255),
            position_xyz_max=(0.625, 0.125, 1.0255),
            rpy_min=(0.0, 0.0, -math.pi),
            rpy_max=(0.0, 0.0, math.pi),
        )
    )

    # 指定放置目标位置（厨房橱柜）
    destination_location = ObjectReference(
        name="destination_location",
        prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
        parent_asset=background,
        object_type=ObjectType.RIGID,
    )

    # 键盘遥操作设备
    teleop_device = device_registry.get_device_by_name("keyboard")(sim_device=builder_cfg.device)

    # 组合场景
    scene = Scene([background, tomato_soup_can])

    # 构建环境
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="franka_kitchen_pickup",
        embodiment=embodiment,
        scene=scene,
        task=PickAndPlaceTask(tomato_soup_can, destination_location, background),
        teleop_device=teleop_device,
    )

    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, builder_cfg)
    env = env_builder.make_registered()
    env.reset()
    simulation_app.update()

    try:
        should_reset = False

        def request_reset() -> None:
            nonlocal should_reset
            should_reset = True

        teleop_interface = create_teleop_device(
            "keyboard",
            env.unwrapped.cfg.teleop_devices.devices,
            callbacks={"R": request_reset},
        )
        teleop_interface.reset()
        print(teleop_interface)
        print("遥操作已启动：W/S、A/D、Q/E 平移，Z/X、T/G、C/V 旋转，K 开合夹爪，R 重置。")

        step = 0
        while simulation_app.is_running():
            if should_reset:
                should_reset = False
                try:
                    print("[INFO] 正在重置环境...")
                    # ProgressTracker 的状态可能在推理模式下首次创建，因此重置时
                    # 对这些张量的原地更新也必须处于相同的推理模式中。
                    with torch.inference_mode():
                        env.reset()
                    teleop_interface.reset()
                    # 让重置产生的 USD/Fabric/RTX 更新在下一次控制步之前完成。
                    simulation_app.update()
                    print("[INFO] 环境重置完成。")
                except Exception:
                    print("[ERROR] 环境重置失败：")
                    traceback.print_exc()
                continue

            try:
                with torch.inference_mode():
                    action = teleop_interface.advance().repeat(env.unwrapped.num_envs, 1)
                    env.step(action)
                simulation_app.update()
                step += 1
            except Exception:
                print("[ERROR] 遥操作循环执行失败：")
                traceback.print_exc()
                # 保持 Kit 窗口存活，避免异常看起来像无日志闪退。
                while simulation_app.is_running():
                    simulation_app.update()
                    time.sleep(0.01)
                break

            if not args_cli.keep_open and step >= args_cli.num_steps:
                break
            time.sleep(env.unwrapped.step_dt)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        simulation_app.close()
