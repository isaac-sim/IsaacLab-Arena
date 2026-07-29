"""Franka Kitchen Pick-and-Place 示例：使用 Quest 右手柄遥操作机械臂。"""

import argparse
import math
import traceback

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser


def parse_args() -> argparse.Namespace:
    """解析命令行参数，要求通过 ``--xr`` 启用 OpenXR。"""
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
    parser.set_defaults(num_envs=1, visualizer=["kit"], enable_cameras=False)
    args = parser.parse_args()
    assert args.xr, "请添加 --xr 以启用 Quest OpenXR 遥操作"
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    """构建场景并在 Kit 可视化窗口中运行仿真循环。"""
    print("[XR] Isaac Sim 已启动，开始构建 Arena 环境。", flush=True)

    import torch

    from isaaclab_teleop import create_isaac_teleop_device
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
    # XR 遥操作使用头显视图，不需要额外初始化 Franka wrist camera。
    # 同时启用 TiledCamera 和 OpenXR 会建立第二条 RTX/SyntheticData 渲染链，
    # 在部分驱动组合下会卡在环境创建阶段。
    embodiment = asset_registry.get_asset_by_name("franka_ik")(enable_cameras=False)
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

    # Quest 右手柄位姿控制末端，Trigger 控制夹爪。
    teleop_device = device_registry.get_device_by_name("openxr")(sim_device=builder_cfg.device)

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

    print("[XR] 正在创建 Franka 环境。", flush=True)
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, builder_cfg)
    env = env_builder.make_registered()
    print("[XR] 环境创建完成，正在 reset。", flush=True)
    env.reset()
    print("[XR] 环境 reset 完成，正在创建 IsaacTeleop 设备。", flush=True)

    try:
        should_reset = False

        # 该独立示例没有实现远程 START/STOP 控制。关闭 control channel 后，
        # controller pipeline 会在 Session 建立后直接逐帧运行。
        env.unwrapped.cfg.isaac_teleop.control_channel_uuid = None

        def request_reset() -> None:
            nonlocal should_reset
            should_reset = True

        teleop_interface = create_isaac_teleop_device(
            env.unwrapped.cfg.isaac_teleop,
            sim_device=str(env.unwrapped.device),
            callbacks={"R": request_reset},
        )
        print("[XR] IsaacTeleop 设备创建完成，正在启动 Teleop Session。", flush=True)
        with teleop_interface:
            print("[XR] Teleop Session 已启动。", flush=True)
            teleop_interface.reset()
            print(teleop_interface)
            print(
                "XR 遥操作已启动：移动 Quest 右手柄控制末端，按下 Trigger 闭合夹爪，松开 Trigger 打开。",
                flush=True,
            )

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
                        print("[INFO] 环境重置完成。")
                    except Exception:
                        print("[ERROR] 环境重置失败：")
                        traceback.print_exc()
                    continue

                try:
                    with torch.inference_mode():
                        action = teleop_interface.advance()
                        if action is None:
                            # 保持 Kit 响应，同时等待 WebXR Session 提供控制器数据。
                            env.unwrapped.sim.render()
                        else:
                            if step == 0:
                                print("[INFO] 已收到 WebXR 控制器数据，开始执行遥操作。", flush=True)
                            env.step(action.repeat(env.unwrapped.num_envs, 1))
                            step += 1
                except Exception:
                    print("[ERROR] 遥操作循环执行失败：")
                    traceback.print_exc()
                    break

                if not args_cli.keep_open and step >= args_cli.num_steps:
                    break
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        simulation_app.close()
