"""SpaceMouse teleoperation demo for ABB IRB1200 with a controllable Robotiq 2F-140.

Run:
    /isaac-sim/python.sh examples/examples_teleop/example21_teleop_spacemouse_abb_irb1200_robotiq_2f140_control.py

This demo intentionally does not modify the shared IRB1200 embodiment module.
It instantiates the normal IRB1200 embodiment, then points its USD path at a
copied USDA that mounts the existing Robotiq 2F-140 physics asset under
link_6. SpaceMouse controls the 6D arm target and its left button toggles the
gripper command when the Robotiq finger joint is recognized by Isaac Lab.
"""

import argparse
import contextlib
import time
import traceback

from isaaclab.app import AppLauncher
from isaaclab.devices.teleop_device_factory import create_teleop_device

from isaaclab_arena.cli.isaaclab_arena_cli import (
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)


def parse_args() -> argparse.Namespace:
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--num_steps", type=int, default=50000, help="Maximum simulation steps.")
    parser.add_argument(
        "--keep_open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the Kit window open after the demo step limit.",
    )
    parser.add_argument("--pos_sensitivity", type=float, default=0.25, help="SpaceMouse translation sensitivity.")
    parser.add_argument("--rot_sensitivity", type=float, default=0.55, help="SpaceMouse rotation sensitivity.")
    parser.add_argument(
        "--debug_joints",
        action="store_true",
        help="Print live joint positions while teleoperating.",
    )
    parser.set_defaults(num_envs=1, visualizer=["kit"], enable_cameras=False)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _find_irb1200_robotiq_usd_path() -> str:
    from isaaclab_arena.embodiments.abb.abb_irb1200 import find_irb1200_robotiq_2f140_physx_usd_path

    return find_irb1200_robotiq_2f140_physx_usd_path(require_exists=True)


def main() -> None:
    import torch
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
    from isaaclab.managers import ActionTermCfg
    from isaaclab.utils.configclass import configclass
    from omni.usd import get_context

    from isaaclab_arena.assets.registries import AssetRegistry, DeviceRegistry
    from isaaclab_arena.embodiments.droid.actions import BinaryJointPositionZeroToOneAction
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.utils.pose import Pose

    builder_cfg = arena_env_builder_cfg_from_argparse(args_cli)
    asset_registry = AssetRegistry()
    device_registry = DeviceRegistry()

    robot = asset_registry.get_asset_by_name("abb_irb1200_ik")(enable_cameras=False)

    @configclass
    class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
        class_type = BinaryJointPositionZeroToOneAction

    @configclass
    class ABBIRB1200RobotiqActionCfg:
        arm_action: ActionTermCfg = robot.action_config.arm_action
        gripper_action: ActionTermCfg = BinaryJointPositionZeroToOneActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.0},
            close_command_expr={"finger_joint": torch.pi / 4},
        )

    robotiq_usd_path = _find_irb1200_robotiq_usd_path()
    robot.scene_config.robot.spawn = robot.scene_config.robot.spawn.replace(usd_path=robotiq_usd_path)
    robot.scene_config.robot.actuators["gripper"] = ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],
        stiffness=None,
        damping=None,
        velocity_limit=5.0,
    )
    robot.scene_config.robot.init_state = robot.scene_config.robot.init_state.replace(
        joint_pos={
            "joint_1": 0.0,
            "joint_2": -0.35,
            "joint_3": 0.65,
            "joint_4": 0.0,
            "joint_5": 0.75,
            "joint_6": 0.0,
            "finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
            "left_outer_finger_joint": 0.0,
            "right_outer_finger_joint": 0.0,
            "left_inner_finger_joint": 0.0,
            "right_inner_finger_joint": 0.0,
            "left_inner_finger_pad_joint": 0.0,
            "right_inner_finger_pad_joint": 0.0,
        }
    )
    robot.action_config = ABBIRB1200RobotiqActionCfg()
    cube = asset_registry.get_asset_by_name("procedural_cube")(instance_name="cube", prim_path="{ENV_REGEX_NS}/cube")
    light = asset_registry.get_asset_by_name("light")()

    robot.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    cube.set_initial_pose(Pose(position_xyz=(0.75, 0.0, 0.04)))

    teleop_device = device_registry.get_device_by_name("spacemouse")(
        sim_device=builder_cfg.device,
        pos_sensitivity=args_cli.pos_sensitivity,
        rot_sensitivity=args_cli.rot_sensitivity,
    )

    scene = Scene([light, cube])
    env_cfg = IsaacLabArenaEnvironment(
        name="teleop_abb_irb1200_spacemouse",
        embodiment=robot,
        scene=scene,
        task=NoTask(),
        teleop_device=teleop_device,
    )

    print("[INFO] Building ABB IRB1200 Arena environment...")
    print(f"[INFO] ABB IRB1200 controllable Robotiq 2F-140 USD: {robotiq_usd_path}")
    env_builder = ArenaEnvBuilder(env_cfg, builder_cfg)
    env = env_builder.make_registered()
    print("[INFO] Environment built. Resetting...")
    env.reset()
    simulation_app.update()
    stage = get_context().get_stage()
    gripper_prim = stage.GetPrimAtPath(
        "/World/envs/env_0/Robot/Geometry/base_link/link_1/link_2/link_3/link_4/link_5/link_6/robotiq_2f140_mount"
    )
    robotiq_base_prim = stage.GetPrimAtPath(
        "/World/envs/env_0/Robot/Geometry/base_link/link_1/link_2/link_3/link_4/link_5/link_6/"
        "robotiq_2f140_mount/robotiq_base_link"
    )
    robotiq_finger_joint_prim = stage.GetPrimAtPath(
        "/World/envs/env_0/Robot/Geometry/base_link/link_1/link_2/link_3/link_4/link_5/link_6/"
        "robotiq_2f140_mount/finger_joint"
    )
    print(f"[CHECK] robotiq_2f140_mount exists: {gripper_prim.IsValid()}")
    print(f"[CHECK] Robotiq base_link exists: {robotiq_base_prim.IsValid()}")
    print(f"[CHECK] Robotiq finger_joint prim exists: {robotiq_finger_joint_prim.IsValid()}")
    print("[INFO] Environment reset complete. Creating SpaceMouse interface...")
    robot_articulation = None
    joint_name_to_index = {}
    with contextlib.suppress(Exception):
        robot_articulation = env.unwrapped.scene["robot"]
        joint_name_to_index = {name: idx for idx, name in enumerate(robot_articulation.data.joint_names)}
        print(f"[INFO] ABB IRB1200 joints: {robot_articulation.data.joint_names}")
        print(f"[INFO] ABB IRB1200 bodies: {robot_articulation.data.body_names}")

    try:
        should_reset = False

        def request_reset() -> None:
            nonlocal should_reset
            should_reset = True

        teleop_interface = create_teleop_device(
            "spacemouse",
            env.unwrapped.cfg.teleop_devices.devices,
            callbacks={"R": request_reset},
        )
        teleop_interface.reset()
        print(teleop_interface)
        print(
            "ABB IRB1200 SpaceMouse teleop started: cap controls 6D delta pose, "
            "right button resets, left button is reserved for Robotiq open/close."
        )

        step = 0
        while simulation_app.is_running():
            if should_reset:
                should_reset = False
                try:
                    print("[INFO] Resetting environment...")
                    with torch.inference_mode():
                        env.reset()
                    teleop_interface.reset()
                    simulation_app.update()
                    print("[INFO] Reset complete.")
                except Exception:
                    print("[ERROR] Environment reset failed:")
                    traceback.print_exc()
                continue

            try:
                with torch.inference_mode():
                    raw_action = teleop_interface.advance()
                    action = raw_action[:7].repeat(env.unwrapped.num_envs, 1)
                    if step % 120 == 0:
                        print(f"[DEBUG] SpaceMouse 7D action: {raw_action[:7].detach().cpu().tolist()}")
                    if args_cli.debug_joints and robot_articulation is not None and step % 30 == 0:
                        joint_pos = robot_articulation.data.joint_pos[0].detach().cpu()
                        joint_debug = {
                            name: round(float(joint_pos[joint_name_to_index[name]]), 4)
                            for name in ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "finger_joint")
                            if name in joint_name_to_index
                        }
                        print(f"[DEBUG] Joint positions rad: {joint_debug}")
                    env.step(action)
                simulation_app.update()
                step += 1
            except Exception:
                print("[ERROR] Teleoperation loop failed:")
                traceback.print_exc()
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
        with contextlib.suppress(KeyboardInterrupt):
            main()
    except Exception:
        print("[ERROR] ABB IRB1200 teleop startup failed:")
        traceback.print_exc()
    finally:
        simulation_app.close()
