"""Franka Newton pick/place demo in the Isaac Lab Arena launcher style.

This script intentionally follows the structure of example23 / example28 /
example29: it uses Arena's CLI, starts Isaac Sim through AppLauncher, defaults
to the Newton preset, and renders through Kit.

Run inside the IsaacLab-Arena Docker container:

    cd /workspaces/isaaclab_arena
    /isaac-sim/python.sh examples/examples_teleop/example30_newton_official_franka_pick_place.py

Keyboard mode:

    /isaac-sim/python.sh examples/examples_teleop/example30_newton_official_franka_pick_place.py --mode keyboard

Controls follow Isaac Lab's Se3Keyboard convention:
    W/S, A/D, Q/E   translate end-effector along x/y/z
    Z/X, T/G, C/V   rotate end-effector around roll/pitch/yaw
    K               toggle gripper open/close
    R               reset environment
"""

from __future__ import annotations

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
    parser.add_argument(
        "--mode",
        choices=("keyboard", "scripted"),
        default="scripted",
        help="Use keyboard teleoperation or a simple scripted pick/place action sequence.",
    )
    parser.add_argument("--num_steps", type=int, default=50000, help="Maximum simulation steps.")
    parser.add_argument("--keep_open", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pos_sensitivity", type=float, default=0.12)
    parser.add_argument("--rot_sensitivity", type=float, default=0.04)
    parser.add_argument(
        "--cube_size",
        type=float,
        default=0.05,
        help="Cube edge length. Default matches Newton's ik_cube_stacking example.",
    )
    parser.add_argument(
        "--cube_density",
        type=float,
        default=400.0,
        help="Cube density in kg/m^3. Newton's ik_cube_stacking samples 300-500; default uses the midpoint.",
    )
    parser.add_argument(
        "--cube_mass",
        type=float,
        default=None,
        help="Override cube mass in kg. If omitted, mass is computed from --cube_density and --cube_size.",
    )
    parser.add_argument(
        "--cube_friction",
        type=float,
        default=0.75,
        help="Cube friction coefficient. Default matches Newton's ik_cube_stacking mu.",
    )
    parser.add_argument("--cube_contact_offset", type=float, default=0.0005)
    parser.add_argument("--cube_rest_offset", type=float, default=0.0)
    parser.add_argument("--world_gravity", type=float, default=-9.81)
    parser.add_argument(
        "--use_newton_actuators",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Newton-native actuators. This is more stable for Franka teleop on the Newton backend.",
    )
    parser.add_argument("--arm_stiffness", type=float, default=1800.0)
    parser.add_argument("--arm_damping", type=float, default=220.0)
    parser.add_argument("--arm_effort_limit", type=float, default=5000.0)
    parser.add_argument("--finger_stiffness", type=float, default=350.0)
    parser.add_argument("--finger_damping", type=float, default=80.0)
    parser.add_argument("--finger_effort_limit", type=float, default=80.0)
    parser.add_argument(
        "--idle_joint_hold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Directly hold Franka arm joints when keyboard input is idle. This reduces Newton teleop sag.",
    )
    parser.add_argument(
        "--idle_action_threshold",
        type=float,
        default=1e-5,
        help="Absolute 6D keyboard action threshold treated as idle for --idle_joint_hold.",
    )
    parser.add_argument("--newton_num_substeps", type=int, default=8)
    parser.add_argument("--newton_solver_iterations", type=int, default=220)
    parser.add_argument("--newton_solver_ls_iterations", type=int, default=35)
    parser.add_argument("--newton_solver_impratio", type=float, default=30.0)
    parser.add_argument("--newton_solver_nconmax", type=int, default=800)
    parser.add_argument("--newton_solver_ccd_iterations", type=int, default=20000)
    parser.add_argument("--newton_solver_cone", choices=("elliptic", "pyramidal"), default="elliptic")
    parser.add_argument("--script_scale", type=float, default=1.0, help="Multiplier for scripted end-effector deltas.")
    parser.add_argument("--debug", action="store_true", help="Print action and cube pose diagnostics.")
    parser.set_defaults(num_envs=1, visualizer=["kit"], enable_cameras=False, presets="newton")
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _manual_env_step(env) -> None:
    for _ in range(env.unwrapped.cfg.decimation):
        env.unwrapped.scene.write_data_to_sim()
        env.unwrapped.sim.step(render=False)
        env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)
    if env.unwrapped.sim.is_rendering:
        env.unwrapped.sim.render(skip_app_pumping=False)


def _write_arm_hold(robot_articulation, arm_joint_ids, hold_joint_pos) -> None:
    import torch

    robot_articulation.write_joint_position_to_sim_index(
        position=hold_joint_pos,
        joint_ids=arm_joint_ids,
    )
    robot_articulation.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(hold_joint_pos),
        joint_ids=arm_joint_ids,
    )
    robot_articulation.set_joint_position_target_index(
        target=hold_joint_pos,
        joint_ids=arm_joint_ids,
    )


def _scripted_action(step: int, device, num_envs: int, scale: float):
    import torch

    action = torch.zeros((num_envs, 7), device=device, dtype=torch.float32)

    # The action is relative SE(3) + binary gripper. These phases are deliberately
    # slow because Newton contact grasping is sensitive to aggressive closure.
    if step < 120:
        action[:, 6] = 1.0
    elif step < 220:
        action[:, 0] = 0.20 * scale
        action[:, 6] = 1.0
    elif step < 320:
        action[:, 2] = -0.15 * scale
        action[:, 6] = 1.0
    elif step < 520:
        action[:, 6] = -1.0
    elif step < 700:
        action[:, 2] = 0.16 * scale
        action[:, 6] = -1.0
    elif step < 900:
        action[:, 1] = -0.18 * scale
        action[:, 6] = -1.0
    elif step < 1040:
        action[:, 2] = -0.12 * scale
        action[:, 6] = -1.0
    elif step < 1180:
        action[:, 6] = 1.0
    elif step < 1350:
        action[:, 2] = 0.12 * scale
        action[:, 6] = 1.0
    else:
        action[:, 6] = 1.0

    return action


def _configure_newton(manager_env_cfg) -> None:
    manager_env_cfg.sim.gravity = (0.0, 0.0, args_cli.world_gravity)
    manager_env_cfg.sim.use_newton_actuators = args_cli.use_newton_actuators
    if manager_env_cfg.sim.physics is None:
        return

    manager_env_cfg.sim.physics.num_substeps = args_cli.newton_num_substeps
    solver_cfg = getattr(manager_env_cfg.sim.physics, "solver_cfg", None)
    if solver_cfg is None:
        return

    solver_cfg.iterations = args_cli.newton_solver_iterations
    solver_cfg.ls_iterations = args_cli.newton_solver_ls_iterations
    solver_cfg.impratio = args_cli.newton_solver_impratio
    solver_cfg.nconmax = args_cli.newton_solver_nconmax
    solver_cfg.ccd_iterations = args_cli.newton_solver_ccd_iterations
    solver_cfg.cone = args_cli.newton_solver_cone
    solver_cfg.use_mujoco_contacts = False


def main() -> None:
    import torch
    from isaaclab.actuators import ImplicitActuatorCfg
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.registries import AssetRegistry, DeviceRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.utils.pose import Pose

    builder_cfg = arena_env_builder_cfg_from_argparse(args_cli)
    asset_registry = AssetRegistry()
    device_registry = DeviceRegistry()

    table = asset_registry.get_asset_by_name("table")()
    robot = asset_registry.get_asset_by_name("franka_ik")(enable_cameras=False)
    robot.scene_config.robot.spawn = robot.scene_config.robot.spawn.replace(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        )
    )
    robot.scene_config.robot.actuators["panda_shoulder"] = ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"],
        effort_limit_sim=args_cli.arm_effort_limit,
        velocity_limit_sim=2.175,
        stiffness=args_cli.arm_stiffness,
        damping=args_cli.arm_damping,
        armature=2e-2,
    )
    robot.scene_config.robot.actuators["panda_forearm"] = ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"],
        effort_limit_sim=args_cli.arm_effort_limit,
        velocity_limit_sim=2.61,
        stiffness=args_cli.arm_stiffness,
        damping=args_cli.arm_damping,
        armature=2e-2,
    )
    robot.scene_config.robot.actuators["panda_hand"] = ImplicitActuatorCfg(
        joint_names_expr=["panda_finger_joint.*"],
        effort_limit_sim=args_cli.finger_effort_limit,
        velocity_limit_sim=0.20,
        stiffness=args_cli.finger_stiffness,
        damping=args_cli.finger_damping,
    )
    light = asset_registry.get_asset_by_name("light")()
    cube_mass = args_cli.cube_mass
    if cube_mass is None:
        cube_mass = args_cli.cube_density * args_cli.cube_size**3
    cube = Object(
        name="newton_franka_cube",
        object_type=ObjectType.RIGID,
        spawner_cfg=sim_utils.CuboidCfg(
            size=(args_cli.cube_size, args_cli.cube_size, args_cli.cube_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=args_cli.cube_contact_offset,
                rest_offset=args_cli.cube_rest_offset,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=cube_mass),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 0.9)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=args_cli.cube_friction,
                dynamic_friction=args_cli.cube_friction,
                restitution=0.0,
            ),
        ),
    )

    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    robot.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0)))
    cube.set_initial_pose(Pose(position_xyz=(0.30, 0.0, args_cli.cube_size)))

    teleop_device = device_registry.get_device_by_name("keyboard")(
        sim_device=builder_cfg.device,
        pos_sensitivity=args_cli.pos_sensitivity,
        rot_sensitivity=args_cli.rot_sensitivity,
    )

    scene = Scene([table, light, cube])
    env_cfg = IsaacLabArenaEnvironment(
        name="newton_franka_pick_place",
        embodiment=robot,
        scene=scene,
        task=NoTask(),
        teleop_device=teleop_device,
    )

    print("[INFO] Building Franka Newton pick/place environment...")
    env_builder = ArenaEnvBuilder(env_cfg, builder_cfg)
    manager_env_cfg, env_kwargs = env_builder.compose_manager_cfg()
    _configure_newton(manager_env_cfg)
    env = env_builder.make_registered(manager_env_cfg, env_kwargs)
    env.reset()
    simulation_app.update()

    robot_articulation = env.unwrapped.scene["robot"]
    joint_name_to_index = {name: idx for idx, name in enumerate(robot_articulation.data.joint_names)}
    arm_joint_ids = [joint_name_to_index[f"panda_joint{i}"] for i in range(1, 8)]
    arm_hold_joint_pos = robot_articulation.data.joint_pos.torch[:, arm_joint_ids].clone()
    if args_cli.idle_joint_hold:
        _write_arm_hold(robot_articulation, arm_joint_ids, arm_hold_joint_pos)
        env.unwrapped.scene.write_data_to_sim()

    try:
        should_reset = False

        def request_reset() -> None:
            nonlocal should_reset
            should_reset = True

        teleop_interface = None
        if args_cli.mode == "keyboard":
            import omni.kit.app

            extension_manager = omni.kit.app.get_app().get_extension_manager()
            extension_manager.set_extension_enabled_immediate("omni.appwindow", True)
            import carb.input  # noqa: F401
            import omni.appwindow  # noqa: F401

            teleop_interface = create_teleop_device(
                "keyboard",
                env.unwrapped.cfg.teleop_devices.devices,
                callbacks={"R": request_reset},
            )
            teleop_interface.reset()
            print(teleop_interface)
            print("Franka Newton teleop started: W/S A/D Q/E move EE, Z/X T/G C/V rotate, K gripper, R reset.")
        else:
            print("[INFO] Scripted pick/place mode started. Use --mode keyboard for manual end-effector control.")

        step = 0
        while simulation_app.is_running():
            if should_reset:
                should_reset = False
                try:
                    print("[INFO] Resetting environment...")
                    with torch.inference_mode():
                        env.reset()
                    if teleop_interface is not None:
                        teleop_interface.reset()
                    arm_hold_joint_pos = robot_articulation.data.joint_pos.torch[:, arm_joint_ids].clone()
                    if args_cli.idle_joint_hold:
                        _write_arm_hold(robot_articulation, arm_joint_ids, arm_hold_joint_pos)
                        env.unwrapped.scene.write_data_to_sim()
                    step = 0
                    simulation_app.update()
                    print("[INFO] Reset complete.")
                except Exception:
                    print("[ERROR] Environment reset failed:")
                    traceback.print_exc()
                continue

            try:
                with torch.inference_mode():
                    if args_cli.mode == "keyboard":
                        action = teleop_interface.advance().repeat(env.unwrapped.num_envs, 1)
                        env.step(action)
                        if args_cli.idle_joint_hold:
                            is_idle = bool(torch.all(torch.abs(action[:, :6]) < args_cli.idle_action_threshold).item())
                            if is_idle:
                                _write_arm_hold(robot_articulation, arm_joint_ids, arm_hold_joint_pos)
                                env.unwrapped.scene.write_data_to_sim()
                            else:
                                arm_hold_joint_pos = robot_articulation.data.joint_pos.torch[:, arm_joint_ids].clone()
                    else:
                        action = _scripted_action(
                            step,
                            builder_cfg.device,
                            env.unwrapped.num_envs,
                            args_cli.script_scale,
                        )
                        env.step(action)

                    if args_cli.debug and step % 60 == 0:
                        cube_pose = env.unwrapped.scene["newton_franka_cube"].data.root_pose_w.torch[0, :3]
                        print(
                            f"[DEBUG] step={step} action={action[0].detach().cpu().tolist()} "
                            f"cube_xyz={cube_pose.detach().cpu().tolist()}"
                        )

                simulation_app.update()
                step += 1
            except Exception:
                print("[ERROR] Franka Newton pick/place loop failed:")
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
        print("[ERROR] Franka Newton pick/place startup failed:")
        traceback.print_exc()
    finally:
        simulation_app.close()
