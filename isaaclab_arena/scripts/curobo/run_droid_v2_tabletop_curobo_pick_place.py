#!/usr/bin/env python3
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

######## From Neel: This script is an example of how to use the curobo planner to pick and place objects.
######## It is not a complete pick and place policy, but it is a good starting point

from __future__ import annotations

import argparse
import json
import random as _random
from copy import deepcopy
from pathlib import Path

import torch

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser




GRIPPER_OPEN_CMD: float = 0.0
GRIPPER_CLOSE_CMD: float = 1.0
DOWN_FACING_QUAT_WXYZ = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)


def add_script_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--pick_order',
        nargs='+',
        type=str,
        default=None,
        help='Explicit object pick order. If not specified, auto-discovers all objects from scene.',
    )
    parser.add_argument(
        '--max_objects',
        type=int,
        default=None,
        help='Optional limit on number of objects to pick from discovered order.',
    )
    parser.add_argument(
        '--pre_grasp_height',
        type=float,
        default=0.20,
        help='Pre-grasp height above object center in meters.',
    )
    parser.add_argument(
        '--grasp_height_offset',
        type=float,
        default=0.02,
        help='Grasp pose z offset from object center in meters.',
    )
    parser.add_argument(
        '--transport_height',
        type=float,
        default=0.22,
        help='Transport z offset above bin center while carrying an object.',
    )
    parser.add_argument(
        '--place_height_offset',
        type=float,
        default=0.12,
        help='Place pose z offset from bin center in meters.',
    )
    parser.add_argument(
        '--bin_half_x',
        type=float,
        default=0.08,
        help='Half-width of bin interior along X in meters (bin ~20cm, usable ~16cm).',
    )
    parser.add_argument(
        '--bin_half_y',
        type=float,
        default=0.10,
        help='Half-width of bin interior along Y in meters (bin ~25cm, usable ~20cm).',
    )
    parser.add_argument(
        '--gripper_settle_steps',
        type=int,
        default=100,
        help='Number of env steps to hold pose while opening/closing gripper.',
    )
    parser.add_argument(
        '--approach_distance',
        type=float,
        default=0.04,
        help='CuRobo planner approach distance.',
    )
    parser.add_argument(
        '--retreat_distance',
        type=float,
        default=0.06,
        help='CuRobo planner retreat distance.',
    )
    parser.add_argument(
        '--time_dilation_factor',
        type=float,
        default=0.6,
        help='CuRobo time dilation factor.',
    )
    parser.add_argument(
        '--debug_planner',
        action='store_true',
        default=True,
        help='Enable verbose planner debugging.',
    )
    parser.add_argument(
        '--dump_spheres_dir',
        type=str,
        default=None,
        help='Directory to dump CuRobo link spheres before/after each planning stage.',
    )
    parser.add_argument(
        '--dump_spheres_png',
        action='store_true',
        default=False,
        help='If set, also save XY/XZ/YZ sphere projection PNGs for each dump.',
    )
    parser.add_argument(
        '--run_sanity_check',
        action='store_true',
        default=False,
        help='Run one pre-flight lift planning check to distinguish setup issues from goal issues.',
    )
    parser.add_argument(
        '--goal_z_boost',
        type=float,
        default=0.0,
        help='Extra Z offset added to ALL goal poses in robot-frame for IK reachability testing.',
    )
    parser.add_argument(
        '--rerun_recording_path',
        type=str,
        default=None,
        help='Optional .rrd output path for Rerun recording; useful when running inside Docker.',
    )


def _add_script_args_to_subparsers(parser: argparse.ArgumentParser) -> None:
    """Register script args on all environment subparsers too.

    Argparse only accepts top-level args before the subcommand.
    This helper allows these script args after the environment name as well.
    """
    for action in parser._actions:
        choices = getattr(action, 'choices', None)
        if isinstance(choices, dict):
            for subparser in choices.values():
                add_script_args(subparser)



def _get_current_eef_pose(env, planner, env_id: int = 0) -> torch.Tensor:
    joint_state = planner._get_current_joint_state_for_curobo()
    ee_pose = planner.get_ee_pose(joint_state)

    ee_pos = planner._to_env_device(ee_pose.position).view(-1, 3)[0]
    ee_quat = planner._to_env_device(ee_pose.quaternion).view(-1, 4)[0]
    ee_rot = math_utils.matrix_from_quat(ee_quat.unsqueeze(0))[0]
    print(f"[DEBUG EEF] CuRobo EEF position: {ee_pos}")
    return math_utils.make_pose(ee_pos, ee_rot)


def _action_from_pose(env, planner, target_pose: torch.Tensor, gripper_binary_action: float, env_id: int = 0) -> torch.Tensor:
    target_pos, target_rot = math_utils.unmake_pose(target_pose)
    curr_pose = _get_current_eef_pose(env, planner, env_id=env_id)
    curr_pos, curr_rot = math_utils.unmake_pose(curr_pose)

    delta_position = target_pos - curr_pos
    delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
    delta_quat = math_utils.quat_from_matrix(delta_rot_mat.unsqueeze(0))[0]
    delta_rotation = math_utils.axis_angle_from_quat(delta_quat.unsqueeze(0))[0]

    pose_action = torch.cat([delta_position, delta_rotation], dim=0)
    pose_action = torch.clamp(pose_action, -1.0, 1.0)
    gripper_action = torch.tensor([gripper_binary_action], device=env.device, dtype=torch.float32)
    action = torch.cat([pose_action, gripper_action], dim=0)
    return action.unsqueeze(0)


def _execute_plan(env, planner, gripper_binary_action: float, env_id: int = 0) -> None:
    planned_poses = planner.get_planned_poses()
    if not planned_poses:
        return
    for pose in planned_poses:
        action = _action_from_pose(env, planner, pose, gripper_binary_action, env_id=env_id)
        env.step(action)


def _execute_gripper_action(env, planner, gripper_binary_action: float, steps: int = 12, env_id: int = 0) -> None:
    state = "OPEN" if gripper_binary_action < 0.5 else "CLOSE"
    robot = env.scene['robot']
    finger_idx = robot.find_joints("finger_joint")[0]
    joint_pos_before = robot.data.joint_pos[env_id, finger_idx].item()
    print(f"[GRIPPER] Commanding {state} (action={gripper_binary_action}) | finger BEFORE: {joint_pos_before:.4f}")
    
    import math
    # Record current arm joint positions and replay them directly - no IK controller
    arm_joint_ids = robot.find_joints("panda_joint.*")[0]
    held_arm_pos = robot.data.joint_pos[env_id, arm_joint_ids].clone()
    finger_target = 0.0 if gripper_binary_action < 0.5 else math.pi / 4
    
    # Build full joint target: arm stays, gripper moves
    all_target = robot.data.joint_pos[env_id, :].clone().unsqueeze(0)
    finger_joint_idx = robot.find_joints("finger_joint")[0]
    all_target[0, finger_joint_idx] = finger_target
    
    # Step sim directly - no action manager, no IK controller
    for _ in range(steps):
        robot.set_joint_position_target(all_target)
        env.scene.write_data_to_sim()
        env.sim.step(render=True)
        env.scene.update(dt=env.physics_dt)
    
    joint_pos_after = robot.data.joint_pos[env_id, finger_idx].item()
    print(f"[GRIPPER] {state} completed | finger AFTER: {joint_pos_after:.4f}")


def _pose_from_pos_quat(pos_xyz: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    rot = math_utils.matrix_from_quat(quat_wxyz.unsqueeze(0))[0]
    return math_utils.make_pose(pos_xyz, rot)


def _get_object_pos(env, object_name: str, env_id: int = 0) -> torch.Tensor:
    """Get object position in robot-base frame (cuRobo's coordinate system).
    
    CuRobo's coordinate system has the robot base at the origin, so we need to
    convert from world coordinates to robot-base-relative coordinates.
    """
    obj = env.scene[object_name]
    robot = env.scene['robot']
    world_pos = obj.data.root_pos_w[env_id, :3]
    robot_base_pos = robot.data.root_pos_w[env_id, :3]
    robot_frame_pos = world_pos - robot_base_pos
    print(f"[DEBUG OBJ] {object_name}: world={world_pos}, robot_base={robot_base_pos}, robot_frame={robot_frame_pos}")
    return robot_frame_pos.clone().detach()


def _auto_pick_order(env, explicit_order: list[str] | None) -> list[str]:
    rigid_object_names = list(env.scene.rigid_objects.keys())
    
    if explicit_order is not None and len(explicit_order) == 1 and explicit_order[0].lower() == 'random':
        explicit_order = None
        shuffle = True
    else:
        shuffle = False
    
    if explicit_order is not None:
        missing = [name for name in explicit_order if name not in rigid_object_names]
        if missing:
            raise ValueError(
                f'Objects {missing} from --pick_order are not in scene rigid objects. Available: {rigid_object_names}'
            )
        return explicit_order

    excluded = {'blue_sorting_bin', 'ground_plane', 'office_table_background'}
    names = [name for name in rigid_object_names if name not in excluded]
    names = [name for name in names if 'table' not in name and 'light' not in name and 'stand' not in name]
    if shuffle:
        _random.shuffle(names)
    else:
        names.sort()
    return names


def _compute_placement_slots(
    bin_center_xyz: torch.Tensor,
    num_objects: int,
    bin_half_x: float,
    bin_half_y: float,
) -> list[torch.Tensor]:
    """Compute placement slots that fit inside the bin.

    Distributes `num_objects` positions evenly within the usable bin area
    so that every object lands inside the bin walls.
    """
    import math

    if num_objects <= 0:
        return []

    if num_objects == 1:
        return [bin_center_xyz.clone()]

    # Determine grid dimensions (cols x rows) that best fit the bin aspect ratio
    aspect = bin_half_x / max(bin_half_y, 1e-6)
    cols = max(1, round(math.sqrt(num_objects * aspect)))
    rows = max(1, math.ceil(num_objects / cols))
    # Rebalance so we don't over-allocate columns
    while cols * rows < num_objects:
        rows += 1

    # Compute step sizes; if only 1 row/col, place at center of that axis
    step_x = (2.0 * bin_half_x) / max(cols, 2) if cols > 1 else 0.0
    step_y = (2.0 * bin_half_y) / max(rows, 2) if rows > 1 else 0.0

    # Grid origin: center the grid within the bin
    origin_x = -step_x * (cols - 1) / 2.0
    origin_y = -step_y * (rows - 1) / 2.0

    slots = []
    for i in range(num_objects):
        col = i % cols
        row = i // cols
        offset_x = origin_x + col * step_x
        offset_y = origin_y + row * step_y
        slot = bin_center_xyz.clone()
        slot[0] += offset_x
        slot[1] += offset_y
        slots.append(slot)

    print(f"[SLOTS] {num_objects} placement slots in bin (half_x={bin_half_x}, half_y={bin_half_y}): "
          f"grid {cols}x{rows}, step=({step_x:.3f}, {step_y:.3f})")
    for i, s in enumerate(slots):
        dx = (s[0] - bin_center_xyz[0]).item()
        dy = (s[1] - bin_center_xyz[1]).item()
        print(f"  slot {i}: offset=({dx:+.3f}, {dy:+.3f})")
    return slots


def _save_sphere_debug_plot(snapshot: dict, png_path: Path) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    centers = np.array(snapshot['centers_xyz'], dtype=float)
    radii = np.array(snapshot['radii'], dtype=float)
    if centers.size == 0:
        return

    sizes = np.clip((radii * 4000.0) ** 2, 15.0, 1400.0)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    views = ((0, 1, 'XY'), (0, 2, 'XZ'), (1, 2, 'YZ'))

    for ax, (ix, iy, label) in zip(axes, views):
        sc = ax.scatter(centers[:, ix], centers[:, iy], c=radii, s=sizes, cmap='viridis', alpha=0.5)
        ax.set_title(f'{label} projection')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

    fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.04, label='Sphere radius [m]')
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)


def _dump_curobo_spheres(planner, stage: str, dump_dir: Path | None, save_png: bool = False) -> None:
    if dump_dir is None:
        return

    dump_dir.mkdir(parents=True, exist_ok=True)

    try:
        joint_state = planner._get_current_joint_state_for_curobo()
        sphere_list = planner.motion_gen.kinematics.get_robot_as_spheres(joint_state.position)[0]
    except Exception as exc:
        print(f'[WARN] Sphere dump failed before stage {stage}: {exc}')
        return

    centers = []
    radii = []
    for sphere in sphere_list:
        center = None
        radius = None
        if hasattr(sphere, 'position'):
            center = sphere.position
        elif hasattr(sphere, 'center'):
            center = sphere.center
        if hasattr(sphere, 'radius'):
            radius = sphere.radius

        if center is None or radius is None:
            continue

        center_t = torch.as_tensor(center).detach().float().cpu().view(-1)
        radius_t = torch.as_tensor(radius).detach().float().cpu().view(-1)
        if center_t.numel() < 3 or radius_t.numel() < 1:
            continue

        centers.append(center_t[:3].tolist())
        radii.append(float(radius_t[0].item()))

    stage_clean = stage.replace(':', '_').replace('/', '_')
    idx = len(list(dump_dir.glob('*.json')))
    out_json = dump_dir / f'{idx:04d}_{stage_clean}.json'

    payload = {
        'stage': stage,
        'num_spheres': len(centers),
        'attached_objects': planner.get_attached_objects(),
        'centers_xyz': centers,
        'radii': radii,
    }
    out_json.write_text(json.dumps(payload, indent=2))

    if save_png and centers:
        out_png = dump_dir / f'{idx:04d}_{stage_clean}.png'
        try:
            _save_sphere_debug_plot(payload, out_png)
        except Exception as exc:
            print(f'[WARN] Sphere PNG dump failed for {stage}: {exc}')


def _make_planner_cfg(args_cli: argparse.Namespace):
    import tempfile
    import yaml

    from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

    repo_root = Path(__file__).resolve().parents[3]
    robot_cfg_template = repo_root / 'assets_local' / 'droid_fixed_mimic_joint' / 'franka_robotiq_2f_85_zero_curobo.yml'
    local_urdf = repo_root / 'assets_local' / 'droid_fixed_mimic_joint' / 'urdf' / 'franka_robotiq_2f_85_zero.urdf'

    if not robot_cfg_template.exists():
        raise FileNotFoundError(f'CuRobo robot config file not found: {robot_cfg_template}')
    if not local_urdf.exists():
        raise FileNotFoundError(f'CuRobo URDF file not found: {local_urdf}')

    with robot_cfg_template.open('r') as f:
        robot_yaml = yaml.safe_load(f)

    robot_yaml['robot_cfg']['kinematics']['urdf_path'] = str(local_urdf)

    tmp_dir = Path(tempfile.mkdtemp(prefix='curobo_robot_cfg_'))
    robot_cfg_file = tmp_dir / 'franka_curobo_runtime.yml'
    with robot_cfg_file.open('w') as f:
        yaml.safe_dump(robot_yaml, f, sort_keys=False)

    # Keep all lock_joints from robot YAML locked in CuRobo and only toggle the actuated finger joint.
    lock_joints = dict(robot_yaml['robot_cfg']['kinematics']['lock_joints'])
    gripper_open_positions = dict(lock_joints)
    gripper_open_positions['finger_joint'] = 0.0
    gripper_closed_positions = dict(lock_joints)
    gripper_closed_positions['finger_joint'] = float(torch.pi / 4)

    return CuroboPlannerCfg(
        robot_config_file=str(robot_cfg_file),
        robot_name='franka_robotiq',
        ee_link_name='base_link',
        gripper_joint_names=['finger_joint'],
        gripper_open_positions=gripper_open_positions,
        gripper_closed_positions=gripper_closed_positions,
        hand_link_names=[
            'base_link',
            'left_inner_finger',
            'left_inner_knuckle',
            'left_outer_finger',
            'left_outer_knuckle',
            'right_inner_finger',
            'right_inner_knuckle',
            'right_outer_finger',
            'right_outer_knuckle',
        ],
        # Current planner grasp check uses first of last two robot joints; use permissive threshold for this embodiment.
        grasp_gripper_open_val=10.0,
        approach_distance=args_cli.approach_distance,
        retreat_distance=args_cli.retreat_distance,
        time_dilation_factor=args_cli.time_dilation_factor,
        collision_activation_distance=0.03,
        visualize_plan=False,
        visualize_spheres=False,
        debug_planner=args_cli.debug_planner,
        # Use None to fall back to the planner's default ignore list, which includes
        # f"{env_prim_path}/Robot" -- this substring also matches Robot_Stand.
        world_ignore_substrings=None,
    )


def _visualize_goal_pose(target_pose: torch.Tensor, goal_pose_visualizer, robot_base_pos: torch.Tensor, stage: str = '') -> None:
    """Visualize goal pose marker and log coordinates."""
    pos_robot_frame = target_pose[:3, 3].detach().cpu()
    pos_world_frame = pos_robot_frame + robot_base_pos.detach().cpu()
    quat = math_utils.quat_from_matrix(target_pose[:3, :3].unsqueeze(0)).detach().cpu()

    print(f"[GOAL MARKER] {stage}: robot_frame={pos_robot_frame.tolist()}, world={pos_world_frame.tolist()}")

    if goal_pose_visualizer is not None:
        goal_pose_visualizer.visualize(translations=pos_world_frame.unsqueeze(0), orientations=quat)


def _plan_and_execute(
    env,
    planner,
    target_pose: torch.Tensor,
    gripper_action: float,
    expected_attached_object: str | None,
    stage: str,
    sphere_dump_dir: Path | None = None,
    sphere_dump_png: bool = False,
    goal_pose_visualizer=None,
) -> bool:
    robot_base_pos = env.scene['robot'].data.root_pos_w[0, :3]
    _visualize_goal_pose(target_pose, goal_pose_visualizer, robot_base_pos, stage=stage)

    if goal_pose_visualizer is not None:
        current_pose = _get_current_eef_pose(env, planner)
        hold_action = _action_from_pose(env, planner, current_pose, gripper_action, env_id=0)
        for _ in range(3):
            env.step(hold_action)

    _dump_curobo_spheres(planner, f'{stage}_pre', sphere_dump_dir, save_png=sphere_dump_png)

    plan_ok = planner.update_world_and_plan_motion(
        target_pose=target_pose,
        expected_attached_object=expected_attached_object,
        env_id=0,
    )
    if not plan_ok:
        print(f"[FAIL] Planning failed at stage '{stage}'")
        _dump_curobo_spheres(planner, f'{stage}_failed', sphere_dump_dir, save_png=sphere_dump_png)
        return False

    _dump_curobo_spheres(planner, f'{stage}_planned', sphere_dump_dir, save_png=sphere_dump_png)
    _execute_plan(env=env, planner=planner, gripper_binary_action=gripper_action, env_id=0)
    return True


def _save_rerun_checkpoint(planner, label: str) -> None:
    """Best-effort explicit Rerun save checkpoint."""
    try:
        viz = getattr(planner, 'plan_visualizer', None)
        save_path = getattr(viz, 'save_path', None)
        if viz is None or save_path is None:
            return
        import rerun as rr

        rr.save(save_path)
        print(f'[RERUN] checkpoint saved after {label}: {save_path}')
    except Exception as exc:
        print(f'[RERUN] checkpoint save failed after {label}: {exc}')


def _run_sanity_check(
    planner,
    env,
    sphere_dump_dir: Path | None,
    sphere_dump_png: bool,
    goal_pose_visualizer=None,
) -> None:
    curr_pose = _get_current_eef_pose(env, planner)
    lifted_pose = curr_pose.clone()
    lifted_pose[2, 3] += 0.05
    robot_base_pos = env.scene['robot'].data.root_pos_w[0, :3]

    ok = _plan_and_execute(
        env=env,
        planner=planner,
        target_pose=lifted_pose,
        gripper_action=GRIPPER_OPEN_CMD,
        expected_attached_object=None,
        stage='sanity_lift_5cm',
        sphere_dump_dir=sphere_dump_dir,
        sphere_dump_png=sphere_dump_png,
        goal_pose_visualizer=goal_pose_visualizer,
    )
    print(f'[SANITY] lift-from-home planning success: {ok}')
    if ok:
        _save_rerun_checkpoint(planner, 'sanity_lift_5cm')


def main() -> None:
    args_parser = get_isaaclab_arena_cli_parser()
    add_script_args(args_parser)
    args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
    _add_script_args_to_subparsers(args_parser)
    args_cli = args_parser.parse_args()

    if args_cli.example_environment != 'droid_v2_tabletop_pick_and_place':
        raise ValueError('This script is intended for the droid_v2_tabletop_pick_and_place environment.')

    with SimulationAppContext(args_cli):
        global math_utils
        import isaaclab.utils.math as math_utils
        from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
        from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner

        sphere_dump_dir = Path(args_cli.dump_spheres_dir) if args_cli.dump_spheres_dir else None

        goal_pose_visualizer = None
        if not getattr(args_cli, 'headless', False):
            marker_cfg = deepcopy(FRAME_MARKER_CFG)
            marker_cfg.prim_path = '/World/Visuals/curobo_goal_pose'
            frame_marker = marker_cfg.markers.get('frame')
            if frame_marker is not None:
                setattr(frame_marker, 'scale', (0.08, 0.08, 0.08))
            goal_pose_visualizer = VisualizationMarkers(marker_cfg)

        arena_builder = get_arena_builder_from_cli(args_cli)
        env = arena_builder.make_registered()
        env.reset()
        
        robot = env.scene['robot']
        robot_world_pos = robot.data.root_pos_w[0, :3]
        print(f"[DEBUG INIT] Robot world position: {robot_world_pos}")
        print(f"[DEBUG INIT] CuRobo coordinate system: robot base at origin, all goals/objects in robot-base frame")
        joint_names = robot.joint_names
        for i, name in enumerate(joint_names):
            s = robot.data.default_joint_stiffness[0, i].item()
            d = robot.data.default_joint_damping[0, i].item()
            if 'finger' in name or 'knuckle' in name or 'panda' in name:
                print(f"[DEBUG JOINT] {name}: default_stiffness={s:.4f}, default_damping={d:.4f}")
        
        planner_cfg = _make_planner_cfg(args_cli)
        planner = CuroboPlanner(
            env=env,
            robot=robot,
            config=planner_cfg,
            env_id=0,
        )

    # ToDo (Neel): WIP rerun recording saving (does not work yet)
        if args_cli.rerun_recording_path is not None:
            import datetime as _dt

            rerun_path = Path(args_cli.rerun_recording_path).expanduser().resolve()
            if rerun_path.suffix.lower() != '.rrd':
                if rerun_path.exists() and rerun_path.is_dir():
                    ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
                    rerun_path = rerun_path / f'curobo_plan_{ts}.rrd'
                else:
                    ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
                    rerun_path = rerun_path / f'curobo_plan_{ts}.rrd' if rerun_path.suffix == '' else rerun_path.with_suffix('.rrd')

            rerun_path.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(planner, 'plan_visualizer'):
                planner.plan_visualizer.save_path = str(rerun_path)
                print(f'[RERUN] recording will be saved to: {rerun_path}')
            else:
                print('[RERUN] visualize_plan is disabled; no recording will be produced.')

        if args_cli.run_sanity_check:
            _run_sanity_check(
                planner,
                env,
                sphere_dump_dir,
                args_cli.dump_spheres_png,
                goal_pose_visualizer,
            )

        pick_order = _auto_pick_order(env, explicit_order=args_cli.pick_order)
        if args_cli.max_objects is not None:
            pick_order = pick_order[: args_cli.max_objects]
        if len(pick_order) == 0:
            raise RuntimeError('No pickable objects were found in scene.rigid_objects.')

        print(f'Resolved pick order: {pick_order}')

        bin_pos = _get_object_pos(env, 'blue_sorting_bin')
        print(f'Bin center position (robot frame): {bin_pos}')

        # Pre-compute placement slots so every object has a guaranteed in-bin position
        placement_slots = _compute_placement_slots(
            bin_pos, len(pick_order), args_cli.bin_half_x, args_cli.bin_half_y,
        )

        # Track success/failure for each object
        results = {}  # object_name -> {stage: success_bool}
        
        for idx, object_name in enumerate(pick_order):
            object_pos = _get_object_pos(env, object_name)
            print(f"\n{'='*80}")
            print(f"[{idx + 1}/{len(pick_order)}] Planning for object '{object_name}' at {object_pos}")
            print(f"{'='*80}")
            
            results[object_name] = {}

            # Plan directly to grasp pose (approach_distance auto-adds retreat+approach phases)
            grasp_xyz = object_pos.clone()
            grasp_xyz[2] += args_cli.grasp_height_offset + args_cli.goal_z_boost
            grasp_pose = _pose_from_pos_quat(grasp_xyz, DOWN_FACING_QUAT_WXYZ.to(env.device))
            
            # Compute reachability metrics
            current_eef_pose = _get_current_eef_pose(env, planner)
            current_eef_pos = current_eef_pose[:3, 3]
            distance_to_goal = torch.norm(grasp_xyz - current_eef_pos).item()
            
            print(f"[DEBUG GOAL] Current EEF (robot-frame): {current_eef_pos}")
            print(f"[DEBUG GOAL] Grasp XYZ (robot-frame): {grasp_xyz}")
            print(f"[DEBUG GOAL] Distance to goal: {distance_to_goal:.3f}m")
            if args_cli.goal_z_boost != 0.0:
                print(f"[DEBUG GOAL] goal_z_boost applied: {args_cli.goal_z_boost}m")
            print(f"[DEBUG GOAL] Approach distance: {args_cli.approach_distance}m (cuRobo multi-phase)")

            slot_center_xyz = placement_slots[idx]
            # place_above_xyz = slot_center_xyz.clone()
            # place_above_xyz[2] += args_cli.transport_height + args_cli.goal_z_boost
            # place_above_pose = _pose_from_pos_quat(place_above_xyz, DOWN_FACING_QUAT_WXYZ.to(env.device))

            place_xyz = slot_center_xyz.clone()
            place_xyz[2] += args_cli.place_height_offset + args_cli.goal_z_boost
            place_pose = _pose_from_pos_quat(place_xyz, DOWN_FACING_QUAT_WXYZ.to(env.device))

            # Single grasp motion (approach_distance adds multi-phase planning automatically)
            grasp_success = _plan_and_execute(
                env,
                planner,
                target_pose=grasp_pose,
                gripper_action=GRIPPER_OPEN_CMD,
                expected_attached_object=None,
                stage=f'{object_name}:grasp',
                sphere_dump_dir=sphere_dump_dir,
                sphere_dump_png=args_cli.dump_spheres_png,
                goal_pose_visualizer=goal_pose_visualizer,
            )
            results[object_name]['grasp'] = grasp_success
            if not grasp_success:
                print(f"[SKIP] Skipping remaining stages for '{object_name}' due to grasp planning failure")
                continue

            _execute_gripper_action(
                env,
                planner,
                gripper_binary_action=GRIPPER_CLOSE_CMD,
                steps=args_cli.gripper_settle_steps,
                env_id=0,
            )

            # Plan directly to place pose (approach_distance ideally should handles final approach)
            place_success = _plan_and_execute(
                env,
                planner,
                target_pose=place_pose,
                gripper_action=GRIPPER_CLOSE_CMD,
                expected_attached_object=object_name,
                stage=f'{object_name}:place',
                sphere_dump_dir=sphere_dump_dir,
                sphere_dump_png=args_cli.dump_spheres_png,
                goal_pose_visualizer=goal_pose_visualizer,
            )
            results[object_name]['place'] = place_success
            if not place_success:
                print(f"[SKIP] Skipping retreat for '{object_name}' due to place failure")
                _execute_gripper_action(env, planner, gripper_binary_action=GRIPPER_OPEN_CMD, steps=args_cli.gripper_settle_steps, env_id=0)
                continue

            _execute_gripper_action(
                env,
                planner,
                gripper_binary_action=GRIPPER_OPEN_CMD,
                steps=args_cli.gripper_settle_steps,
                env_id=0,
            )

            retreat_xyz = place_xyz.clone()
            retreat_xyz[2] += 0.10
            retreat_pose = _pose_from_pos_quat(retreat_xyz, DOWN_FACING_QUAT_WXYZ.to(env.device))
            retreat_success = _plan_and_execute(
                env,
                planner,
                target_pose=retreat_pose,
                gripper_action=GRIPPER_OPEN_CMD,
                expected_attached_object=None,
                stage=f'{object_name}:retreat',
                sphere_dump_dir=sphere_dump_dir,
                sphere_dump_png=args_cli.dump_spheres_png,
                goal_pose_visualizer=goal_pose_visualizer,
            )
            results[object_name]['retreat'] = retreat_success

        # Print summary
        print(f"\n{'='*80}")
        print("PICK AND PLACE RESULTS SUMMARY")
        print(f"{'='*80}")
        
        successful_objects = []
        failed_objects = []
        
        for obj_name, stages in results.items():
            all_success = all(stages.values()) if stages else False
            if all_success:
                successful_objects.append(obj_name)
                print(f"{obj_name}: SUCCESS (all stages completed)")
            else:
                failed_objects.append(obj_name)
                failed_stage = next((stage for stage, success in stages.items() if not success), 'unknown')
                print(f"{obj_name}: FAILED at stage '{failed_stage}'")
                for stage, success in stages.items():
                    status = "Successful" if success else "Failed"
                    print(f" {stage}: {status}")
        
        print(f"\nTotal: {len(successful_objects)}/{len(results)} objects successfully picked and placed")
        
        if failed_objects:
            print(f"\nFailed objects: {', '.join(failed_objects)}")
        if successful_objects:
            print(f"Successful objects: {', '.join(successful_objects)}")
        
        print(f"{'='*80}\n")
        
        if len(successful_objects) == 0:
            print('[INFO] All objects failed; skipping final EEF query.')
            env.close()
            return

        curr_pose = _get_current_eef_pose(env, planner)
        curr_quat = math_utils.quat_from_matrix(curr_pose[:3, :3].unsqueeze(0))[0]
        curr_pos = curr_pose[:3, 3]
        print(f'Final EEF pose: pos={curr_pos}, quat_wxyz={curr_quat}')

        for _ in range(10):
            env.step(_action_from_pose(env, planner, curr_pose, GRIPPER_OPEN_CMD))

        env.close()


if __name__ == '__main__':
    main()
