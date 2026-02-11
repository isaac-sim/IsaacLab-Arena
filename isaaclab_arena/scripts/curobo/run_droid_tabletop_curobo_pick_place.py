#!/usr/bin/env python3
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
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
        help='Explicit object pick order. Defaults to deterministic auto-discovery from scene rigid objects.',
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
        '--slot_spacing',
        type=float,
        default=0.05,
        help='XY spacing between successive placement slots in bin coordinates.',
    )
    parser.add_argument(
        '--gripper_settle_steps',
        type=int,
        default=16,
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
        default=1.0,
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
        default=True,
        help='Run one pre-flight lift planning check to distinguish setup issues from goal issues.',
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
    import isaaclab.utils.math as math_utils

    # Query current pose from CuRobo kinematics to avoid fragile ee_frame targets
    # (DROID config includes tool frames that may not exist for all gripper variants).
    joint_state = planner._get_current_joint_state_for_curobo()
    ee_pose = planner.get_ee_pose(joint_state)

    ee_pos = planner._to_env_device(ee_pose.position).view(-1, 3)[0]
    ee_quat = planner._to_env_device(ee_pose.quaternion).view(-1, 4)[0]
    ee_rot = math_utils.matrix_from_quat(ee_quat.unsqueeze(0))[0]
    return math_utils.make_pose(ee_pos, ee_rot)


def _action_from_pose(env, planner, target_pose: torch.Tensor, gripper_binary_action: float, env_id: int = 0) -> torch.Tensor:
    import isaaclab.utils.math as math_utils

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
    curr_pose = _get_current_eef_pose(env, planner, env_id=env_id)
    for _ in range(steps):
        action = _action_from_pose(env, planner, curr_pose, gripper_binary_action, env_id=env_id)
        env.step(action)


def _pose_from_pos_quat(pos_xyz: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    import isaaclab.utils.math as math_utils

    rot = math_utils.matrix_from_quat(quat_wxyz.unsqueeze(0))[0]
    return math_utils.make_pose(pos_xyz, rot)


def _get_object_pos(env, object_name: str, env_id: int = 0) -> torch.Tensor:
    obj = env.scene[object_name]
    env_origin = env.scene.env_origins[env_id, 0:3]
    return (obj.data.root_pos_w[env_id, :3] - env_origin).clone().detach()


def _auto_pick_order(env, explicit_order: list[str] | None) -> list[str]:
    rigid_object_names = list(env.scene.rigid_objects.keys())
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
    names.sort()
    return names


def _placement_slot(bin_center_xyz: torch.Tensor, slot_index: int, spacing: float) -> torch.Tensor:
    slot_offsets = (
        (0.00, 0.00),
        (1.00, 0.00),
        (-1.00, 0.00),
        (0.00, 1.00),
        (0.00, -1.00),
        (1.00, 1.00),
        (1.00, -1.00),
        (-1.00, 1.00),
        (-1.00, -1.00),
    )
    off_x, off_y = slot_offsets[slot_index % len(slot_offsets)]
    slot_center = bin_center_xyz.clone()
    slot_center[0] += off_x * spacing
    slot_center[1] += off_y * spacing
    return slot_center


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
    robot_cfg_template = repo_root / 'assets_local' / 'franka_curobo.yml'
    local_urdf = repo_root / 'assets_local' / 'urdf' / 'franka_curobo.urdf'

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
            'left_outer_finger',
            'right_inner_finger',
            'right_outer_finger',
        ],
        # Current planner grasp check uses first of last two robot joints; use permissive threshold for this embodiment.
        grasp_gripper_open_val=10.0,
        approach_distance=args_cli.approach_distance,
        retreat_distance=args_cli.retreat_distance,
        time_dilation_factor=args_cli.time_dilation_factor,
        collision_activation_distance=0.01,
        visualize_plan=True,
        visualize_spheres=False,
        debug_planner=args_cli.debug_planner,
        world_ignore_substrings=['/World/defaultGroundPlane', '/curobo'],
    )


def _visualize_goal_pose(target_pose: torch.Tensor, goal_pose_visualizer) -> None:
    if goal_pose_visualizer is None:
        return

    import isaaclab.utils.math as math_utils

    pos = target_pose[:3, 3].detach().cpu().unsqueeze(0)
    quat = math_utils.quat_from_matrix(target_pose[:3, :3].unsqueeze(0)).detach().cpu()
    goal_pose_visualizer.visualize(translations=pos, orientations=quat)


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
    _visualize_goal_pose(target_pose, goal_pose_visualizer)
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
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        import isaaclab.utils.math as math_utils
        from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner

        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        _add_script_args_to_subparsers(args_parser)
        args_cli = args_parser.parse_args()

        if args_cli.example_environment != 'droid_tabletop_pick_and_place':
            raise ValueError('This script is intended for the droid_tabletop_pick_and_place environment.')

        from copy import deepcopy

        from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers

        sphere_dump_dir = Path(args_cli.dump_spheres_dir) if args_cli.dump_spheres_dir else None

        goal_pose_visualizer = None
        if not getattr(args_cli, 'headless', True):
            marker_cfg = deepcopy(FRAME_MARKER_CFG)
            marker_cfg.prim_path = '/World/Visuals/curobo_goal_pose'
            frame_marker = marker_cfg.markers.get('frame')
            if frame_marker is not None:
                setattr(frame_marker, 'scale', (0.1, 0.1, 0.1))
            goal_pose_visualizer = VisualizationMarkers(marker_cfg)

        arena_builder = get_arena_builder_from_cli(args_cli)
        env = arena_builder.make_registered()
        env.reset()

        robot = env.scene['robot']
        planner_cfg = _make_planner_cfg(args_cli)
        planner = CuroboPlanner(
            env=env,
            robot=robot,
            config=planner_cfg,
            env_id=0,
        )

        if args_cli.rerun_recording_path is not None:
            import datetime as _dt

            rerun_path = Path(args_cli.rerun_recording_path).expanduser().resolve()
            # Accept either a directory path or a file path. Always write a .rrd file.
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
            _run_sanity_check(planner, env, sphere_dump_dir, args_cli.dump_spheres_png, goal_pose_visualizer)

        pick_order = _auto_pick_order(env, explicit_order=args_cli.pick_order)
        if args_cli.max_objects is not None:
            pick_order = pick_order[: args_cli.max_objects]
        if len(pick_order) == 0:
            raise RuntimeError('No pickable objects were found in scene.rigid_objects.')

        print(f'Resolved pick order: {pick_order}')

        bin_pos = _get_object_pos(env, 'blue_sorting_bin')
        print(f'Bin center position (env frame): {bin_pos}')

        had_any_failure = False
        for idx, object_name in enumerate(pick_order):
            object_pos = _get_object_pos(env, object_name)
            print(f"[{idx + 1}/{len(pick_order)}] Planning for object '{object_name}' at {object_pos}")

            pre_grasp_xyz = object_pos.clone()
            pre_grasp_xyz[2] += args_cli.pre_grasp_height
            pre_grasp_pose = _pose_from_pos_quat(pre_grasp_xyz, DOWN_FACING_QUAT_WXYZ.to(env.device))

            grasp_xyz = object_pos.clone()
            grasp_xyz[2] += args_cli.grasp_height_offset
            grasp_pose = _pose_from_pos_quat(grasp_xyz, DOWN_FACING_QUAT_WXYZ.to(env.device))

            slot_center_xyz = _placement_slot(bin_pos, idx, args_cli.slot_spacing)
            place_above_xyz = slot_center_xyz.clone()
            place_above_xyz[2] += args_cli.transport_height
            place_above_pose = _pose_from_pos_quat(place_above_xyz, DOWN_FACING_QUAT_WXYZ.to(env.device))

            place_xyz = slot_center_xyz.clone()
            place_xyz[2] += args_cli.place_height_offset
            place_pose = _pose_from_pos_quat(place_xyz, DOWN_FACING_QUAT_WXYZ.to(env.device))

            if not _plan_and_execute(
                env,
                planner,
                target_pose=pre_grasp_pose,
                gripper_action=GRIPPER_OPEN_CMD,
                expected_attached_object=None,
                stage=f'{object_name}:approach',
                sphere_dump_dir=sphere_dump_dir,
                sphere_dump_png=args_cli.dump_spheres_png,
                goal_pose_visualizer=goal_pose_visualizer,
            ):
                had_any_failure = True
                continue

            if not _plan_and_execute(
                env,
                planner,
                target_pose=grasp_pose,
                gripper_action=GRIPPER_OPEN_CMD,
                expected_attached_object=None,
                stage=f'{object_name}:descend',
                sphere_dump_dir=sphere_dump_dir,
                sphere_dump_png=args_cli.dump_spheres_png,
                goal_pose_visualizer=goal_pose_visualizer,
            ):
                had_any_failure = True
                continue

            _execute_gripper_action(
                env,
                planner,
                gripper_binary_action=GRIPPER_CLOSE_CMD,
                steps=args_cli.gripper_settle_steps,
                env_id=0,
            )

            if not _plan_and_execute(
                env,
                planner,
                target_pose=place_above_pose,
                gripper_action=GRIPPER_CLOSE_CMD,
                expected_attached_object=object_name,
                stage=f'{object_name}:transport',
                sphere_dump_dir=sphere_dump_dir,
                sphere_dump_png=args_cli.dump_spheres_png,
                goal_pose_visualizer=goal_pose_visualizer,
            ):
                had_any_failure = True
                continue

            if not _plan_and_execute(
                env,
                planner,
                target_pose=place_pose,
                gripper_action=GRIPPER_CLOSE_CMD,
                expected_attached_object=object_name,
                stage=f'{object_name}:place',
                sphere_dump_dir=sphere_dump_dir,
                sphere_dump_png=args_cli.dump_spheres_png,
                goal_pose_visualizer=goal_pose_visualizer,
            ):
                had_any_failure = True
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
            _plan_and_execute(
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

        if had_any_failure:
            print('[INFO] One or more planning stages failed; skipping final EEF query to avoid cascading CUDA assert state.')
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
