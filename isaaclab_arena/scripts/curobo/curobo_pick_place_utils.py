# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""
Shared CuRobo pick-and-place planning and execution utilities.

Single source of truth for droid_v2_tabletop CuRobo logic used by:
- run_droid_v2_tabletop_curobo_pick_place.py (interactive execution)
- record_curobo_demos.py (record plan execution to HDF5)
"""

from __future__ import annotations

import argparse
import json
import math
import random as _random
import types
from pathlib import Path

import torch

import isaaclab.utils.math as math_utils

# Constants
GRIPPER_OPEN_CMD: float = 0.0
GRIPPER_CLOSE_CMD: float = 1.0
DOWN_FACING_QUAT_WXYZ = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)

# Re-export CLI args from lightweight module (no torch/isaaclab) for single import in record_curobo_demos
from isaaclab_arena.scripts.curobo.curobo_cli_args import add_script_args, add_script_args_to_subparsers


def get_current_eef_pose(env, planner, env_id: int = 0) -> torch.Tensor:
    """Return current EEF pose as a 4x4 matrix in robot-base frame via CuRobo FK."""
    joint_state = planner._get_current_joint_state_for_curobo()
    ee_pose = planner.get_ee_pose(joint_state)
    ee_pos = planner._to_env_device(ee_pose.position).view(-1, 3)[0]
    ee_quat = planner._to_env_device(ee_pose.quaternion).view(-1, 4)[0]
    ee_rot = math_utils.matrix_from_quat(ee_quat.unsqueeze(0))[0]
    return math_utils.make_pose(ee_pos, ee_rot)


def compute_pose_error(pose_a: torch.Tensor, pose_b: torch.Tensor) -> tuple[float, float]:
    """Compute positional (m) and rotational (rad) error between two 4x4 poses."""
    pos_err = torch.norm(pose_a[:3, 3] - pose_b[:3, 3]).item()
    rot_err_mat = pose_a[:3, :3] @ pose_b[:3, :3].T
    trace_val = torch.clamp((rot_err_mat.trace() - 1.0) / 2.0, -1.0, 1.0)
    rot_err = torch.acos(trace_val).item()
    return pos_err, rot_err


def action_from_pose(
    env, planner, target_pose: torch.Tensor, gripper_binary_action: float, env_id: int = 0
) -> torch.Tensor:
    """Build a delta-pose + gripper action to drive the EEF toward target_pose."""
    target_pos, target_rot = math_utils.unmake_pose(target_pose)
    curr_pose = get_current_eef_pose(env, planner, env_id=env_id)
    curr_pos, curr_rot = math_utils.unmake_pose(curr_pose)
    delta_position = target_pos - curr_pos
    delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
    delta_quat = math_utils.quat_from_matrix(delta_rot_mat.unsqueeze(0))[0]
    delta_rotation = math_utils.axis_angle_from_quat(delta_quat.unsqueeze(0))[0]
    pose_action = torch.cat([delta_position, delta_rotation], dim=0)
    gripper_action = torch.tensor([gripper_binary_action], device=env.device, dtype=torch.float32)
    return torch.cat([pose_action, gripper_action], dim=0).unsqueeze(0)


def update_ee_visualizer(ee_visualizer, eef_pose: torch.Tensor, robot_base_pos: torch.Tensor) -> None:
    """Move the EEF tracking marker to match the given pose (robot-base frame)."""
    if ee_visualizer is None:
        return
    pos_world = (eef_pose[:3, 3] + robot_base_pos).detach().cpu()
    quat = math_utils.quat_from_matrix(eef_pose[:3, :3].unsqueeze(0)).detach().cpu()
    ee_visualizer.visualize(translations=pos_world.unsqueeze(0), orientations=quat)


def execute_plan(
    env,
    planner,
    gripper_binary_action: float,
    env_id: int = 0,
    converge_pos_threshold: float = 0.001,
    converge_rot_threshold: float = 0.05,
    max_converge_steps: int = 100,
    ee_visualizer=None,
    use_env_step_batch: bool = True,
) -> None:
    """Execute planned poses. use_env_step_batch=True for recording (env.step with batched action)."""
    planned_poses = planner.get_planned_poses()
    if not planned_poses:
        return
    robot_base_pos = env.scene["robot"].data.root_pos_w[env_id, :3]
    for pose in planned_poses:
        action = action_from_pose(env, planner, pose, gripper_binary_action, env_id=env_id)
        if use_env_step_batch:
            env.step(action.repeat(env.num_envs, 1))
        else:
            env.step(action)
        if ee_visualizer is not None:
            update_ee_visualizer(ee_visualizer, get_current_eef_pose(env, planner, env_id), robot_base_pos)
    final_pose = planned_poses[-1]
    curr_pose = get_current_eef_pose(env, planner, env_id=env_id)
    pos_err, rot_err = compute_pose_error(final_pose, curr_pose)
    if not use_env_step_batch:
        print(f"[TRACKING] After open-loop: pos_err={pos_err:.4f}m, rot_err={rot_err:.4f}rad")
    if max_converge_steps <= 0:
        return
    for step in range(max_converge_steps):
        if pos_err < converge_pos_threshold and rot_err < converge_rot_threshold:
            if not use_env_step_batch:
                print(f"[CONVERGE] Reached final waypoint in {step} extra steps")
            return
        action = action_from_pose(env, planner, final_pose, gripper_binary_action, env_id=env_id)
        if use_env_step_batch:
            env.step(action.repeat(env.num_envs, 1))
        else:
            env.step(action)
        curr_pose = get_current_eef_pose(env, planner, env_id=env_id)
        if ee_visualizer is not None:
            update_ee_visualizer(ee_visualizer, curr_pose, robot_base_pos)
        pos_err, rot_err = compute_pose_error(final_pose, curr_pose)
    if not use_env_step_batch:
        print(f"[CONVERGE] Did not fully converge after {max_converge_steps} steps")


def execute_gripper_action(env, planner, gripper_binary_action: float, steps: int = 12, env_id: int = 0) -> None:
    """Open/close gripper by direct sim step (not recorded). Use for interactive run script."""
    state = "OPEN" if gripper_binary_action < 0.5 else "CLOSE"
    robot = env.scene["robot"]
    finger_idx = robot.find_joints("finger_joint")[0]
    finger_target = 0.0 if gripper_binary_action < 0.5 else math.pi / 4
    all_target = robot.data.joint_pos[env_id, :].clone().unsqueeze(0)
    all_target[0, finger_idx] = finger_target
    for _ in range(steps):
        robot.set_joint_position_target(all_target)
        env.scene.write_data_to_sim()
        env.sim.step(render=True)
        env.scene.update(dt=env.physics_dt)


def execute_gripper_action_recordable(
    env, planner, gripper_binary_action: float, steps: int, env_id: int = 0
) -> None:
    """Open/close gripper using env.step with hold pose so the recorder captures every step."""
    current_pose = get_current_eef_pose(env, planner, env_id=env_id)
    action = action_from_pose(env, planner, current_pose, gripper_binary_action, env_id=env_id)
    for _ in range(steps):
        env.step(action.repeat(env.num_envs, 1))


def pose_from_pos_quat(pos_xyz: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    rot = math_utils.matrix_from_quat(quat_wxyz.unsqueeze(0))[0]
    return math_utils.make_pose(pos_xyz, rot)


def get_object_pos(env, object_name: str, env_id: int = 0, verbose: bool = True) -> torch.Tensor:
    """Get object position in robot root-link frame (CuRobo base_link for Droid). Same transform as _sync_robot_base_frame."""
    obj = env.scene[object_name]
    robot = env.scene["robot"]
    obj_pos_w = obj.data.root_pos_w[env_id, :3]
    robot_pos_w = robot.data.root_pos_w[env_id, :3]
    robot_quat_w = robot.data.root_quat_w[env_id, :4]
    R_w2r = math_utils.matrix_from_quat(robot_quat_w.unsqueeze(0))[0].T
    pos_robot = (R_w2r @ (obj_pos_w - robot_pos_w).unsqueeze(-1)).squeeze(-1)
    if verbose:
        print(f"[DEBUG OBJ] {object_name}: world={obj_pos_w}, root_link={robot_pos_w}, robot_frame={pos_robot}")
    return pos_robot.clone().detach()


def get_bin_interior_center(
    env,
    bin_name: str,
    env_id: int = 0,
    verbose: bool = True,
) -> torch.Tensor:
    """Bin center for placement in robot root-link frame (CuRobo base_link for Droid). Same transform as _sync_robot_base_frame."""
    obj = env.scene[bin_name]
    robot = env.scene["robot"]
    obj_pos_w = obj.data.root_pos_w[env_id, :3]
    robot_pos_w = robot.data.root_pos_w[env_id, :3]
    robot_quat_w = robot.data.root_quat_w[env_id, :4]
    R_w2r = math_utils.matrix_from_quat(robot_quat_w.unsqueeze(0))[0].T
    pos_robot = (R_w2r @ (obj_pos_w - robot_pos_w).unsqueeze(-1)).squeeze(-1)
    if verbose:
        print(
            f"[DEBUG BIN] {bin_name} interior center: world={obj_pos_w}, robot_frame={pos_robot}"
        )
    return pos_robot.clone().detach()


def get_object_quat(env, object_name: str, env_id: int = 0) -> torch.Tensor:
    obj = env.scene[object_name]
    return obj.data.root_quat_w[env_id, :4].clone().detach()


def compute_grasp_quat(strategy: str, object_quat_wxyz: torch.Tensor, device) -> torch.Tensor:
    down_quat = DOWN_FACING_QUAT_WXYZ.to(device)
    if strategy == "top_down":
        return down_quat
    if strategy == "object_yaw":
        obj_euler = math_utils.euler_xyz_from_quat(object_quat_wxyz.unsqueeze(0))
        yaw = obj_euler[2][0]
        yaw_quat = math_utils.quat_from_euler_xyz(
            torch.zeros(1, device=device), torch.zeros(1, device=device), yaw.unsqueeze(0)
        )[0]
        return math_utils.quat_mul(yaw_quat.unsqueeze(0), down_quat.unsqueeze(0))[0]
    if strategy == "object_aligned":
        return math_utils.quat_mul(object_quat_wxyz.unsqueeze(0), down_quat.unsqueeze(0))[0]
    raise ValueError(f"Unknown grasp orientation strategy: {strategy}")


def auto_pick_order(env, explicit_order: list[str] | None) -> list[str]:
    rigid_object_names = list(env.scene.rigid_objects.keys())
    if explicit_order is not None and len(explicit_order) == 1 and explicit_order[0].lower() == "random":
        explicit_order = None
        shuffle = True
    else:
        shuffle = False
    if explicit_order is not None:
        missing = [name for name in explicit_order if name not in rigid_object_names]
        if missing:
            raise ValueError(
                f"Objects {missing} from --pick_order are not in scene. Available: {rigid_object_names}"
            )
        return explicit_order
    excluded = {"blue_sorting_bin", "ground_plane", "office_table_background"}
    names = [n for n in rigid_object_names if n not in excluded]
    names = [n for n in names if "table" not in n and "light" not in n and "stand" not in n]
    if shuffle:
        _random.shuffle(names)
    else:
        names.sort()
    return names


def compute_placement_slots(
    bin_center_xyz: torch.Tensor,
    num_objects: int,
    bin_half_x: float,
    bin_half_y: float,
    verbose: bool = True,
) -> list[torch.Tensor]:
    """Place 1st center +x, 2nd center -x, 3rd center +y, 4th center -y; 5+ corners."""
    if num_objects <= 0:
        return []
    # Fraction of half-extent so positions are far from center but inside bin
    margin = 0.6
    offset_x = margin * bin_half_x
    offset_y = margin * bin_half_y

    slots = []
    # 1st: center +x
    slot = bin_center_xyz.clone()
    slot[0] += offset_x
    slots.append(slot)
    if num_objects <= 1:
        if verbose:
            _log_slots(slots, bin_center_xyz)
        return slots

    # 2nd: center -x
    slot = bin_center_xyz.clone()
    slot[0] -= offset_x
    slots.append(slot)
    if num_objects <= 2:
        if verbose:
            _log_slots(slots, bin_center_xyz)
        return slots

    # 3rd: center +y
    slot = bin_center_xyz.clone()
    slot[1] += offset_y
    slots.append(slot)
    if num_objects <= 3:
        if verbose:
            _log_slots(slots, bin_center_xyz)
        return slots

    # 4th: center -y
    slot = bin_center_xyz.clone()
    slot[1] -= offset_y
    slots.append(slot)
    if num_objects <= 4:
        if verbose:
            _log_slots(slots, bin_center_xyz)
        return slots

    # 5th+: corners (+x+y, -x+y, -x-y, +x-y), then repeat
    corners = [
        (offset_x, offset_y),
        (-offset_x, offset_y),
        (-offset_x, -offset_y),
        (offset_x, -offset_y),
    ]
    for i in range(4, num_objects):
        cx, cy = corners[(i - 4) % 4]
        slot = bin_center_xyz.clone()
        slot[0] += cx
        slot[1] += cy
        slots.append(slot)

    if verbose:
        _log_slots(slots, bin_center_xyz)
    return slots


def _log_slots(slots: list, bin_center_xyz: torch.Tensor) -> None:
    """Log slot offsets from bin center."""
    print(f"[SLOTS] {len(slots)} placement slots (1st=+x, 2nd=-x, 3rd=+y, 4th=-y, 5+=corners)")
    for i, s in enumerate(slots):
        dx = (s[0] - bin_center_xyz[0]).item()
        dy = (s[1] - bin_center_xyz[1]).item()
        print(f"  slot {i}: offset=({dx:+.3f}, {dy:+.3f})")


def make_planner_cfg(args_cli: argparse.Namespace):
    import tempfile
    import yaml

    from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

    repo_root = Path(__file__).resolve().parents[3]
    robot_cfg_template = repo_root / "assets_local" / "droid_fixed_mimic_joint" / "franka_robotiq_2f_85_zero_curobo.yml"
    local_urdf = repo_root / "assets_local" / "droid_fixed_mimic_joint" / "urdf" / "franka_robotiq_2f_85_zero.urdf"
    if not robot_cfg_template.exists():
        raise FileNotFoundError(f"CuRobo robot config not found: {robot_cfg_template}")
    if not local_urdf.exists():
        raise FileNotFoundError(f"CuRobo URDF not found: {local_urdf}")
    with robot_cfg_template.open("r") as f:
        robot_yaml = yaml.safe_load(f)
    robot_yaml["robot_cfg"]["kinematics"]["urdf_path"] = str(local_urdf)
    tmp_dir = Path(tempfile.mkdtemp(prefix="curobo_robot_cfg_"))
    robot_cfg_file = tmp_dir / "franka_curobo_runtime.yml"
    with robot_cfg_file.open("w") as f:
        yaml.safe_dump(robot_yaml, f, sort_keys=False)
    lock_joints = dict(robot_yaml["robot_cfg"]["kinematics"]["lock_joints"])
    gripper_open_positions = dict(lock_joints)
    gripper_open_positions["finger_joint"] = 0.0
    gripper_closed_positions = dict(lock_joints)
    gripper_closed_positions["finger_joint"] = float(torch.pi / 4)
    return CuroboPlannerCfg(
        robot_config_file=str(robot_cfg_file),
        robot_name="franka_robotiq",
        ee_link_name="base_link",
        gripper_joint_names=["finger_joint"],
        gripper_open_positions=gripper_open_positions,
        gripper_closed_positions=gripper_closed_positions,
        hand_link_names=[
            "base_link",
            "left_inner_finger", "left_inner_knuckle", "left_outer_finger", "left_outer_knuckle",
            "right_inner_finger", "right_inner_knuckle", "right_outer_finger", "right_outer_knuckle",
        ],
        grasp_gripper_open_val=10.0,
        approach_distance=args_cli.approach_distance,
        retreat_distance=args_cli.retreat_distance,
        time_dilation_factor=args_cli.time_dilation_factor,
        collision_activation_distance=0.05,
        motion_step_size=None,
        trajopt_tsteps=42,
        visualize_plan=False,
        visualize_spheres=False,
        debug_planner=getattr(args_cli, "debug_planner", False),
        world_ignore_substrings=None,
    )


def fix_planner_object_sync_frame(planner) -> None:
    """Replace the planner's object sync so poses are in robot-base frame."""
    _orig_logger = planner.logger

    def _sync_robot_base_frame(self):
        object_mappings = self._get_object_mappings()
        world_model = self.motion_gen.world_coll_checker.world_model
        rigid_objects = self.env.scene.rigid_objects
        robot_pos_w = self.robot.data.root_pos_w[self.env_id, :3]
        robot_quat_w = self.robot.data.root_quat_w[self.env_id, :4]
        R_w2r = math_utils.matrix_from_quat(robot_quat_w.unsqueeze(0))[0].T
        robot_quat_inv = math_utils.quat_inv(robot_quat_w.unsqueeze(0))[0]
        updated_count = 0
        for object_name, object_path in object_mappings.items():
            if object_name not in rigid_objects:
                continue
            static_objects = getattr(self.config, "static_objects", [])
            if any(s in object_name.lower() for s in static_objects):
                continue
            obj = rigid_objects[object_name]
            obj_pos_w = obj.data.root_pos_w[self.env_id, :3]
            obj_quat_w = obj.data.root_quat_w[self.env_id, :4]
            pos_robot = (R_w2r @ (obj_pos_w - robot_pos_w).unsqueeze(-1)).squeeze(-1)
            quat_robot = math_utils.quat_mul(robot_quat_inv.unsqueeze(0), obj_quat_w.unsqueeze(0))[0]
            pos_c = self._to_curobo_device(pos_robot)
            quat_c = self._to_curobo_device(quat_robot)
            pose_list = [
                float(pos_c[0]), float(pos_c[1]), float(pos_c[2]),
                float(quat_c[0]), float(quat_c[1]), float(quat_c[2]), float(quat_c[3]),
            ]
            if self._update_object_in_world_model(world_model, object_name, object_path, pose_list):
                updated_count += 1
        if updated_count > 0:
            for object_name, object_path in object_mappings.items():
                if object_name not in rigid_objects:
                    continue
                static_objects = getattr(self.config, "static_objects", [])
                if any(s in object_name.lower() for s in static_objects):
                    continue
                obj = rigid_objects[object_name]
                obj_pos_w = obj.data.root_pos_w[self.env_id, :3]
                obj_quat_w = obj.data.root_quat_w[self.env_id, :4]
                pos_robot = (R_w2r @ (obj_pos_w - robot_pos_w).unsqueeze(-1)).squeeze(-1)
                quat_robot = math_utils.quat_mul(robot_quat_inv.unsqueeze(0), obj_quat_w.unsqueeze(0))[0]
                curobo_pose = self._make_pose(
                    position=self._to_curobo_device(pos_robot),
                    quaternion=self._to_curobo_device(quat_robot),
                )
                self.motion_gen.world_coll_checker.update_obstacle_pose(
                    object_path, curobo_pose, update_cpu_reference=True
                )
        _orig_logger.debug(f"SYNC (robot-base frame): Updated {updated_count} object poses")
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    planner._sync_object_poses_with_isaaclab = types.MethodType(_sync_robot_base_frame, planner)


def visualize_goal_pose(
    target_pose: torch.Tensor, goal_pose_visualizer, robot_base_pos: torch.Tensor, stage: str = ""
) -> None:
    pos_robot_frame = target_pose[:3, 3].detach().cpu()
    pos_world_frame = pos_robot_frame + robot_base_pos.detach().cpu()
    quat = math_utils.quat_from_matrix(target_pose[:3, :3].unsqueeze(0)).detach().cpu()
    print(f"[GOAL MARKER] {stage}: robot_frame={pos_robot_frame.tolist()}, world={pos_world_frame.tolist()}")
    if goal_pose_visualizer is not None:
        goal_pose_visualizer.visualize(translations=pos_world_frame.unsqueeze(0), orientations=quat)


def save_sphere_debug_plot(snapshot: dict, png_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    centers = np.array(snapshot["centers_xyz"], dtype=float)
    radii = np.array(snapshot["radii"], dtype=float)
    if centers.size == 0:
        return
    sizes = np.clip((radii * 4000.0) ** 2, 15.0, 1400.0)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    views = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    for ax, (ix, iy, label) in zip(axes, views):
        sc = ax.scatter(centers[:, ix], centers[:, iy], c=radii, s=sizes, cmap="viridis", alpha=0.5)
        ax.set_title(f"{label} projection")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.04, label="Sphere radius [m]")
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)


def dump_curobo_spheres(planner, stage: str, dump_dir: Path | None, save_png: bool = False) -> None:
    if dump_dir is None:
        return
    dump_dir.mkdir(parents=True, exist_ok=True)
    try:
        joint_state = planner._get_current_joint_state_for_curobo()
        sphere_list = planner.motion_gen.kinematics.get_robot_as_spheres(joint_state.position)[0]
    except Exception as exc:
        print(f"[WARN] Sphere dump failed before stage {stage}: {exc}")
        return
    centers, radii = [], []
    for sphere in sphere_list:
        center = getattr(sphere, "position", None) or getattr(sphere, "center", None)
        radius = getattr(sphere, "radius", None)
        if center is None or radius is None:
            continue
        center_t = torch.as_tensor(center).detach().float().cpu().view(-1)
        radius_t = torch.as_tensor(radius).detach().float().cpu().view(-1)
        if center_t.numel() < 3 or radius_t.numel() < 1:
            continue
        centers.append(center_t[:3].tolist())
        radii.append(float(radius_t[0].item()))
    stage_clean = stage.replace(":", "_").replace("/", "_")
    idx = len(list(dump_dir.glob("*.json")))
    out_json = dump_dir / f"{idx:04d}_{stage_clean}.json"
    payload = {
        "stage": stage,
        "num_spheres": len(centers),
        "attached_objects": planner.get_attached_objects(),
        "centers_xyz": centers,
        "radii": radii,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    if save_png and centers:
        out_png = dump_dir / f"{idx:04d}_{stage_clean}.png"
        try:
            save_sphere_debug_plot(payload, out_png)
        except Exception as exc:
            print(f"[WARN] Sphere PNG dump failed for {stage}: {exc}")


def log_goal_vs_achieved(env, planner, target_pose: torch.Tensor, stage: str) -> None:
    achieved_pose = get_current_eef_pose(env, planner)
    pos_err, rot_err = compute_pose_error(target_pose, achieved_pose)
    goal_pos = target_pose[:3, 3]
    achieved_pos = achieved_pose[:3, 3]
    per_axis_err = (goal_pos - achieved_pos).tolist()
    goal_quat = math_utils.quat_from_matrix(target_pose[:3, :3].unsqueeze(0))[0]
    achieved_quat = math_utils.quat_from_matrix(achieved_pose[:3, :3].unsqueeze(0))[0]
    print(f"[DEBUG GOAL] --- {stage} ---")
    print(f"[DEBUG GOAL] Goal pos:     {goal_pos.tolist()}")
    print(f"[DEBUG GOAL] Achieved pos: {achieved_pos.tolist()}")
    print(f"[DEBUG GOAL] Pos error:    {per_axis_err} (norm: {pos_err:.4f} m)")
    print(f"[DEBUG GOAL] Goal quat:     {goal_quat.tolist()}")
    print(f"[DEBUG GOAL] Achieved quat: {achieved_quat.tolist()}")
    print(f"[DEBUG GOAL] Rot error:    {rot_err:.4f} rad ({rot_err * 180.0 / 3.14159265:.2f} deg)")


def plan_and_execute(
    env,
    planner,
    target_pose: torch.Tensor,
    gripper_action: float,
    expected_attached_object: str | None,
    stage: str,
    sphere_dump_dir: Path | None = None,
    sphere_dump_png: bool = False,
    goal_pose_visualizer=None,
    ee_visualizer=None,
    debug_goal: bool = False,
    use_env_step_batch: bool = False,
) -> bool:
    """Plan to target_pose and execute. use_env_step_batch=True for recording (batched env.step)."""
    robot_base_pos = env.scene["robot"].data.root_pos_w[0, :3]
    visualize_goal_pose(target_pose, goal_pose_visualizer, robot_base_pos, stage=stage)
    if goal_pose_visualizer is not None:
        current_pose = get_current_eef_pose(env, planner)
        hold_action = action_from_pose(env, planner, current_pose, gripper_action, env_id=0)
        for _ in range(3):
            env.step(hold_action.repeat(env.num_envs, 1) if use_env_step_batch else hold_action)
    dump_curobo_spheres(planner, f"{stage}_pre", sphere_dump_dir, save_png=sphere_dump_png)
    plan_ok = planner.update_world_and_plan_motion(
        target_pose=target_pose,
        expected_attached_object=expected_attached_object,
        env_id=0,
        step_size=planner.step_size,
    )
    if not plan_ok:
        print(f"[FAIL] Planning failed at stage '{stage}'")
        dump_curobo_spheres(planner, f"{stage}_failed", sphere_dump_dir, save_png=sphere_dump_png)
        return False
    dump_curobo_spheres(planner, f"{stage}_planned", sphere_dump_dir, save_png=sphere_dump_png)
    execute_plan(
        env=env,
        planner=planner,
        gripper_binary_action=gripper_action,
        env_id=0,
        ee_visualizer=ee_visualizer,
        use_env_step_batch=use_env_step_batch,
    )
    if debug_goal:
        log_goal_vs_achieved(env, planner, target_pose, stage)
    return True
