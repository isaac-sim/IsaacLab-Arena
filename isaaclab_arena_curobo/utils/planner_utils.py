# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import tempfile
import torch
import yaml
from pathlib import Path
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import warp as wp
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

from isaaclab_arena_curobo.embodiment_curobo_registry import get_embodiment_curobo_cfg
from isaaclab_arena_curobo.utils.frame_utils import top_down_grasp_matrix, world_pose_to_robot_frame
from isaaclab_arena_curobo.utils.robot_cfg_utils import load_patched_robot_yaml

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase


def make_planner_cfg(
    embodiment: EmbodimentBase,
    debug_planner: bool = False,
):
    """Build a CuroboPlannerCfg from an embodiment's cuRobo description.

    Reads the robot identity and planner tuning (asset paths, link/joint names, gripper targets,
    approach/retreat/time-dilation) from the cuRobo embodiment registry (keyed by ``embodiment.name``),
    patches the bundled robot config with the downloaded URDF path and the gripper open/closed joint
    targets, and returns a config ready for ``CuroboPlanner``.

    Args:
        embodiment: Embodiment whose name has a registered CuroboEmbodimentCfg (see
            ``embodiment_curobo_registry``).
        debug_planner: Enable cuRobo planner debug output.
    """
    curobo_cfg = get_embodiment_curobo_cfg(embodiment)

    # cuRobo reads real on-disk files, so pull + patch the robot config (URDF spliced in) from the
    # asset server, then re-dump it to a temp file CuroboPlannerCfg can point at.
    robot_yaml = load_patched_robot_yaml(curobo_cfg)

    tmp_dir = Path(tempfile.mkdtemp(prefix="curobo_robot_cfg_"))
    robot_cfg_file = tmp_dir / "curobo_runtime.yml"
    with robot_cfg_file.open("w") as f:
        yaml.safe_dump(robot_yaml, f, sort_keys=False)

    # Start from the robot config's locked joints, then overlay the embodiment's gripper targets.
    lock_joints = dict(robot_yaml["robot_cfg"]["kinematics"]["lock_joints"])
    gripper_open_positions = {**lock_joints, **curobo_cfg.gripper_open_joint_pos}
    gripper_closed_positions = {**lock_joints, **curobo_cfg.gripper_closed_joint_pos}

    return CuroboPlannerCfg(
        robot_config_file=str(robot_cfg_file),
        robot_name=curobo_cfg.robot_name,
        ee_link_name=curobo_cfg.ee_link_name,
        gripper_joint_names=curobo_cfg.gripper_joint_names,
        gripper_open_positions=gripper_open_positions,
        gripper_closed_positions=gripper_closed_positions,
        hand_link_names=curobo_cfg.hand_link_names,
        grasp_gripper_open_val=curobo_cfg.grasp_gripper_open_val,
        approach_distance=curobo_cfg.approach_distance,
        retreat_distance=curobo_cfg.retreat_distance,
        time_dilation_factor=curobo_cfg.time_dilation_factor,
        collision_activation_distance=curobo_cfg.collision_activation_distance,
        motion_step_size=None,
        trajopt_tsteps=curobo_cfg.trajopt_tsteps,
        visualize_plan=False,
        visualize_spheres=False,
        debug_planner=debug_planner,
        world_ignore_substrings=curobo_cfg.world_ignore_substrings,
    )


def sync_object_poses_in_robot_base_frame(planner) -> None:
    """Push every dynamic object's pose into the planner's collision world, in the robot base frame."""
    # USD prim path per scene object, used as the key into cuRobo's collision world model.
    object_mappings = planner._get_object_mappings()
    world_model = planner.motion_gen.world_coll_checker.world_model
    rigid_objects = planner.env.scene.rigid_objects
    # Robot base pose in world frame.
    # Frames: W = world, R = robot base, O = object. root_quat_w is (x, y, z, w)
    t_W_R = wp.to_torch(planner.robot.data.root_pos_w)[planner.env_id, :3]
    q_W_R_xyzw = wp.to_torch(planner.robot.data.root_quat_w)[planner.env_id, :4]
    updated_count = 0
    for object_name, object_path in object_mappings.items():
        # Skip objects flagged static in the planner config (e.g. the table) — they aren't re-synced.
        static_objects = getattr(planner.config, "static_objects", [])
        if object_name in rigid_objects and not any(s in object_name.lower() for s in static_objects):
            obj = rigid_objects[object_name]
            # Object pose in world frame (warp -> torch).
            t_W_O = wp.to_torch(obj.data.root_pos_w)[planner.env_id, :3]
            q_W_O_xyzw = wp.to_torch(obj.data.root_quat_w)[planner.env_id, :4]
            # Re-express the object pose in the robot base frame so it matches the IK goal frame.
            t_R_O, q_R_O_xyzw = world_pose_to_robot_frame(t_W_O, q_W_O_xyzw, t_W_R, q_W_R_xyzw)
            # Convert the object pose to the cuRobo device.
            pos_c = planner._to_curobo_device(t_R_O)
            quat_c_xyzw = planner._to_curobo_device(q_R_O_xyzw)
            # The world-model writer wants a flat [x, y, z, qw, qx, qy, qz] list (position + wxyz quat).
            quat_c_wxyz = math_utils.convert_quat(quat_c_xyzw, to="wxyz")
            pose_list = torch.cat((pos_c, quat_c_wxyz)).tolist()
            # Update the CPU-side world model first; only push to the GPU collision checker if it takes.
            if planner._update_object_in_world_model(world_model, object_name, object_path, pose_list):
                curobo_pose = planner._make_pose(position=pos_c, quaternion=quat_c_wxyz, quat_is_xyzw=False)
                planner.motion_gen.world_coll_checker.update_obstacle_pose(
                    object_path, curobo_pose, update_cpu_reference=True
                )
                updated_count += 1
    planner.logger.debug(f"SYNC (robot-base frame): Updated {updated_count} object poses")
    # Block until the pose writes land so the next IK solve sees the synced world.
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_curobo_planner(
    env: ManagerBasedEnv,
    embodiment: EmbodimentBase,
    env_id: int = 0,
    robot_scene_name: str | None = None,
    debug_planner: bool = False,
):
    """Construct a CuroboPlanner for an embodiment, bound to the env's robot and the robot base frame.

    The robot identity and planner tuning come entirely from the cuRobo embodiment registry (keyed by
    ``embodiment.name``; it errors if none is registered), so this is robot-agnostic.

    Args:
        env: The (unwrapped) Isaac Lab env; must expose the robot articulation in its scene.
        embodiment: Embodiment whose name has a registered CuroboEmbodimentCfg.
        env_id: Index of the parallel env the planner reads object/robot poses from.
        robot_scene_name: Scene key of the robot articulation. Defaults to the embodiment's scene name.
        debug_planner: Enable cuRobo planner debug output.
    """
    if robot_scene_name is None:
        robot_scene_name = embodiment.get_embodiment_name_in_scene()
    planner_cfg = make_planner_cfg(embodiment, debug_planner=debug_planner)
    # cuRobo-Lab's MotionGen/collision world is single-env only for now.
    planner = CuroboPlanner(env=env, robot=env.scene[robot_scene_name], config=planner_cfg, env_id=env_id)
    return planner


def top_down_grasp_pose_from_env(
    env: ManagerBasedEnv,
    object_name: str,
    grasp_z_offset: float = 0.02,
    env_id: int = 0,
    robot_scene_name: str = "robot",
    align_yaw_to_object: bool = True,
) -> torch.Tensor:
    """Top-down grasp pose at an object's center, expressed in the robot base frame.

    Args:
        env: The (unwrapped) Isaac Lab env holding both the robot and the target object.
        object_name: Scene key of the object to grasp.
        grasp_z_offset: Height (m) added above the object center for the grasp.
        env_id: Index of the parallel env to read poses from.
        robot_scene_name: Scene key of the robot articulation.
        align_yaw_to_object: Rotate the grasp about the vertical to match the object's yaw.

    Returns:
        A 4x4 homogeneous transform (robot base frame) on the env's device.
    """
    # Frames: W = world, R = robot base, O = object. root_quat_w is (x, y, z, w).
    robot = env.scene[robot_scene_name]
    t_W_R = wp.to_torch(robot.data.root_pos_w)[env_id, :3]
    q_W_R_xyzw = wp.to_torch(robot.data.root_quat_w)[env_id, :4]
    t_W_O = wp.to_torch(env.scene[object_name].data.root_pos_w)[env_id, :3]
    q_W_O_xyzw = wp.to_torch(env.scene[object_name].data.root_quat_w)[env_id, :4]

    t_R_O, q_R_O_xyzw = world_pose_to_robot_frame(t_W_O, q_W_O_xyzw, t_W_R, q_W_R_xyzw)
    return top_down_grasp_matrix(t_R_O, q_R_O_xyzw, grasp_z_offset, align_yaw_to_object)
