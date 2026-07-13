# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import tempfile
import torch
import types
import yaml
from pathlib import Path
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import warp as wp
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

# Top-down grasp orientation (gripper pointing -Z) expressed in the robot base frame, (w, x, y, z).
DOWN_FACING_QUAT_WXYZ = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)


def pose_from_pos_quat(pos_xyz: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Build a 4x4 homogeneous transform from a position and a (w, x, y, z) quaternion."""
    rot = math_utils.matrix_from_quat(quat_wxyz.unsqueeze(0))[0]
    return math_utils.make_pose(pos_xyz, rot)


def make_planner_cfg(
    embodiment: EmbodimentBase,
    debug_planner: bool = False,
):
    """Build a CuroboPlannerCfg from an embodiment's cuRobo description.

    Reads the robot identity and planner tuning (asset paths, link/joint names, gripper targets,
    approach/retreat/time-dilation) off ``embodiment.get_curobo_cfg()``, patches the bundled robot
    config with the downloaded URDF path and the gripper open/closed joint targets, and returns a
    config ready for ``CuroboPlanner``.

    Args:
        embodiment: Embodiment that must expose a CuroboEmbodimentCfg via ``get_curobo_cfg()``.
        debug_planner: Enable cuRobo planner debug output.
    """
    curobo_cfg = embodiment.get_curobo_cfg()
    assert curobo_cfg is not None, (
        f"Embodiment '{embodiment.name}' has no cuRobo config; set its `curobo_config` "
        "(CuroboEmbodimentCfg) before building a cuRobo planner."
    )

    # cuRobo reads real on-disk files, so pull the robot config + URDF from the Nucleus/S3 asset server.
    robot_cfg_path = retrieve_file_path(curobo_cfg.robot_cfg_template)
    robot_urdf_path = retrieve_file_path(curobo_cfg.robot_urdf)

    with open(robot_cfg_path) as f:
        robot_yaml = yaml.safe_load(f)
    robot_yaml["robot_cfg"]["kinematics"]["urdf_path"] = robot_urdf_path

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


def fix_planner_object_sync_frame(planner) -> None:
    """Replace the planner's object-pose sync so obstacle poses are expressed in the robot base frame.

    ``CuroboPlanner`` defaults to world-frame obstacle poses; this validation drives goals
    in the robot base frame, so the collision world must match. Re-bind the sync method to
    transform every rigid object into the robot frame before updating the planner's world model.
    """

    def _sync_robot_base_frame(self):
        # USD prim path per scene object, used as the key into cuRobo's collision world model.
        object_mappings = self._get_object_mappings()
        world_model = self.motion_gen.world_coll_checker.world_model
        rigid_objects = self.env.scene.rigid_objects
        # Robot base pose in world frame (warp arrays -> torch before indexing in this Newton build).
        # Naming follows the ASL transform convention: X_B_A maps a vector in frame A to frame B
        # (t_B = X_B_A @ t_A), and t_A_B_in_F is the A->B translation expressed in frame F.
        # Frames: W = world, R = robot base, O = object.
        t_W_R_in_W = wp.to_torch(self.robot.data.root_pos_w)[self.env_id, :3]
        q_W_R_wxyz = wp.to_torch(self.robot.data.root_quat_w)[self.env_id, :4]
        # World->robot rotation (R_R_W = R_W_R^T) for positions; inverse quaternion (q_R_W) for orientations.
        R_R_W = math_utils.matrix_from_quat(q_W_R_wxyz.unsqueeze(0))[0].T
        q_R_W_wxyz = math_utils.quat_inv(q_W_R_wxyz.unsqueeze(0))[0]
        updated_count = 0
        for object_name, object_path in object_mappings.items():
            # Skip objects flagged static in the planner config (e.g. the table) — they aren't re-synced.
            static_objects = getattr(self.config, "static_objects", [])
            if object_name in rigid_objects and not any(s in object_name.lower() for s in static_objects):
                obj = rigid_objects[object_name]
                # Object pose in world frame (warp -> torch).
                t_W_O_in_W = wp.to_torch(obj.data.root_pos_w)[self.env_id, :3]
                q_W_O_wxyz = wp.to_torch(obj.data.root_quat_w)[self.env_id, :4]
                # Re-express the object pose in the robot base frame so it matches the IK goal frame.
                t_R_O_in_R = (R_R_W @ (t_W_O_in_W - t_W_R_in_W).unsqueeze(-1)).squeeze(-1)
                q_R_O_wxyz = math_utils.quat_mul(q_R_W_wxyz.unsqueeze(0), q_W_O_wxyz.unsqueeze(0))[0]
                pos_c = self._to_curobo_device(t_R_O_in_R)
                quat_c = self._to_curobo_device(q_R_O_wxyz)
                # cuRobo's world-model writer wants a flat [x, y, z, qw, qx, qy, qz] list.
                pose_list = torch.cat((pos_c, quat_c)).tolist()
                # Update the CPU-side world model first; only push to the GPU collision checker if it took.
                if self._update_object_in_world_model(world_model, object_name, object_path, pose_list):
                    curobo_pose = self._make_pose(position=pos_c, quaternion=quat_c)
                    self.motion_gen.world_coll_checker.update_obstacle_pose(
                        object_path, curobo_pose, update_cpu_reference=True
                    )
                    updated_count += 1
        self.logger.debug(f"SYNC (robot-base frame): Updated {updated_count} object poses")
        # Block until the pose writes land so the next IK solve sees the synced world.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    planner._sync_object_poses_with_isaaclab = types.MethodType(_sync_robot_base_frame, planner)


def make_curobo_planner(
    env: ManagerBasedEnv,
    embodiment: EmbodimentBase,
    env_id: int = 0,
    robot_scene_name: str | None = None,
    debug_planner: bool = False,
):
    """Construct a CuroboPlanner for an embodiment, bound to the env's robot and the robot base frame.

    The robot identity and planner tuning come entirely from ``embodiment.get_curobo_cfg()`` (it
    errors if the embodiment has no cuRobo config), so this is robot-agnostic.

    Args:
        env: The (unwrapped) Isaac Lab env; must expose the robot articulation in its scene.
        embodiment: Embodiment whose CuroboEmbodimentCfg describes the robot.
        env_id: Index of the parallel env the planner reads object/robot poses from.
        robot_scene_name: Scene key of the robot articulation. Defaults to the embodiment's scene name.
        debug_planner: Enable cuRobo planner debug output.
    """
    if robot_scene_name is None:
        robot_scene_name = embodiment.get_embodiment_name_in_scene()
    planner_cfg = make_planner_cfg(embodiment, debug_planner=debug_planner)
    # cuRobo-Lab's MotionGen/collision world is single-env only for now.
    planner = CuroboPlanner(env=env, robot=env.scene[robot_scene_name], config=planner_cfg, env_id=env_id)
    fix_planner_object_sync_frame(planner)
    return planner


def top_down_grasp_pose_in_robot_frame(
    env: ManagerBasedEnv,
    object_name: str,
    grasp_z_offset: float = 0.02,
    env_id: int = 0,
    robot_scene_name: str = "robot",
) -> torch.Tensor:
    """Top-down grasp pose at an object's center, expressed in the robot base frame.

    Args:
        env: The (unwrapped) Isaac Lab env holding both the robot and the target object.
        object_name: Scene key of the object to grasp.
        grasp_z_offset: Height (m) added above the object center for the grasp.
        env_id: Index of the parallel env to read poses from.
        robot_scene_name: Scene key of the robot articulation.

    Returns:
        A 4x4 homogeneous transform (robot base frame) on the env's device.
    """
    # Naming follows the ASL transform convention (see fix_planner_object_sync_frame):
    # X_B_A maps a vector in frame A to B. Frames: W = world, R = robot base, O = object.
    robot = env.scene[robot_scene_name]
    t_W_R_in_W = wp.to_torch(robot.data.root_pos_w)[env_id, :3]
    q_W_R_wxyz = wp.to_torch(robot.data.root_quat_w)[env_id, :4]
    R_R_W = math_utils.matrix_from_quat(q_W_R_wxyz.unsqueeze(0))[0].T

    t_W_O_in_W = wp.to_torch(env.scene[object_name].data.root_pos_w)[env_id, :3]
    t_R_O_in_R = (R_R_W @ (t_W_O_in_W - t_W_R_in_W).unsqueeze(-1)).squeeze(-1).clone()
    t_R_O_in_R[2] += grasp_z_offset

    return pose_from_pos_quat(t_R_O_in_R, DOWN_FACING_QUAT_WXYZ.to(t_R_O_in_R.device))
