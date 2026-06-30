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
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Top-down grasp orientation (gripper pointing -Z) expressed in the robot base frame, (w, x, y, z).
DOWN_FACING_QUAT_WXYZ = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)

_ROBOT_ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/robot_library/droid/droid_fixed_mimic_joint"
_ROBOT_CFG_TEMPLATE = f"{_ROBOT_ASSET_DIR}/franka_robotiq_2f_85_zero_curobo.yml"
_ROBOT_URDF = f"{_ROBOT_ASSET_DIR}/urdf/franka_robotiq_2f_85_zero.urdf"


def pose_from_pos_quat(pos_xyz: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Build a 4x4 homogeneous transform from a position and a (w, x, y, z) quaternion."""
    rot = math_utils.matrix_from_quat(quat_wxyz.unsqueeze(0))[0]
    return math_utils.make_pose(pos_xyz, rot)


def make_planner_cfg_for_droid(
    approach_distance: float = 0.04,
    retreat_distance: float = 0.06,
    time_dilation_factor: float = 0.6,
    debug_planner: bool = False,
):
    """Build a CuroboPlannerCfg for the DROID embodiment.

    Patches the bundled robot config with the local URDF path and the gripper open/closed
    joint targets, then returns a config ready to hand to ``CuroboPlanner``. The defaults
    match the dev/stark pick-and-place CLI.

    Args:
        approach_distance: cuRobo approach distance (m).
        retreat_distance: cuRobo retreat distance (m).
        time_dilation_factor: cuRobo time dilation factor.
        debug_planner: Enable cuRobo planner debug output.
    """
    # cuRobo reads real on-disk files, so pull the robot config + URDF from the Nucleus/S3 asset server.
    robot_cfg_path = retrieve_file_path(_ROBOT_CFG_TEMPLATE)
    robot_urdf_path = retrieve_file_path(_ROBOT_URDF)

    with open(robot_cfg_path) as f:
        robot_yaml = yaml.safe_load(f)
    robot_yaml["robot_cfg"]["kinematics"]["urdf_path"] = robot_urdf_path

    tmp_dir = Path(tempfile.mkdtemp(prefix="curobo_robot_cfg_"))
    robot_cfg_file = tmp_dir / "droid_curobo_runtime.yml"
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
            "left_inner_finger",
            "left_inner_knuckle",
            "left_outer_finger",
            "left_outer_knuckle",
            "right_inner_finger",
            "right_inner_knuckle",
            "right_outer_finger",
            "right_outer_knuckle",
        ],
        grasp_gripper_open_val=10.0,
        approach_distance=approach_distance,
        retreat_distance=retreat_distance,
        time_dilation_factor=time_dilation_factor,
        collision_activation_distance=0.05,
        motion_step_size=None,
        trajopt_tsteps=42,
        visualize_plan=False,
        visualize_spheres=False,
        debug_planner=debug_planner,
        world_ignore_substrings=None,
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
        robot_pos_w = wp.to_torch(self.robot.data.root_pos_w)[self.env_id, :3]
        robot_quat_w = wp.to_torch(self.robot.data.root_quat_w)[self.env_id, :4]
        # World->robot rotation: transpose of the base rotation matrix; quat inverse for orientations.
        r_w2r = math_utils.matrix_from_quat(robot_quat_w.unsqueeze(0))[0].T
        robot_quat_inv = math_utils.quat_inv(robot_quat_w.unsqueeze(0))[0]
        updated_count = 0
        for object_name, object_path in object_mappings.items():
            # Skip objects flagged static in the planner config (e.g. the table) — they aren't re-synced.
            static_objects = getattr(self.config, "static_objects", [])
            if object_name in rigid_objects and not any(s in object_name.lower() for s in static_objects):
                obj = rigid_objects[object_name]
                # Object pose in world frame (warp -> torch).
                obj_pos_w = wp.to_torch(obj.data.root_pos_w)[self.env_id, :3]
                obj_quat_w = wp.to_torch(obj.data.root_quat_w)[self.env_id, :4]
                # Re-express the object pose in the robot base frame so it matches the IK goal frame.
                pos_robot = (r_w2r @ (obj_pos_w - robot_pos_w).unsqueeze(-1)).squeeze(-1)
                quat_robot = math_utils.quat_mul(robot_quat_inv.unsqueeze(0), obj_quat_w.unsqueeze(0))[0]
                pos_c = self._to_curobo_device(pos_robot)
                quat_c = self._to_curobo_device(quat_robot)
                # cuRobo's world-model writer wants a flat [x, y, z, qw, qx, qy, qz] list.
                pose_list = [
                    float(pos_c[0]),
                    float(pos_c[1]),
                    float(pos_c[2]),
                    float(quat_c[0]),
                    float(quat_c[1]),
                    float(quat_c[2]),
                    float(quat_c[3]),
                ]
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


def make_curobo_planner_for_droid(
    env: ManagerBasedEnv,
    env_id: int = 0,
    robot_scene_name: str = "robot",
    approach_distance: float = 0.04,
    retreat_distance: float = 0.06,
    time_dilation_factor: float = 0.6,
    debug_planner: bool = False,
):
    """Construct a DROID CuroboPlanner bound to the env's robot and patched to the robot base frame.

    Args:
        env: The (unwrapped) Isaac Lab env; must expose the robot articulation in its scene.
        env_id: Index of the parallel env the planner reads object/robot poses from.
        robot_scene_name: Scene key of the robot articulation (Arena's convention is ``"robot"``).
        approach_distance: cuRobo approach distance (m).
        retreat_distance: cuRobo retreat distance (m).
        time_dilation_factor: cuRobo time dilation factor.
        debug_planner: Enable cuRobo planner debug output.
    """
    planner_cfg = make_planner_cfg_for_droid(
        approach_distance=approach_distance,
        retreat_distance=retreat_distance,
        time_dilation_factor=time_dilation_factor,
        debug_planner=debug_planner,
    )
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
    robot = env.scene[robot_scene_name]
    robot_pos_w = wp.to_torch(robot.data.root_pos_w)[env_id, :3]
    robot_quat_w = wp.to_torch(robot.data.root_quat_w)[env_id, :4]
    r_w2r = math_utils.matrix_from_quat(robot_quat_w.unsqueeze(0))[0].T

    obj_pos_w = wp.to_torch(env.scene[object_name].data.root_pos_w)[env_id, :3]
    pos_robot = (r_w2r @ (obj_pos_w - robot_pos_w).unsqueeze(-1)).squeeze(-1).clone()
    pos_robot[2] += grasp_z_offset

    return pose_from_pos_quat(pos_robot, DOWN_FACING_QUAT_WXYZ.to(pos_robot.device))
