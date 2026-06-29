# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Construct a CuroboPlanner for the Franka/Robotiq embodiment and build grasp poses for IK checks.

These helpers back the offline IK-reachability validation. They are deliberately
narrow: a single embodiment, a top-down grasp, and a robot-base-frame pose, which is all
the reachability oracle in ``ik_utils`` needs. Imports of ``isaaclab_mimic`` are deferred to
call time so this module can be imported before the SimulationApp starts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Top-down grasp orientation (gripper pointing -Z) expressed in the robot base frame, (w, x, y, z).
DOWN_FACING_QUAT_WXYZ = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)

# Repo root: this file is isaaclab_arena_curobo/curobo_planner_utils.py
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ROBOT_CFG_TEMPLATE = _REPO_ROOT / "assets_local" / "droid_fixed_mimic_joint" / "franka_robotiq_2f_85_zero_curobo.yml"
_ROBOT_URDF = _REPO_ROOT / "assets_local" / "droid_fixed_mimic_joint" / "urdf" / "franka_robotiq_2f_85_zero.urdf"


def pose_from_pos_quat(pos_xyz: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Build a 4x4 homogeneous transform from a position and a (w, x, y, z) quaternion."""
    rot = math_utils.matrix_from_quat(quat_wxyz.unsqueeze(0))[0]
    return math_utils.make_pose(pos_xyz, rot)


def make_planner_cfg(
    approach_distance: float = 0.04,
    retreat_distance: float = 0.06,
    time_dilation_factor: float = 0.6,
    debug_planner: bool = False,
):
    """Build a CuroboPlannerCfg for the Franka/Robotiq 2F-85 embodiment.

    Patches the bundled robot config with the local URDF path and the gripper open/closed
    joint targets, then returns a config ready to hand to ``CuroboPlanner``. The defaults
    match the dev/stark pick-and-place CLI.

    Args:
        approach_distance: cuRobo approach distance (m).
        retreat_distance: cuRobo retreat distance (m).
        time_dilation_factor: cuRobo time dilation factor.
        debug_planner: Enable cuRobo planner debug output.
    """
    from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

    assert _ROBOT_CFG_TEMPLATE.exists(), f"CuRobo robot config not found: {_ROBOT_CFG_TEMPLATE}"
    assert _ROBOT_URDF.exists(), f"CuRobo URDF not found: {_ROBOT_URDF}"

    with _ROBOT_CFG_TEMPLATE.open("r") as f:
        robot_yaml = yaml.safe_load(f)
    robot_yaml["robot_cfg"]["kinematics"]["urdf_path"] = str(_ROBOT_URDF)

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
    import types

    def _sync_robot_base_frame(self):
        object_mappings = self._get_object_mappings()
        world_model = self.motion_gen.world_coll_checker.world_model
        rigid_objects = self.env.scene.rigid_objects
        robot_pos_w = self.robot.data.root_pos_w[self.env_id, :3]
        robot_quat_w = self.robot.data.root_quat_w[self.env_id, :4]
        r_w2r = math_utils.matrix_from_quat(robot_quat_w.unsqueeze(0))[0].T
        robot_quat_inv = math_utils.quat_inv(robot_quat_w.unsqueeze(0))[0]
        updated_count = 0
        for object_name, object_path in object_mappings.items():
            static_objects = getattr(self.config, "static_objects", [])
            if object_name in rigid_objects and not any(s in object_name.lower() for s in static_objects):
                obj = rigid_objects[object_name]
                obj_pos_w = obj.data.root_pos_w[self.env_id, :3]
                obj_quat_w = obj.data.root_quat_w[self.env_id, :4]
                pos_robot = (r_w2r @ (obj_pos_w - robot_pos_w).unsqueeze(-1)).squeeze(-1)
                quat_robot = math_utils.quat_mul(robot_quat_inv.unsqueeze(0), obj_quat_w.unsqueeze(0))[0]
                pos_c = self._to_curobo_device(pos_robot)
                quat_c = self._to_curobo_device(quat_robot)
                pose_list = [
                    float(pos_c[0]),
                    float(pos_c[1]),
                    float(pos_c[2]),
                    float(quat_c[0]),
                    float(quat_c[1]),
                    float(quat_c[2]),
                    float(quat_c[3]),
                ]
                if self._update_object_in_world_model(world_model, object_name, object_path, pose_list):
                    curobo_pose = self._make_pose(position=pos_c, quaternion=quat_c)
                    self.motion_gen.world_coll_checker.update_obstacle_pose(
                        object_path, curobo_pose, update_cpu_reference=True
                    )
                    updated_count += 1
        self.logger.debug(f"SYNC (robot-base frame): Updated {updated_count} object poses")
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    planner._sync_object_poses_with_isaaclab = types.MethodType(_sync_robot_base_frame, planner)


def make_curobo_planner(
    env: ManagerBasedEnv,
    env_id: int = 0,
    approach_distance: float = 0.04,
    retreat_distance: float = 0.06,
    time_dilation_factor: float = 0.6,
    debug_planner: bool = False,
):
    """Construct a CuroboPlanner bound to the env's ``robot`` and patched to the robot base frame.

    Args:
        env: The (unwrapped) Isaac Lab env; must expose a ``robot`` articulation in its scene.
        env_id: Index of the parallel env the planner reads object/robot poses from.
        approach_distance: cuRobo approach distance (m).
        retreat_distance: cuRobo retreat distance (m).
        time_dilation_factor: cuRobo time dilation factor.
        debug_planner: Enable cuRobo planner debug output.
    """
    from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner

    planner_cfg = make_planner_cfg(
        approach_distance=approach_distance,
        retreat_distance=retreat_distance,
        time_dilation_factor=time_dilation_factor,
        debug_planner=debug_planner,
    )
    planner = CuroboPlanner(env=env, robot=env.scene["robot"], config=planner_cfg, env_id=env_id)
    fix_planner_object_sync_frame(planner)
    return planner


def top_down_grasp_pose_in_robot_frame(
    env: ManagerBasedEnv,
    object_name: str,
    grasp_z_offset: float = 0.02,
    env_id: int = 0,
) -> torch.Tensor:
    """Top-down grasp pose at an object's center, expressed in the robot base frame.

    Reads the object's world pose, transforms its position into the robot base frame, lifts it
    by ``grasp_z_offset``, and pairs it with a fixed downward-facing orientation.

    Args:
        env: The (unwrapped) Isaac Lab env holding both ``robot`` and the target object.
        object_name: Scene key of the object to grasp.
        grasp_z_offset: Height (m) added above the object center for the grasp.
        env_id: Index of the parallel env to read poses from.

    Returns:
        A 4x4 homogeneous transform (robot base frame) on the env's device.
    """
    robot = env.scene["robot"]
    robot_pos_w = robot.data.root_pos_w[env_id, :3]
    robot_quat_w = robot.data.root_quat_w[env_id, :4]
    r_w2r = math_utils.matrix_from_quat(robot_quat_w.unsqueeze(0))[0].T

    obj_pos_w = env.scene[object_name].data.root_pos_w[env_id, :3]
    pos_robot = (r_w2r @ (obj_pos_w - robot_pos_w).unsqueeze(-1)).squeeze(-1).clone()
    pos_robot[2] += grasp_z_offset

    return pose_from_pos_quat(pos_robot, DOWN_FACING_QUAT_WXYZ.to(pos_robot.device))
