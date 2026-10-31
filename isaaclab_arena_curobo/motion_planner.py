# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime cuRobo v2 pose planning for Arena robot-control recipes."""

from __future__ import annotations

import numpy as np
import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState
from curobo.types import Pose as CuroboPose

from isaaclab_arena_curobo.curobo_embodiment_cfg import CuroboEmbodimentCfg
from isaaclab_arena_curobo.utils.robot_cfg_utils import load_patched_robot_yaml


class CuroboPosePlanner:
    """Plan joint trajectories and return their end-effector poses."""

    def __init__(self, cfg: CuroboEmbodimentCfg, device: str | torch.device) -> None:
        robot_cfg = load_patched_robot_yaml(cfg)["robot_cfg"]
        planner_cfg = MotionPlannerCfg.create(
            robot=robot_cfg,
            collision_cache={"cuboid": 64},
            device_cfg=_device_cfg(device),
            interpolation_dt=0.025,
            position_tolerance=0.01,
            orientation_tolerance=0.1,
        )
        self._planner = MotionPlanner(planner_cfg)
        self._T_F_E = None
        self._planner.warmup()

    @property
    def joint_names(self) -> list[str]:
        """Return active joints in planner order."""
        return list(self._planner.joint_names)

    def plan_pose_waypoints(
        self,
        joint_positions: torch.Tensor,
        current_position: np.ndarray,
        current_quaternion_xyzw: np.ndarray,
        target_position: np.ndarray,
        target_quaternion_xyzw: np.ndarray,
        max_attempts: int = 5,
    ) -> np.ndarray:
        """Plan to a root-frame pose and return dense root-frame XYZW pose waypoints."""
        device = self._planner.default_joint_state.position.device
        current = JointState(
            position=joint_positions.to(device=device, dtype=torch.float32).unsqueeze(0),
            joint_names=self.joint_names,
        )
        if self._T_F_E is None:
            active_current = current.reorder(list(self._planner.kinematics.joint_names))
            current_fk = self._planner.compute_kinematics(active_current).tool_poses
            assert current_fk is not None, "cuRobo forward kinematics returned no current tool pose"
            curobo_current = np.concatenate((
                current_fk.position.reshape(-1, 3)[0].detach().cpu().numpy(),
                torch.roll(current_fk.quaternion.reshape(-1, 4)[0], shifts=-1).detach().cpu().numpy(),
            ))
            sim_current = np.concatenate((current_position, current_quaternion_xyzw))
            # Both poses use the robot base frame. F is the URDF tool frame;
            # E is the USD end-effector frame controlled by Arena. Their fixed
            # offset belongs on the right, so it rotates with the wrist.
            self._T_F_E = _pose_multiply(_pose_inverse(curobo_current), sim_current)
        T_F_E = self._T_F_E
        curobo_target = _pose_multiply(np.concatenate((target_position, target_quaternion_xyzw)), _pose_inverse(T_F_E))
        position = torch.as_tensor(curobo_target[:3], device=device, dtype=torch.float32).reshape(1, 3)
        quaternion_xyzw = torch.as_tensor(curobo_target[3:], device=device, dtype=torch.float32).reshape(1, 4)
        pose = CuroboPose(position=position, quaternion=torch.roll(quaternion_xyzw, shifts=1, dims=-1))
        goal = GoalToolPose.from_poses({self._planner.tool_frames[0]: pose})
        result = self._planner.plan_pose(goal, current, max_attempts=max_attempts)
        assert result is not None and bool(result.success.item()), "cuRobo failed to find a pose trajectory"
        trajectory = result.interpolated_trajectory
        assert trajectory is not None, "cuRobo returned no interpolated trajectory"
        trajectory = trajectory.reorder(list(self._planner.kinematics.joint_names))
        last_tstep = int(result.interpolated_last_tstep.reshape(-1)[0].item())
        joint_trajectory = trajectory.position
        while joint_trajectory.ndim > 2:
            joint_trajectory = joint_trajectory[0]
        joint_trajectory = joint_trajectory[: last_tstep + 1]
        kinematics = self._planner.compute_kinematics(
            JointState(position=joint_trajectory, joint_names=list(self._planner.kinematics.joint_names))
        )
        tool_poses = kinematics.tool_poses
        assert tool_poses is not None, "cuRobo forward kinematics returned no tool poses"
        positions = tool_poses.position
        quaternions_wxyz = tool_poses.quaternion
        if positions.ndim == 3:
            positions = positions[:, 0]
            quaternions_wxyz = quaternions_wxyz[:, 0]
        quaternions_xyzw = torch.roll(quaternions_wxyz, shifts=-1, dims=-1)
        curobo_waypoints = torch.cat((positions, quaternions_xyzw), dim=-1).reshape(-1, 7).detach().cpu().numpy()
        return np.stack([_pose_multiply(waypoint, T_F_E) for waypoint in curobo_waypoints])


def _device_cfg(device: str | torch.device):
    from curobo.types import DeviceCfg

    resolved = torch.device(device)
    if resolved.type != "cuda":
        resolved = torch.device("cuda", torch.cuda.current_device())
    return DeviceCfg(device=resolved, dtype=torch.float32)


def _pose_multiply(T_A_B: np.ndarray, T_B_C: np.ndarray) -> np.ndarray:
    q_A_B = T_A_B[3:]
    q_B_C = T_B_C[3:]
    return np.concatenate((
        T_A_B[:3] + _rotate(q_A_B, T_B_C[:3]),
        _quat_multiply(q_A_B, q_B_C),
    ))


def _pose_inverse(T_A_B: np.ndarray) -> np.ndarray:
    q_B_A = np.concatenate((-T_A_B[3:6], T_A_B[6:]))
    return np.concatenate((_rotate(q_B_A, -T_A_B[:3]), q_B_A))


def _rotate(q_xyzw: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = q_xyzw[:3]
    return vector + 2.0 * np.cross(xyz, np.cross(xyz, vector) + q_xyzw[3] * vector)


def _quat_multiply(q_A_B: np.ndarray, q_B_C: np.ndarray) -> np.ndarray:
    xyz = q_A_B[3] * q_B_C[:3] + q_B_C[3] * q_A_B[:3] + np.cross(q_A_B[:3], q_B_C[:3])
    scalar = q_A_B[3] * q_B_C[3] - np.dot(q_A_B[:3], q_B_C[:3])
    quaternion = np.concatenate((xyz, [scalar]))
    return quaternion / np.linalg.norm(quaternion)
