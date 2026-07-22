# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Import-light cuRobo frame and grasp-pose helpers shared across the env-coupled and sim-free paths."""

from __future__ import annotations

import os
import torch

import isaaclab.utils.math as math_utils

from isaaclab_arena.utils.device import resolve_cuda_device
from isaaclab_arena.utils.pose import Pose

# cuRobo captures a CUDA graph on the IK solver's first solve and, by default, errors when a later
# solve changes the "goal type" (warmup runs a single-goal solve, then we issue a batched
# ``solve_batch``). On CUDA >= 12 cuRobo resets the graph instead of erroring when this opts in.
# ``setdefault`` keeps any explicit user override; setting it at import lands it before any solve.
os.environ.setdefault("CUROBO_TORCH_CUDA_GRAPH_RESET", "1")

# Top-down grasp orientation (gripper approach axis pointing -Z) in the robot base frame, (x, y, z, w).
DOWN_FACING_QUAT_XYZW = (0.0, 1.0, 0.0, 0.0)


def world_pose_to_robot_frame(
    pos_w: torch.Tensor,
    quat_w_xyzw: torch.Tensor,
    robot_base_pos_w: torch.Tensor,
    robot_base_quat_w_xyzw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-express a world-frame pose in the robot base frame.

    Frames: W = world, R = robot base, O = object. Returns ``(t_R_O, q_R_O_xyzw)``. All inputs are
    tensors; the robot-base inputs are the base pose in world frame.
    """
    R_R_W = math_utils.matrix_from_quat(robot_base_quat_w_xyzw.unsqueeze(0))[0].T
    q_R_W_xyzw = math_utils.quat_inv(robot_base_quat_w_xyzw.unsqueeze(0))[0]
    t_R_O = (R_R_W @ (pos_w - robot_base_pos_w).unsqueeze(-1)).squeeze(-1)
    q_R_O_xyzw = math_utils.quat_mul(q_R_W_xyzw.unsqueeze(0), quat_w_xyzw.unsqueeze(0))[0]
    return t_R_O, q_R_O_xyzw


def top_down_grasp_matrix(
    t_R_O: torch.Tensor,
    q_R_O_xyzw: torch.Tensor,
    grasp_z_offset: float = 0.02,
    align_yaw_to_object: bool = True,
) -> torch.Tensor:
    """Top-down grasp pose at an object, in the robot base frame, as a 4x4 transform.

    Takes the object's pose(in the robot base frame), lifts the target by ``grasp_z_offset``, and
    faces the gripper down; when ``align_yaw_to_object`` it also spins the grasp about the vertical axis to
    match the object's yaw, folded into [-pi/2, pi/2] to stay within the wrist joint limits. Device/dtype follow ``t_R_O``.
    """
    device = t_R_O.device
    t_R_O = t_R_O.clone()
    # uplift by z offset
    t_R_O[2] += grasp_z_offset

    R_down = math_utils.matrix_from_quat(
        torch.tensor(DOWN_FACING_QUAT_XYZW, dtype=torch.float32, device=device).unsqueeze(0)
    )[0]
    # Useful for asymmetric objects to align the grasp with the object's yaw
    if align_yaw_to_object:
        R_R_O = math_utils.matrix_from_quat(q_R_O_xyzw.unsqueeze(0))[0]
        yaw = torch.atan2(R_R_O[1, 0], R_R_O[0, 0])
        yaw = torch.remainder(yaw + torch.pi / 2, torch.pi) - torch.pi / 2
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        R_yaw = torch.eye(3, device=device, dtype=t_R_O.dtype)
        R_yaw[0, 0], R_yaw[0, 1] = cos_y, -sin_y
        R_yaw[1, 0], R_yaw[1, 1] = sin_y, cos_y
        R_grasp = R_yaw @ R_down
    else:
        R_grasp = R_down

    q_grasp_xyzw = math_utils.quat_from_matrix(R_grasp.unsqueeze(0))[0]
    pose = Pose(position_xyz=tuple(t_R_O.tolist()), rotation_xyzw=tuple(q_grasp_xyzw.tolist()))
    return pose.to_transform_matrix(device)


def top_down_grasp_pose_from_world_poses(
    object_pos_w: tuple[float, float, float],
    object_quat_w_xyzw: tuple[float, float, float, float],
    robot_base_pos_w: tuple[float, float, float],
    robot_base_quat_w_xyzw: tuple[float, float, float, float],
    grasp_z_offset: float = 0.02,
    align_yaw_to_object: bool = True,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Top-down grasp pose at an object's center, in the robot base frame, as a 4x4 transform."""
    dev = resolve_cuda_device(device)
    # Pure-math computation of the object pose in the robot base frame
    t_W_O = torch.tensor(object_pos_w, dtype=torch.float32, device=dev)
    q_W_O_xyzw = torch.tensor(object_quat_w_xyzw, dtype=torch.float32, device=dev)
    t_W_R = torch.tensor(robot_base_pos_w, dtype=torch.float32, device=dev)
    q_W_R_xyzw = torch.tensor(robot_base_quat_w_xyzw, dtype=torch.float32, device=dev)

    t_R_O, q_R_O_xyzw = world_pose_to_robot_frame(t_W_O, q_W_O_xyzw, t_W_R, q_W_R_xyzw)
    return top_down_grasp_matrix(t_R_O, q_R_O_xyzw, grasp_z_offset, align_yaw_to_object)
