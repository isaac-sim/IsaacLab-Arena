# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Import-light cuRobo frame, config, and IK helpers shared across the env-coupled and sim-free paths."""

from __future__ import annotations

import os
import torch
import yaml
from typing import TYPE_CHECKING, Protocol

import isaaclab.utils.math as math_utils
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    import logging

    from curobo.types.math import Pose as CuroboPose
    from curobo.wrap.reacher.ik_solver import IKSolver

    from isaaclab_arena_curobo.curobo_embodiment_cfg import CuroboEmbodimentCfg


class IKSolverContext(Protocol):
    """The host that owns a cuRobo IK solver plus the device/pose plumbing to drive it."""

    logger: logging.Logger

    def _to_curobo_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a tensor onto the cuRobo device/dtype."""

    def _make_pose(self, position: torch.Tensor, quaternion: torch.Tensor, *, quat_is_xyzw: bool = True) -> CuroboPose:
        """Build a cuRobo ``Pose`` on the cuRobo device."""


def resolve_ik_solver(ik_context: IKSolverContext) -> IKSolver:
    """Get the cuRobo ``IKSolver`` from a host, whichever way it exposes it.

    ``SimFreeIKSolver`` holds it as ``ik_solver``; the upstream ``CuroboPlanner`` (not ours to change)
    holds it as ``motion_gen.ik_solver``. Centralizing the lookup lets the solve take just the host.
    """
    ik_solver = getattr(ik_context, "ik_solver", None)
    if ik_solver is None:
        ik_solver = ik_context.motion_gen.ik_solver
    return ik_solver


# cuRobo captures a CUDA graph on the IK solver's first solve and, by default, errors when a later
# solve changes the "goal type" (warmup runs a single-goal solve, then we issue a batched
# ``solve_batch``). On CUDA >= 12 cuRobo resets the graph instead of erroring when this opts in.
# ``setdefault`` keeps any explicit user override; setting it at import lands it before any solve.
os.environ.setdefault("CUROBO_TORCH_CUDA_GRAPH_RESET", "1")

# Top-down grasp orientation (gripper approach axis pointing -Z) in the robot base frame, (x, y, z, w).
DOWN_FACING_QUAT_XYZW = (0.0, 1.0, 0.0, 0.0)


def load_patched_robot_yaml(curobo_cfg: CuroboEmbodimentCfg) -> dict:
    """Load an embodiment's cuRobo robot yaml and splice in its downloaded URDF path."""
    robot_cfg_path = retrieve_file_path(curobo_cfg.robot_cfg_template)
    robot_urdf_path = retrieve_file_path(curobo_cfg.robot_urdf)
    with open(robot_cfg_path) as f:
        robot_yaml = yaml.safe_load(f)
    robot_yaml["robot_cfg"]["kinematics"]["urdf_path"] = robot_urdf_path
    return robot_yaml


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


def solve_ik_feasibility(
    ik_context: IKSolverContext,
    target_poses: torch.Tensor,
    seed_config: torch.Tensor | None = None,
    position_threshold: float = 0.01,
    rotation_threshold: float = 0.1,
    require_collision_free: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched IK feasibility of all ``target_poses`` against a cuRobo IK solver. Shared between sim-free and env-coupled paths.
    Runs a single ``solve_batch`` for one layout.

    Args:
        ik_context: The host that owns the solver and supplies device/pose plumbing -- a ``CuroboPlanner`` (env-coupled) or a ``SimFreeIKSolver``.
        target_poses: ``(b, 4, 4)`` end-effector goal transforms in the robot base frame.
        seed_config: Optional joint seed tensor.
        position_threshold: Max position error (m) to count as feasible.
        rotation_threshold: Max rotation error (rad) to count as feasible.
        require_collision_free: Also require a collision-free joint solution (cuRobo ``success``), not
            just pose convergence. The caller must first exclude the grasped object's own contact --
            e.g. disable the hand-link spheres -- or the gripper over its target reads as a collision.

    Returns:
        ``(feasible, position_error, rotation_error)``, each length ``b`` and aligned with the input;
        errors are the best-seed values per pose.
    """
    ik_solver = resolve_ik_solver(ik_context)
    target_poses = ik_context._to_curobo_device(target_poses)
    positions, rotations = math_utils.unmake_pose(target_poses)
    goal_pose = ik_context._make_pose(
        position=positions,
        quaternion=math_utils.quat_from_matrix(rotations),  # xyzw
        quat_is_xyzw=True,
    )

    ik_seed = None
    if seed_config is not None:
        ik_seed = ik_context._to_curobo_device(seed_config)
        while ik_seed.dim() < 3:
            ik_seed = ik_seed.unsqueeze(0)

    ik_result = ik_solver.solve_batch(goal_pose, seed_config=ik_seed)

    num_poses = positions.shape[0]
    pos_err = ik_result.position_error.view(num_poses, -1)
    rot_err = ik_result.rotation_error.view(num_poses, -1)

    ok = (pos_err < position_threshold) & (rot_err < rotation_threshold)
    # TODO(xinjieyao, 2026-07-15): Support collision-free IK by disabling the hand-link spheres during collision checking
    assert require_collision_free is False, "Collision-free IK is not supported yet. Needs extra machinery."
    if require_collision_free:
        # cuRobo folds collision-free-ness into ``success`` (success = converged AND feasible).
        ok = ok & ik_result.success.view(num_poses, -1).bool()
    feasible = ok.any(dim=1)

    # Best-seed errors, reported for logging/return only (not part of the accept decision).
    best_idx = pos_err.argmin(dim=1, keepdim=True)
    best_pos_err = pos_err.gather(1, best_idx).squeeze(1)
    best_rot_err = rot_err.gather(1, best_idx).squeeze(1)

    ik_context.logger.debug(f"Batch IK feasibility: {int(feasible.sum().item())}/{num_poses} feasible")
    return feasible, best_pos_err, best_rot_err
