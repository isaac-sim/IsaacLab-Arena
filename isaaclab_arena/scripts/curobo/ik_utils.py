# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Standalone IK feasibility utilities that operate on a CuroboPlanner instance.

Kept outside the IsaacLab submodule so changes can be committed directly
to IsaacLab-Arena without forking the upstream planner.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as PoseUtils


def get_current_joint_config(planner) -> torch.Tensor:
    """Return the robot's current joint positions on the cuRobo device.

    Args:
        planner: ``CuroboPlanner`` instance.

    Returns:
        Joint position tensor of shape ``(1, dof)``.
    """
    return planner._get_current_joint_state_for_curobo().position.detach().clone()


def check_ik_feasibility(
    planner,
    target_pose: torch.Tensor,
    seed_config: torch.Tensor | None = None,
    position_threshold: float = 0.01,
    rotation_threshold: float = 0.1,
) -> tuple[bool, float, float, torch.Tensor | None]:
    """Check if a target end-effector pose is reachable via inverse kinematics.

    Uses cuRobo's IK solver for fast, collision-aware feasibility checking
    without full trajectory optimization.  The caller controls the success
    criteria via *position_threshold* and *rotation_threshold* so that
    false negatives from cuRobo's internal thresholds do not pollute the
    planning heuristic.

    Args:
        planner: ``CuroboPlanner`` instance (provides IK solver, device
            conversion, and pose helpers).
        target_pose: 4x4 homogeneous transform in robot base frame.
        seed_config: Optional joint config tensor to seed the solver
            (shape ``(dof,)``, ``(1, dof)``, or ``(1, 1, dof)``).
            Pass the solution from a previous call to evaluate
            sequential reachability.
        position_threshold: Maximum position error (m) to consider feasible.
        rotation_threshold: Maximum rotation error (rad, geodesic) to
            consider feasible.

    Returns:
        ``(is_feasible, position_error, rotation_error, joint_solution)``
        where *joint_solution* is the best joint config (shape ``(dof,)``)
        when feasible, or ``None`` otherwise.
    """
    target_pose_cuda = planner._to_curobo_device(target_pose)
    target_pos, target_rot = PoseUtils.unmake_pose(target_pose_cuda)
    goal_pose = planner._make_pose(
        position=target_pos,
        quaternion=PoseUtils.quat_from_matrix(target_rot),
    )

    ik_seed = None
    if seed_config is not None:
        ik_seed = planner._to_curobo_device(seed_config)
        while ik_seed.dim() < 3:
            ik_seed = ik_seed.unsqueeze(0)

    ik_result = planner.motion_gen.ik_solver.solve_single(
        goal_pose, seed_config=ik_seed,
    )

    pos_err = ik_result.position_error.view(-1)
    rot_err = ik_result.rotation_error.view(-1)
    best_idx = int(pos_err.argmin().item())

    best_pos_err = float(pos_err[best_idx].item())
    best_rot_err = float(rot_err[best_idx].item())

    feasible = best_pos_err < position_threshold and best_rot_err < rotation_threshold

    joint_solution = None
    if feasible:
        dof = ik_result.solution.shape[-1]
        joint_solution = ik_result.solution.view(-1, dof)[best_idx].detach().clone()

    planner.logger.debug(
        f'IK feasibility: feasible={feasible}, '
        f'pos_err={best_pos_err:.4f}m, rot_err={best_rot_err:.4f}rad'
    )
    return feasible, best_pos_err, best_rot_err, joint_solution
