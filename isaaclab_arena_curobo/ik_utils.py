# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone IK feasibility utilities that operate on a CuroboPlanner instance.

Kept outside the IsaacLab submodule so changes can be committed directly
to IsaacLab-Arena without forking the upstream planner.
"""

from __future__ import annotations

import os
import torch

import isaaclab.utils.math as math_utils

# cuRobo captures a CUDA graph on the IK solver's first solve and, by default, errors when a later
# solve changes the "goal type". In this use case, the planner warms up with a single-goal
# solve, then we issue a batched ``solve_batch``. On CUDA >= 12 cuRobo can reset the graph instead of
# erroring, but only when this env var opts in. ``setdefault`` keeps any explicit user override.
os.environ.setdefault("CUROBO_TORCH_CUDA_GRAPH_RESET", "1")


def check_ik_feasibility_batch_goal_poses(
    planner,
    target_poses: torch.Tensor,
    seed_config: torch.Tensor | None = None,
    position_threshold: float = 0.01,
    rotation_threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solves all *target_poses* IK reaching in a single cuRobo ``solve_batch`` call for one layout.

    Args:
        planner: ``CuroboPlanner`` instance.
        target_poses: Target end-effector poses as a ``(b, 4, 4)`` batch of homogeneous
            transforms in the robot base frame.
        seed_config: Optional joint config tensor to seed the solver.
        position_threshold: Maximum position error (m) to consider feasible.
        rotation_threshold: Maximum rotation error (rad) to consider feasible.

    Returns:
        ``(feasible, position_error, rotation_error)``, each a length-``b`` tensor aligned with
        *target_poses*; *feasible* is boolean, the errors are the best-seed errors per pose.
    """
    target_poses = planner._to_curobo_device(target_poses)
    positions, rotations = math_utils.unmake_pose(target_poses)
    goal_pose = planner._make_pose(
        position=positions,
        # in xyzw
        quaternion=math_utils.quat_from_matrix(rotations),
        quat_is_xyzw=True,
    )

    ik_seed = None
    if seed_config is not None:
        ik_seed = planner._to_curobo_device(seed_config)
        while ik_seed.dim() < 3:
            ik_seed = ik_seed.unsqueeze(0)

    ik_result = planner.motion_gen.ik_solver.solve_batch(goal_pose, seed_config=ik_seed)

    num_poses = positions.shape[0]
    pos_err = ik_result.position_error.view(num_poses, -1)
    rot_err = ik_result.rotation_error.view(num_poses, -1)
    best_idx = pos_err.argmin(dim=1, keepdim=True)
    best_pos_err = pos_err.gather(1, best_idx).squeeze(1)
    best_rot_err = rot_err.gather(1, best_idx).squeeze(1)
    # Pose-residual only; see the module note on why success is not gated here.
    feasible = (best_pos_err < position_threshold) & (best_rot_err < rotation_threshold)

    planner.logger.debug(f"Batch IK feasibility: {int(feasible.sum().item())}/{num_poses} feasible")
    return feasible, best_pos_err, best_rot_err
