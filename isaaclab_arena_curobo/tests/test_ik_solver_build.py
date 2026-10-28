# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end CuroboIKSolver tests against the real cuRobo install, on GPU.

The other cuRobo tests mock the solver or bypass its ``__init__``, so they cannot catch breakage in cuRobo's own
solver build (e.g. the Droid robot yaml failing to load in cuRobo's config layout). These build the real Droid
solver sim-free, as ``ReachabilityValidator`` does, and solve one batch.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.curobo_deps


@pytest.fixture(scope="module")
def droid_solver():
    """Real cuRobo IK solver for the Droid embodiment, built once per module."""
    from isaaclab_arena_curobo.embodiment_curobo_registry import get_curobo_cfg_by_name
    from isaaclab_arena_curobo.ik_solver import CuroboIKSolver

    return CuroboIKSolver(get_curobo_cfg_by_name("droid"), num_seeds=8)


def _default_ee_pose(solver):
    """End-effector pose of the solver's default configuration, as a ``(1, 4, 4)`` transform in the robot base frame."""
    import isaaclab.utils.math as math_utils

    ik_solver = solver.ik_solver
    state = ik_solver.compute_kinematics(ik_solver.default_joint_state)
    T_B_E = state.tool_poses.get_link_pose(ik_solver.tool_frames[0])
    q_B_E_xyzw = math_utils.convert_quat(T_B_E.quaternion, to="xyzw")
    return math_utils.make_pose(T_B_E.position, math_utils.matrix_from_quat(q_B_E_xyzw))


def test_solver_reaches_its_own_default_pose(droid_solver):
    """IK on the forward-kinematics pose of the default configuration converges."""
    from isaaclab_arena_curobo.utils.ik_solver_utils import solve_ik_feasibility

    result = solve_ik_feasibility(droid_solver, _default_ee_pose(droid_solver))
    assert result.feasible.tolist() == [True]


def test_solver_rejects_a_pose_out_of_reach(droid_solver):
    """IK on a pose 5 m beyond the default pose does not converge."""
    from isaaclab_arena_curobo.utils.ik_solver_utils import solve_ik_feasibility

    T_B_E = _default_ee_pose(droid_solver)
    T_B_E[:, 0, 3] += 5.0
    result = solve_ik_feasibility(droid_solver, T_B_E)
    assert result.feasible.tolist() == [False]


def test_solved_configuration_forward_kinematics_to_the_hand_spheres(droid_solver):
    """The solved joint configuration poses one row of collision spheres, and the hand mask selects some of them."""
    from isaaclab_arena_curobo.utils.ik_solver_utils import (
        hand_sphere_mask,
        robot_collision_spheres,
        solve_ik_feasibility,
    )

    result = solve_ik_feasibility(droid_solver, _default_ee_pose(droid_solver))
    spheres = robot_collision_spheres(droid_solver, result.joint_positions)
    mask = hand_sphere_mask(droid_solver)

    assert spheres.shape == (1, mask.shape[0], 4)
    assert 0 < int(mask.sum()) < mask.shape[0]
