# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end CuroboIKSolver tests against the real cuRobo install, on GPU.

The other cuRobo tests mock the solver or bypass its ``__init__``, so they cannot catch breakage in cuRobo's own
solver build (e.g. its collision checker calling a warp API that the pinned warp-lang no longer has). These build
the real Droid solver sim-free, as ``ReachabilityValidator`` does, and solve one batch.
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


def _retract_ee_pose(solver):
    """End-effector pose of the solver's retract configuration, as a ``(1, 4, 4)`` transform in the robot base frame."""
    import isaaclab.utils.math as math_utils

    state = solver.ik_solver.fk(solver.ik_solver.get_retract_config().view(1, -1))
    q_B_E_xyzw = math_utils.convert_quat(state.ee_quaternion, to="xyzw")
    return math_utils.make_pose(state.ee_position, math_utils.matrix_from_quat(q_B_E_xyzw))


def test_warp_torch_namespace_forwards_to_top_level_warp():
    """``warp.torch`` exposes the top-level warp torch interop functions cuRobo calls."""
    import warp as wp

    from isaaclab_arena_curobo.utils.warp_compat import add_warp_torch_namespace

    add_warp_torch_namespace()
    add_warp_torch_namespace()  # idempotent
    assert wp.torch.device_from_torch is wp.device_from_torch


def test_solver_reaches_its_own_retract_pose(droid_solver):
    """IK on the forward-kinematics pose of the retract configuration converges."""
    from isaaclab_arena_curobo.utils.ik_solver_utils import solve_ik_feasibility

    result = solve_ik_feasibility(droid_solver, _retract_ee_pose(droid_solver))
    assert result.feasible.tolist() == [True]


def test_solver_rejects_a_pose_out_of_reach(droid_solver):
    """IK on a pose 5 m beyond the retract pose does not converge."""
    from isaaclab_arena_curobo.utils.ik_solver_utils import solve_ik_feasibility

    T_B_E = _retract_ee_pose(droid_solver)
    T_B_E[:, 0, 3] += 5.0
    result = solve_ik_feasibility(droid_solver, T_B_E)
    assert result.feasible.tolist() == [False]
