# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for runtime cuRobo motion-planning helpers."""

import numpy as np
import torch
from types import SimpleNamespace
from unittest.mock import Mock

from curobo.types import JointState

from isaaclab_arena_curobo.motion_planner import _pose_inverse, _pose_multiply


def test_pose_inverse_round_trip():
    pose = np.array([0.3, -0.2, 0.7, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])

    np.testing.assert_allclose(
        _pose_multiply(pose, _pose_inverse(pose)),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        atol=1e-6,
    )


def test_tool_frame_offset_follows_wrist_rotation(monkeypatch):
    from isaaclab_arena_curobo import motion_planner

    # The USD palm is 2 cm along URDF tool X, with a 90-degree Y rotation.
    # Rotating the wrist 90 degrees about base Z must rotate that offset too.
    half_sqrt_two = np.sqrt(0.5)
    current_arena = np.array([1.02, 0, 0, 0, half_sqrt_two, 0, half_sqrt_two])
    target_arena = np.array([0, 1.02, 0, -0.5, 0.5, 0.5, 0.5])
    current_fk = SimpleNamespace(
        position=torch.tensor([[1.0, 0.0, 0.0]]),
        quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    path_fk = SimpleNamespace(
        position=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0], [half_sqrt_two, 0, 0, half_sqrt_two]]),
    )
    planner = motion_planner.CuroboPosePlanner.__new__(motion_planner.CuroboPosePlanner)
    planner._T_F_E = None
    planner._planner = SimpleNamespace(
        default_joint_state=JointState(position=torch.zeros(1, 1), joint_names=["joint"]),
        joint_names=["joint"],
        kinematics=SimpleNamespace(joint_names=["joint"]),
        tool_frames=["palm"],
        compute_kinematics=Mock(
            side_effect=[SimpleNamespace(tool_poses=current_fk), SimpleNamespace(tool_poses=path_fk)]
        ),
        plan_pose=Mock(
            return_value=SimpleNamespace(
                success=torch.tensor([True]),
                interpolated_trajectory=JointState(position=torch.zeros(2, 1), joint_names=["joint"]),
                interpolated_last_tstep=torch.tensor([1]),
            )
        ),
    )
    monkeypatch.setattr(motion_planner.GoalToolPose, "from_poses", lambda poses: poses)

    waypoints = planner.plan_pose_waypoints(
        torch.zeros(1), current_arena[:3], current_arena[3:], target_arena[:3], target_arena[3:]
    )

    goal = planner._planner.plan_pose.call_args.args[0]["palm"]
    np.testing.assert_allclose(goal.position.numpy(), [[0, 1, 0]], atol=1e-6)
    np.testing.assert_allclose(goal.quaternion.numpy(), [[half_sqrt_two, 0, 0, half_sqrt_two]], atol=1e-6)
    np.testing.assert_allclose(waypoints, [current_arena, target_arena], atol=1e-6)
