# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch

from isaaclab_arena.utils.pose import (
    Pose,
    PosePerEnv,
    rotate_points_by_yaw,
    rotate_points_by_yaw_batch,
    rotate_quat_by_yaw,
    wrap_angle_to_pi,
)


def _yaw_of(quat_xyzw: tuple[float, float, float, float]) -> float:
    return 2.0 * math.atan2(quat_xyzw[2], quat_xyzw[3])


def test_rotate_quat_by_yaw_about_identity():
    """Yawing the identity gives a pure-Z quaternion with that yaw."""
    q = rotate_quat_by_yaw((0.0, 0.0, 0.0, 1.0), math.pi / 2)
    assert abs(q[0]) < 1e-6 and abs(q[1]) < 1e-6
    assert abs(wrap_angle_to_pi(_yaw_of(q) - math.pi / 2)) < 1e-6


def test_rotate_quat_by_yaw_composes_and_wraps():
    """Yaw composes additively about Z, and out-of-range angles wrap to [-pi, pi)."""
    base = rotate_quat_by_yaw((0.0, 0.0, 0.0, 1.0), math.pi / 6)
    composed = rotate_quat_by_yaw(base, math.pi / 3)
    assert abs(wrap_angle_to_pi(_yaw_of(composed) - math.pi / 2)) < 1e-6

    # yaw == 0 (and full turns) return the base unchanged.
    assert rotate_quat_by_yaw(base, 0.0) == base
    assert rotate_quat_by_yaw(base, 2.0 * math.pi) == base


def test_pose_composition():
    T_B_A = Pose(position_xyz=(1.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    T_C_B = Pose(position_xyz=(2.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))

    T_C_A = T_C_B.multiply(T_B_A)

    assert T_C_A.position_xyz == (3.0, 0.0, 0.0)
    assert T_C_A.rotation_xyzw == (0.0, 0.0, 0.0, 1.0)


def test_rotate_points_by_yaw_batch_matches_scalar():
    """Batch rotation with per-element yaws produces the same result as scalar rotation per row."""
    points = torch.tensor([[1.0, 2.0, 0.5], [3.0, -1.0, 1.0], [0.0, 4.0, -0.3]])
    yaws = torch.tensor([0.7, -1.2, 2.1])
    batch_result = rotate_points_by_yaw_batch(points, yaws)
    for i in range(len(yaws)):
        scalar_result = rotate_points_by_yaw(points[i : i + 1], yaws[i].item())
        assert torch.allclose(batch_result[i], scalar_result.squeeze(0), atol=1e-6)


def test_pose_per_env_stores_poses():
    """Test that PosePerEnv stores the list of Pose objects correctly."""
    poses = [
        Pose(position_xyz=(1.0, 2.0, 3.0)),
        Pose(position_xyz=(4.0, 5.0, 6.0)),
        Pose(position_xyz=(7.0, 8.0, 9.0)),
    ]
    pose_per_env = PosePerEnv(poses=poses)

    assert len(pose_per_env.poses) == 3
    assert pose_per_env.poses[0].position_xyz == (1.0, 2.0, 3.0)
    assert pose_per_env.poses[1].position_xyz == (4.0, 5.0, 6.0)
    assert pose_per_env.poses[2].position_xyz == (7.0, 8.0, 9.0)
