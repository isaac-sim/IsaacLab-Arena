# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.utils.pose import Pose, PosePerEnv


def test_pose_composition():
    T_B_A = Pose(position_xyz=(1.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    T_C_B = Pose(position_xyz=(2.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))

    T_C_A = T_C_B.multiply(T_B_A)

    assert T_C_A.position_xyz == (3.0, 0.0, 0.0)
    assert T_C_A.rotation_xyzw == (0.0, 0.0, 0.0, 1.0)


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
