# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.devices.openxr import XrCfg

from isaaclab_arena.utils.pose import Pose


def get_default_xr_cfg(initial_pose: Pose | None = None, xr_offset: Pose | None = None) -> XrCfg:
    """
    Get the default XR configuration for the robot.
    Args:
        initial_pose: The initial pose of the robot.
        xr_offset: The offset of the XR device from the robot.

    Returns:
        The default XR configuration for the robot.
    """
    if initial_pose is None and xr_offset is None:
        raise ValueError("initial_pose or xr_offset must be provided")
    # If robot has an initial pose, compose it with the XR offset
    if initial_pose is not None:
        from isaaclab_arena.utils.pose import compose_poses

        # Compose robot pose with XR offset: T_world_xr = T_world_robot * T_robot_xr
        xr_pose_global = compose_poses(initial_pose, xr_offset)

        return XrCfg(
            anchor_pos=xr_pose_global.position_xyz,
            anchor_rot=xr_pose_global.rotation_wxyz,
        )
    else:
        # If no initial pose set, use the offset as global coordinates (robot at origin)
        return XrCfg(
            anchor_pos=xr_offset.position_xyz,
            anchor_rot=xr_offset.rotation_wxyz,
        )
