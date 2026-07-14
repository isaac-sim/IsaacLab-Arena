# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-embodiment cuRobo description, owned by the cuRobo extension (not core)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CuroboEmbodimentCfg:
    """Per-embodiment inputs cuRobo needs to plan for this robot."""

    robot_cfg_template: str
    """Nucleus/S3 path to the cuRobo robot ``.yml`` (its ``urdf_path`` is patched in at runtime)."""

    robot_urdf: str
    """Nucleus/S3 path to the robot URDF cuRobo loads for kinematics."""

    robot_name: str
    """cuRobo ``robot_name`` (a hardware identifier, not the Arena scene key)."""

    ee_link_name: str
    """Link cuRobo treats as the end-effector / IK goal frame."""

    gripper_joint_names: list[str]
    """Gripper joint names cuRobo actuates."""

    gripper_open_joint_pos: dict[str, float]
    """Gripper joint targets for the open pose, merged over the robot config's locked joints."""

    gripper_closed_joint_pos: dict[str, float]
    """Gripper joint targets for the closed pose, merged over the robot config's locked joints."""

    hand_link_names: list[str]
    """Links forming the hand, excluded from self-collision against the grasped object."""

    grasp_gripper_open_val: float = 10.0
    """cuRobo grasp-approach gripper opening value."""

    approach_distance: float = 0.04
    """Straight-line distance (m) cuRobo approaches the grasp along before closing."""

    retreat_distance: float = 0.06
    """Straight-line distance (m) cuRobo retreats after grasping."""

    time_dilation_factor: float = 0.6
    """Scales the planned trajectory speed (1.0 = full speed)."""

    collision_activation_distance: float = 0.05
    """Distance (m) at which cuRobo starts penalizing collisions."""

    trajopt_tsteps: int = 42
    """Number of trajectory-optimization time steps."""

    world_ignore_substrings: list[str] | None = field(default=None)
    """Substrings of obstacle names cuRobo should ignore in its collision world."""
