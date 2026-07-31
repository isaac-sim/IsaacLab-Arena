# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Parity constants and Droid-specific setup for Gear Assembly."""

from __future__ import annotations

import math
import torch
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from isaaclab_arena.utils.pose import Pose

DroidGearAssemblyEmbodiment = Literal["droid_abs_joint_pos", "droid_rel_joint_pos", "droid_differential_ik"]
GearAssemblyMode = Literal["play", "randomized"]
PhysicsBackend = Literal["newton", "physx"]

DROID_GEAR_ASSEMBLY_EMBODIMENTS: tuple[DroidGearAssemblyEmbodiment, ...] = (
    "droid_abs_joint_pos",
    "droid_rel_joint_pos",
    "droid_differential_ik",
)

GEAR_TYPES = ("gear_small", "gear_medium", "gear_large")
GEAR_OFFSETS = {
    "gear_small": [0.076125, 0.0, 0.0],
    "gear_medium": [0.030375, 0.0, 0.0],
    "gear_large": [-0.045375, 0.0, 0.0],
}
DROID_BASE_GEAR_POSE = Pose(position_xyz=(0.481, -0.073, 0.071), rotation_xyzw=(0.0, 0.0, 0.70711, -0.70711))
MAPLE_TABLE_TOP_Z = 0.003000684082508087
MAPLE_TABLE_POSE = Pose(position_xyz=(0.0, 0.0, DROID_BASE_GEAR_POSE.position_xyz[2] - MAPLE_TABLE_TOP_Z))
MAPLE_TABLE_TOP_COLLISION_SIZE = (0.7, 1.0)
MAPLE_TABLE_TOP_COLLISION_THICKNESS = 0.02
MAPLE_TABLE_TOP_COLLISION_POSE = Pose(
    position_xyz=(
        0.5485909044742584,
        0.02206302247941494,
        DROID_BASE_GEAR_POSE.position_xyz[2],
    )
)
GEAR_TABLETOP_PARKING_Z = MAPLE_TABLE_TOP_COLLISION_POSE.position_xyz[2] + 0.043
GEAR_TABLETOP_PARKING_POSITIONS = {
    "gear_small": (
        MAPLE_TABLE_TOP_COLLISION_POSE.position_xyz[0] - 0.20,
        MAPLE_TABLE_TOP_COLLISION_POSE.position_xyz[1] + 0.12,
        GEAR_TABLETOP_PARKING_Z,
    ),
    "gear_medium": (
        MAPLE_TABLE_TOP_COLLISION_POSE.position_xyz[0] + 0.20,
        MAPLE_TABLE_TOP_COLLISION_POSE.position_xyz[1] + 0.12,
        GEAR_TABLETOP_PARKING_Z,
    ),
    "gear_large": (
        MAPLE_TABLE_TOP_COLLISION_POSE.position_xyz[0],
        MAPLE_TABLE_TOP_COLLISION_POSE.position_xyz[1] + 0.32,
        GEAR_TABLETOP_PARKING_Z,
    ),
}
GEAR_TABLETOP_ORIENTATION_XYZW = (1.0, 0.0, 0.0, 0.0)
GEAR_INACTIVE_TABLE_PARKING_Z = GEAR_TABLETOP_PARKING_Z
GEAR_INACTIVE_PARKING_POSITIONS = GEAR_TABLETOP_PARKING_POSITIONS
GEAR_INACTIVE_TABLETOP_ORIENTATION_XYZW = GEAR_TABLETOP_ORIENTATION_XYZW

GEAR_POSE_RANGE = {
    "x": [-0.1, 0.1],
    "y": [-0.25, 0.25],
    "z": [0.0, 0.0],
    "roll": [0.0, 0.0],
    "pitch": [0.0, 0.0],
    "yaw": [-math.pi / 6, math.pi / 6],
}
SELECTED_GEAR_POS_RANGE = {
    "x": [-0.02, 0.02],
    "y": [-0.02, 0.02],
    "z": [0.0575, 0.0775],
}

DROID_ARM_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]


@dataclass(frozen=True)
class GearAssemblyRobotSpec:
    """Robot-specific Gear Assembly config consumed by the source manager terms."""

    name: str
    joint_names: list[str]
    num_arm_joints: int
    end_effector_body_name: str
    grasp_rot_offset: list[float]
    gear_offsets_grasp: dict[str, list[float]]
    hand_grasp_width: dict[str, float]
    hand_close_width: dict[str, float]
    gripper_joint_setter_func: Callable[[torch.Tensor, list[int], list[int], float], None]
    state_space: int
    observation_space: int
    startup_materials: dict[str, tuple[float, float, float]]
    reset_randomizes_robot: bool
    set_grasp_pos_randomization_range: dict[str, list[float]]


def get_droid_robot_spec() -> GearAssemblyRobotSpec:
    """Return the Gear Assembly spec for Arena's Droid Franka/Robotiq embodiment."""
    return GearAssemblyRobotSpec(
        name="droid",
        joint_names=list(DROID_ARM_JOINT_NAMES),
        num_arm_joints=7,
        end_effector_body_name="base_link",
        grasp_rot_offset=[math.sqrt(2.0) / 2.0, math.sqrt(2.0) / 2.0, 0.0, 0.0],
        gear_offsets_grasp={
            "gear_small": [0.0, GEAR_OFFSETS["gear_small"][0], -0.19],
            "gear_medium": [0.0, GEAR_OFFSETS["gear_medium"][0], -0.19],
            "gear_large": [0.0, GEAR_OFFSETS["gear_large"][0], -0.19],
        },
        hand_grasp_width={"gear_small": 0.64, "gear_medium": 0.46, "gear_large": 0.4},
        hand_close_width={"gear_small": 0.69, "gear_medium": 0.51, "gear_large": 0.45},
        gripper_joint_setter_func=_set_droid_gripper_joint_pos,
        state_space=28,
        observation_space=21,
        startup_materials={
            "factory_gear_small": (0.75, 0.75, 0.0),
            "factory_gear_medium": (0.75, 0.75, 0.0),
            "factory_gear_large": (0.75, 0.75, 0.0),
            "factory_gear_base": (0.75, 0.75, 0.0),
            "robot": (0.75, 0.75, 0.0),
        },
        reset_randomizes_robot=False,
        set_grasp_pos_randomization_range={
            "x": [-0.0, 0.0],
            "y": [-0.005, 0.005],
            "z": [-0.003, 0.003],
        },
    )


def gear_pose_for_mode(mode: GearAssemblyMode) -> Pose:
    """Return the construction gear/base pose for Droid Gear Assembly."""
    return DROID_BASE_GEAR_POSE


def _set_droid_gripper_joint_pos(
    joint_pos: torch.Tensor,
    reset_ind_joint_pos: list[int],
    gripper_joints: list[int],
    joint_position: float,
) -> None:
    """Set Droid's Robotiq gripper joints for the source reset term."""
    assert len(gripper_joints) >= 6, f"Droid gripper requires at least 6 gripper joints, got {len(gripper_joints)}"
    for idx in reset_ind_joint_pos:
        joint_pos[idx, gripper_joints[0]] = joint_position
        joint_pos[idx, gripper_joints[1]] = joint_position
        joint_pos[idx, gripper_joints[2]] = -joint_position
        joint_pos[idx, gripper_joints[3]] = joint_position
        joint_pos[idx, gripper_joints[4]] = -joint_position
        joint_pos[idx, gripper_joints[5]] = -joint_position
