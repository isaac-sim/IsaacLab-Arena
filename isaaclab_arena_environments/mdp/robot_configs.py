# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""Task-specific robot configurations for manipulation tasks."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG


def franka_gripper_joint_setter(
    joint_pos: torch.Tensor,
    row_indices: Sequence[int],
    finger_joint_indices: Sequence[int],
    width: float,
) -> None:
    """Set Franka Panda finger joints to achieve a given total opening *width*.

    Each finger joint is set to ``width / 2`` (symmetric grasp).
    """
    for jid in finger_joint_indices:
        joint_pos[row_indices, jid] = width / 2.0

# ===========================================================================================
# Franka Panda robot configuration optimized for assembly tasks (peg insert, gear mesh, etc.).
# ===========================================================================================

FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()

# Enable contact sensors for assembly tasks
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.spawn.activate_contact_sensors = True

# Disable gravity on the robot for better control stability
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True

# Increase stiffness and damping for precise positioning
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 150.0
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 30.0
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 150.0
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.actuators["panda_forearm"].damping = 30.0
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.actuators["panda_hand"].stiffness = 150.0
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.actuators["panda_hand"].damping = 30.0

# Set initial position for tabletop tasks
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.init_state.pos = (0.0, 0.0, 0.0)


_FACTORY_ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"

FRANKA_MIMIC_OSC_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{_FACTORY_ASSET_DIR}/franka_mimic.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint2": 0.04,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
    ),
    actuators={
        "panda_arm1": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
            effort_limit_sim=87,
            velocity_limit_sim=124.6,
        ),
        "panda_arm2": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
            effort_limit_sim=12,
            velocity_limit_sim=149.5,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint[1-2]"],
            effort_limit_sim=40.0,
            velocity_limit_sim=0.04,
            stiffness=7500.0,
            damping=173.0,
            friction=0.1,
            armature=0.0,
        ),
    },
)
