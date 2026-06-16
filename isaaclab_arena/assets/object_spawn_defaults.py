# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Spawn/physics defaults for library objects (no USD / pxr imports)."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

# Predefined rigid body property configurations for assembly tasks
# High iteration count for precision tasks (peg/hole insertion)
RIGID_BODY_PROPS_HIGH_PRECISION = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    max_depenetration_velocity=5.0,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=3666.0,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=192,
    solver_velocity_iteration_count=1,
    max_contact_impulse=1e32,
)

# Standard iteration count for gear mesh tasks
RIGID_BODY_PROPS_MEDIUM_PRECISION = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    max_depenetration_velocity=5.0,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=3666.0,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=32,
    solver_velocity_iteration_count=32,
    max_contact_impulse=1e32,
)

# Initial state configuration for articulations without joints (e.g., rigid bodies treated as articulations).
# We explicitly set joint_pos and joint_vel to empty dicts to avoid the default pattern {".*": 0.0} in ArticulationCfg.InitialStateCfg,
# which would fail to match when there are no joints in the articulation.
EMPTY_ARTICULATION_INIT_STATE_CFG = ArticulationCfg.InitialStateCfg(
    joint_pos={},
    joint_vel={},
)
