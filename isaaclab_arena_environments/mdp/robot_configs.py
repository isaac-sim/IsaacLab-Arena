# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""Task-specific robot configurations for manipulation tasks."""

from __future__ import annotations

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

# ===========================================================================================
# Franka Panda robot configuration optimized for assembly tasks (peg insert, gear mesh, etc.).
# ===========================================================================================

FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()

# The Isaac 6.0 asset tree replaced FrankaEmika/panda_instanceable.usd with a
# restructured MuJoCo-converted asset that upstream FRANKA_PANDA_CFG has not
# been updated for, so the inherited usd_path 404s. Load the identical asset
# from the Isaac 5.0 tree instead.
# TODO(2026.07.16, Remove once isaaclab_assets points at a valid Franka USD)
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.spawn.usd_path = FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.spawn.usd_path.replace(
    "/Assets/Isaac/6.0/", "/Assets/Isaac/5.0/"
)

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
