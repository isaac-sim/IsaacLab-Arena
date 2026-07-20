# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""Task-specific robot configurations for manipulation tasks."""

from __future__ import annotations

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

_LEGACY_FRANKA_PANDA_USD_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/Legacy/panda_instanceable.usd"

# ===========================================================================================
# Franka Panda robot configuration optimized for assembly tasks (peg insert, gear mesh, etc.).
# ===========================================================================================

FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
# Use the available Franka asset under Legacy instead of the unavailable path in the pinned Isaac Lab config.
# test_franka_assembly_asset_override_is_still_required flags future upstream path changes.
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.spawn.usd_path = _LEGACY_FRANKA_PANDA_USD_PATH

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
