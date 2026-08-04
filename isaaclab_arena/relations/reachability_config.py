# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase


@dataclass
class ReachabilityConfig:
    """Declarative tuning for the optional build-time IK-reachability check.

    Pure data forwarded to the extension that builds the check (cuRobo); core placement never reads it.
    """

    embodiment: EmbodimentBase | None = None
    """Robot embodiment the grasps must be reachable by; the cuRobo check builds its IK solver from it."""

    grasp_z_offset_m: float = 0.02
    """Height above each object's root for the top-down grasp pose the check tests."""

    ik_position_threshold_m: float = 0.01
    """Max IK position error (m) for a grasp to count as reachable."""

    ik_rotation_threshold_rad: float = 0.1
    """Max IK rotation error (rad) for a grasp to count as reachable."""

    require_collision_free: bool = True
    """If True, it's also collision-free with the robot itself."""
