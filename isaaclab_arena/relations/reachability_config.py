# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase


@dataclass
class ReachabilityConfig:
    """Declarative tuning for the optional build-time IK-reachability check.

    Pure data forwarded to the extension that builds the check (cuRobo); core placement never reads it.
    """

    embodiment: EmbodimentBase | None = field(default=None, metadata={"allow_config_override": False})
    """Robot embodiment the grasps must be reachable by; the cuRobo check builds its IK solver from it."""

    grasp_z_offset_m: float = 0.02
    """Height above each object's root for the top-down grasp pose the check tests."""

    ik_position_threshold_m: float = 0.01
    """Max IK position error (m) for a grasp to count as reachable."""

    ik_rotation_threshold_rad: float = 0.2
    """Max IK rotation error (rad, geodesic angle between the solved and goal orientations) for a grasp to count as
    reachable."""

    require_collision_free: bool = True
    """If True, it's also collision-free with the robot itself."""

    include_background_obstacles: bool = True
    """If True, the IK collision world also holds the fixed background mesh (e.g. kitchen cabinets and walls),
    covered by world-axis-aligned cuboids on a voxel grid. Each layout loads only the cuboids within
    ``background_obstacle_radius_m`` of its robot base. Scenes without a background mesh load none."""

    background_voxel_size_m: float = 0.05
    """Voxel edge length (m) the background mesh is covered at; coarser means fewer cuboids and more inflation."""

    background_obstacle_radius_m: float = 1.2
    """Distance (m) from the robot base within which a background piece is loaded as an obstacle."""
