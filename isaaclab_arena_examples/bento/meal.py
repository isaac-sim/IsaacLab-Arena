# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Geometry and placement criteria for one four-compartment bento."""

from __future__ import annotations

import math
from dataclasses import dataclass

TABLE_TOP = 0.75
TRAY_FLOOR = TABLE_TOP + 0.012
HOVER_Z = TABLE_TOP + 0.33
TCP_OFFSET = (0.0, 0.0, -0.20)
EE_QUAT_XYZW = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
ROBOT_ASSET = "Robots/Yaskawa/Motoman Next/NHC12/NHC12_A00.usd"


@dataclass(frozen=True)
class Food:
    """Describe a rigid food portion and its assigned compartment in meters."""

    name: str
    size: tuple[float, float, float]
    color: tuple[float, float, float]
    source_xy: tuple[float, float]
    target_xy: tuple[float, float]
    bounds_xy: tuple[float, float, float, float]
    """Inner compartment bounds: xmin, xmax, ymin, ymax."""
    mass: float
    sphere: bool = False


FOODS = (
    Food(
        "rice",
        (0.164, 0.115, 0.043),
        (0.95, 0.94, 0.85),
        (0.42, -0.30),
        (0.47, 0.16),
        (0.376, 0.564, 0.086, 0.234),
        0.16,
    ),
    Food(
        "tamago",
        (0.135, 0.055, 0.035),
        (1.0, 0.72, 0.08),
        (0.60, -0.30),
        (0.47, 0.32),
        (0.376, 0.564, 0.246, 0.394),
        0.075,
    ),
    Food(
        "karaage",
        (0.10, 0.10, 0.066),
        (0.48, 0.19, 0.045),
        (0.69, -0.15),
        (0.67, 0.16),
        (0.576, 0.764, 0.086, 0.234),
        0.10,
    ),
    Food(
        "broccoli",
        (0.082, 0.082, 0.073),
        (0.12, 0.45, 0.08),
        (0.51, -0.15),
        (0.67, 0.32),
        (0.576, 0.764, 0.246, 0.394),
        0.025,
    ),
)


def placement_ok(
    food: Food,
    pose_xyzw: list[float],
    velocity: list[float],
    released: bool,
) -> bool:
    """Check full containment and rest near the tray floor after release.

    Args:
        food: Portion geometry and destination bounds.
        pose_xyzw: World position followed by an XYZW quaternion.
        velocity: World linear and angular velocities.
        released: Whether the controller has released this portion.

    Returns:
        Whether the portion fits its compartment and rests near its floor.
    """
    if not released or not all(math.isfinite(v) for v in (*pose_xyzw, *velocity)):
        return False
    px, py, pz, x, y, z, w = pose_xyzw
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-8:
        return False
    x, y, z, w = (v / norm for v in (x, y, z, w))
    rotation = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    extents = (
        [food.size[0] / 2] * 3
        if food.sphere
        else [sum(abs(row[j]) * food.size[j] / 2 for j in range(3)) for row in rotation]
    )
    xmin, xmax, ymin, ymax = food.bounds_xy
    return (
        xmin <= px - extents[0]
        and px + extents[0] <= xmax
        and ymin <= py - extents[1]
        and py + extents[1] <= ymax
        and abs(pz - extents[2] - TRAY_FLOOR) <= 0.008
        and math.sqrt(sum(v * v for v in velocity[:3])) < 0.02
        and math.sqrt(sum(v * v for v in velocity[3:])) < 0.1
    )
