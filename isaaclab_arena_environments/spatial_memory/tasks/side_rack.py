# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The two-tier rack beside the robot, and open boxes (bins, trays).

The rack stands to the robot's left (+y) with six slots, three per tier. The Franka looks at it with the wrist camera
facing it, then turns to the counter in front, so the rack is out of view while the robot works at the counter.
"""

from __future__ import annotations

import math

from isaaclab_arena_environments.spatial_memory.franka_controller import top_down_quat
from isaaclab_arena_environments.spatial_memory.layout_sampling import pose
from isaaclab_arena_environments.spatial_memory.parts import PartSpec, board

TIER_SIZE_M = (0.44, 0.16, 0.04)
"""Extents of each tier board."""

RISER_HEIGHT_M = 0.30
"""Height of the block that carries the high tier."""

SLOT_XS_M = (0.02, 0.145, 0.27)
"""Environment-frame x of the three slots per tier, in the part of the Franka's top-down workspace beside it where the
scripted reach is fast and keeps the rack out of view, at a spacing that clears the hand."""

TIER_YS_M = (0.48, 0.64)
"""Environment-frame y of the center lines of the low and the high tier."""

RACK_CENTER_X_M = 0.145
"""Environment-frame x of the center of both tiers."""

LOOK_POSITION_E = (0.145, 0.40, 0.62)
"""Tool center point position from which the whole rack is in view before the robot turns to the counter."""

RACK_EDGE_Y_M = 0.24
"""Environment-frame y, just in front of the low tier, from which the final approach to a slot starts."""

COUNTER_WAIT_Z_M = 0.25
"""Tool center point height while the robot waits above the counter for an instruction."""

LOOK_JOINT_POSE = (1.45, -0.6, 0.0, -2.0, 0.0, 1.45, -2.356)
"""Franka arm spawn posture: turned towards the rack with the wrist camera looking down at it."""

REST_JOINT_POSE = (0.6, 0.2, 0.0, -2.0, 0.0, 2.2, -2.356)
"""Franka arm rest posture for the controller's null space; joint 7 is turned half a turn from Arena's ready posture so
the wrist camera trails on the robot side."""

BIN_BASE_M = (0.105, 0.13, 0.01)
"""Extents of a bin's base board."""

BIN_WALL_HEIGHT_M = 0.03
"""Height of a bin's walls above its base."""

BOX_WALL_THICKNESS_M = 0.006
"""Thickness of the walls of bins and trays."""

OPEN_BOX_SIDES = ("base", "front", "back", "left", "right")
"""Boards of an open box, in the order ``open_box_part_specs`` creates them."""


def rack_part_specs() -> list[PartSpec]:
    """Return the boards of the rack: the low tier, the high tier and the riser under the high tier."""
    return [
        board("tier_low", TIER_SIZE_M, color="wood"),
        board("tier_high", TIER_SIZE_M, color="wood"),
        board("riser", (TIER_SIZE_M[0], TIER_SIZE_M[1], RISER_HEIGHT_M), color="dark"),
    ]


def place_rack(poses: dict[str, list[float]], low_z: float, high_z: float) -> None:
    """Set the rack's board poses for tier top faces at ``low_z`` and ``high_z``."""
    poses["tier_low"] = pose(RACK_CENTER_X_M, TIER_YS_M[0], low_z - 0.5 * TIER_SIZE_M[2])
    poses["tier_high"] = pose(RACK_CENTER_X_M, TIER_YS_M[1], high_z - 0.5 * TIER_SIZE_M[2])
    poses["riser"] = pose(RACK_CENTER_X_M, TIER_YS_M[1], high_z - TIER_SIZE_M[2] - 0.5 * RISER_HEIGHT_M)


def rack_slots(low_z: float, high_z: float) -> list[tuple[float, float, float]]:
    """Return the (x, y, tier top z) of the six rack slots, low tier first."""
    slots = []
    for tier_y, tier_z in zip(TIER_YS_M, (low_z, high_z)):
        for slot_x in SLOT_XS_M:
            slots.append((slot_x, tier_y, tier_z))
    return slots


def open_box_names(prefix: str) -> list[str]:
    """Return the board names of the open box ``prefix``."""
    return [f"{prefix}_{side}" for side in OPEN_BOX_SIDES]


def open_box_part_specs(
    prefix: str, base_size_m: tuple[float, float, float], wall_height_m: float, base_color: str, wall_color: str
) -> list[PartSpec]:
    """Return the boards of an open box: a base and four walls standing on its edges."""
    base, front, back, left, right = open_box_names(prefix)
    long_wall = (base_size_m[0], BOX_WALL_THICKNESS_M, wall_height_m)
    short_wall = (BOX_WALL_THICKNESS_M, base_size_m[1], wall_height_m)
    return [
        board(base, base_size_m, color=base_color),
        board(front, long_wall, color=wall_color),
        board(back, long_wall, color=wall_color),
        board(left, short_wall, color=wall_color),
        board(right, short_wall, color=wall_color),
    ]


def place_open_box(
    poses: dict[str, list[float]],
    prefix: str,
    center_xy: tuple[float, float],
    base_size_m: tuple[float, float, float],
    wall_height_m: float,
    base_top_z: float,
) -> None:
    """Set the board poses of the open box ``prefix`` with its base's top face at ``base_top_z``."""
    center_x, center_y = center_xy
    base, front, back, left, right = open_box_names(prefix)
    wall_z = base_top_z + 0.5 * wall_height_m
    half_x = 0.5 * base_size_m[0] - 0.5 * BOX_WALL_THICKNESS_M
    half_y = 0.5 * base_size_m[1] - 0.5 * BOX_WALL_THICKNESS_M
    poses[base] = pose(center_x, center_y, base_top_z - 0.5 * base_size_m[2])
    poses[front] = pose(center_x, center_y + half_y, wall_z)
    poses[back] = pose(center_x, center_y - half_y, wall_z)
    poses[left] = pose(center_x + half_x, center_y, wall_z)
    poses[right] = pose(center_x - half_x, center_y, wall_z)


def top_down_quat_camera_towards_base(x: float, y: float) -> tuple[float, float, float, float]:
    """Return a top-down tool orientation at (x, y) with the wrist camera on the side towards the robot base.

    The rack is radially outward from the base, so the camera trails it, and this orientation family is the one the
    arm keeps naturally as it turns.
    """
    return top_down_quat(math.atan2(-y, -x))
