# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Waypoint building blocks shared by the scripted Franka plans."""

from __future__ import annotations

import math

from isaaclab_arena_environments.spatial_memory.franka_controller import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    HOLD_FOREVER_STEPS,
    Waypoint,
    pitched_approach,
    pitched_quat,
    top_down_quat,
)

PITCH_45_RAD = math.radians(45.0)
"""Pitch below horizontal of the side grasps and insertions; a forward-level hand is out of reach near the counter."""

PREGRASP_SETBACK_M = 0.10
"""Distance back along the approach direction from which a side grasp moves in."""


def pitched_pick(
    t_E_item_xy: tuple[float, float], grasp_z: float, yaw_rad: float = 0.0, pitch_rad: float = PITCH_45_RAD
) -> list[Waypoint]:
    """Return waypoints that side-grasp an upright item with the hand pitched below horizontal.

    The fingers close horizontally, so the item stays upright. The plan ends with the gripper closed at the grasp.
    """
    q_E_tcp = pitched_quat(yaw_rad, pitch_rad)
    approach = pitched_approach(yaw_rad, pitch_rad)
    pregrasp = tuple(
        target - PREGRASP_SETBACK_M * direction
        for target, direction in zip((t_E_item_xy[0], t_E_item_xy[1], grasp_z), approach)
    )
    return [
        Waypoint(
            (pregrasp[0], pregrasp[1], pregrasp[2] + 0.08),
            q_E_tcp,
            label="above_pregrasp",
            tolerance_m=0.03,
            tolerance_rad=0.15,
        ),
        Waypoint(pregrasp, q_E_tcp, label="pregrasp", max_speed_mps=0.2),
        Waypoint(
            (t_E_item_xy[0], t_E_item_xy[1], grasp_z), q_E_tcp, label="grasp", max_speed_mps=0.08, tolerance_m=0.006
        ),
        Waypoint((t_E_item_xy[0], t_E_item_xy[1], grasp_z), q_E_tcp, GRIPPER_CLOSE, hold_steps=15, label="close"),
    ]


def top_down_pick(
    t_E_item_xy: tuple[float, float], grasp_z: float, camera_yaw_rad: float = math.pi, approach_height_m: float = 0.12
) -> list[Waypoint]:
    """Return waypoints that grasp an item from above; ``camera_yaw_rad`` sets where the wrist camera sits."""
    q_E_tcp = top_down_quat(camera_yaw_rad)
    return [
        Waypoint(
            (t_E_item_xy[0], t_E_item_xy[1], grasp_z + approach_height_m),
            q_E_tcp,
            label="above_item",
            tolerance_m=0.02,
            tolerance_rad=0.12,
        ),
        Waypoint(
            (t_E_item_xy[0], t_E_item_xy[1], grasp_z), q_E_tcp, label="grasp", max_speed_mps=0.1, tolerance_m=0.006
        ),
        Waypoint((t_E_item_xy[0], t_E_item_xy[1], grasp_z), q_E_tcp, GRIPPER_CLOSE, hold_steps=15, label="close"),
    ]


def release_and_hold(
    t_E_release: tuple[float, float, float],
    q_E_tcp: tuple[float, float, float, float],
    t_E_retreat: tuple[float, float, float],
) -> list[Waypoint]:
    """Return waypoints that open the gripper, retreat, and then hold still until the episode ends."""
    return [
        Waypoint(t_E_release, q_E_tcp, GRIPPER_OPEN, hold_steps=8, label="release"),
        Waypoint(t_E_retreat, q_E_tcp, label="retreat", max_speed_mps=0.12),
        Waypoint(t_E_retreat, q_E_tcp, label="done", max_steps=HOLD_FOREVER_STEPS),
    ]
