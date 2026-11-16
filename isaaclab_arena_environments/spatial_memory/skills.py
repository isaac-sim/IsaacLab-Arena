# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Waypoint building blocks shared by the scripted Franka plans."""

from __future__ import annotations

import math

from isaaclab_arena_environments.spatial_memory.controller import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    Waypoint,
    pitched_approach,
    pitched_quat,
    top_down_quat,
)

PITCH_45_RAD = math.radians(45.0)


def pitched_pick(
    item_xy: tuple[float, float], grasp_z: float, yaw_rad: float = 0.0, pitch_rad: float = PITCH_45_RAD
) -> list[Waypoint]:
    """Return waypoints that side-grasp an upright item with the hand pitched below horizontal.

    The fingers close horizontally, so the item stays upright. The plan ends with the gripper closed at the grasp.
    """
    quat = pitched_quat(yaw_rad, pitch_rad)
    approach = pitched_approach(yaw_rad, pitch_rad)
    pregrasp = (item_xy[0] - 0.10 * approach[0], item_xy[1] - 0.10 * approach[1], grasp_z - 0.10 * approach[2])
    return [
        Waypoint(
            (pregrasp[0], pregrasp[1], pregrasp[2] + 0.08),
            quat,
            label="above_pregrasp",
            tolerance_m=0.03,
            tolerance_rad=0.15,
        ),
        Waypoint(pregrasp, quat, label="pregrasp", max_speed_mps=0.2),
        Waypoint((item_xy[0], item_xy[1], grasp_z), quat, label="grasp", max_speed_mps=0.08, tolerance_m=0.006),
        Waypoint((item_xy[0], item_xy[1], grasp_z), quat, GRIPPER_CLOSE, hold_steps=15, label="close"),
    ]


def top_down_pick(
    item_xy: tuple[float, float], grasp_z: float, camera_yaw_rad: float = math.pi, approach_height_m: float = 0.12
) -> list[Waypoint]:
    """Return waypoints that grasp an item from above; ``camera_yaw_rad`` sets where the wrist camera sits."""
    quat = top_down_quat(camera_yaw_rad)
    return [
        Waypoint(
            (item_xy[0], item_xy[1], grasp_z + approach_height_m),
            quat,
            label="above_item",
            tolerance_m=0.02,
            tolerance_rad=0.12,
        ),
        Waypoint((item_xy[0], item_xy[1], grasp_z), quat, label="grasp", max_speed_mps=0.1, tolerance_m=0.006),
        Waypoint((item_xy[0], item_xy[1], grasp_z), quat, GRIPPER_CLOSE, hold_steps=15, label="close"),
    ]


def release_and_hold(
    position_e: tuple[float, float, float],
    quat: tuple[float, float, float, float],
    retreat_e: tuple[float, float, float],
) -> list[Waypoint]:
    """Return waypoints that open the gripper, retreat, and then hold still until the episode ends."""
    return [
        Waypoint(position_e, quat, GRIPPER_OPEN, hold_steps=8, label="release"),
        Waypoint(retreat_e, quat, label="retreat", max_speed_mps=0.12),
        Waypoint(retreat_e, quat, label="done", max_steps=10_000),
    ]
