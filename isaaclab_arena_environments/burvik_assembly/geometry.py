# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Meter-scale BURVIK layout and nominal tray-to-part transforms (xyzw)."""

import numpy as np
from scipy.spatial.transform import Rotation

BENCH_HEIGHT = 0.75
PART_NAMES = ("tray", "leg_1", "leg_2", "leg_3", "leg_4", "handle")
ASSEMBLY_ORDER = PART_NAMES[1:]
MOUNTS = {"left": (0.0, 0.58, BENCH_HEIGHT), "right": (0.0, -0.58, BENCH_HEIGHT)}
TCP_OFFSET = (0.0, -0.044, 0.13)


def initial_part_poses(robot_layout=False):
    """Return supported loose-part poses as positions and scipy rotations."""
    poses = {
        "tray": (np.array([0.0, 0.06, BENCH_HEIGHT + 0.064]), Rotation.from_euler("x", 180, degrees=True)),
        "handle": (
            np.array([-0.18, -0.30, BENCH_HEIGHT + 0.008]),
            Rotation.from_euler("z", 90, degrees=True) * Rotation.from_euler("x", 90, degrees=True),
        ),
    }
    for index, x in enumerate((-0.49, -0.29, 0.29, 0.49), 1):
        poses[f"leg_{index}"] = (
            np.array([x, -0.10, BENCH_HEIGHT + 0.012]),
            Rotation.from_euler("x", 90, degrees=True),
        )
    if robot_layout:
        # Full hand poses need more clearance than positional TCP reachability alone.
        for name, x, y, roll in (
            ("leg_1", 0.25, 0.50, -90),
            ("leg_2", 0.25, -0.50, 90),
            ("leg_3", -0.25, 0.50, -90),
            ("leg_4", 0.33, 0.56, -90),
        ):
            poses[name] = (np.array([x, y, BENCH_HEIGHT + 0.012]), Rotation.from_euler("x", roll, degrees=True))
        poses["handle"] = (
            np.array([-0.541, -0.20, BENCH_HEIGHT + 0.043]),
            Rotation.from_euler("z", 90, degrees=True) * Rotation.from_euler("x", 90, degrees=True),
        )
    return poses


def attachment_pose(name, specification):
    """Return the nominal T_tray_part, including each leg's splay."""
    if name == "handle":
        # The 6 mm tube shoulders need clearance above the 35 mm tray surface.
        # Contact probes converge without penetration at this 1.5 mm stand-off.
        return np.array([0.0, 0.0, 0.0015]), Rotation.identity()
    socket = specification["sockets"][int(name[-1]) - 1]
    return np.array(socket["seat"]), Rotation.align_vectors([socket["axis"]], [[0, 0, 1]])[0]


def grasp_point(name):
    """Return a candidate centerline grasp point in the part frame."""
    if name.startswith("leg_"):
        return np.array([0.0, 0.0, -0.20])
    if name == "handle":
        return np.array([0.0, 0.0, 0.30])
    return np.array([0.0, -0.17, 0.035])
