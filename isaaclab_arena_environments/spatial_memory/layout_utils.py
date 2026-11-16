# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Small helpers shared by the task layouts: sampling, poses and the common counter."""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.parts import PartSpec, board

FLOOR_Z_M = -0.8755
"""Floor height in the environment frame: the Franka base sits on its stand, 0.8755 m above the floor."""

BOARD_T_M = 0.02
"""Thickness of fixture boards."""

COUNTER_SIZE_M = (0.80, 1.40, 0.04)
"""Counter slab in front of the robot, top face at z = 0."""


def uniform(generator: torch.Generator, low: float, high: float) -> float:
    """Return one sample from U(low, high)."""
    return low + (high - low) * float(torch.rand(1, generator=generator))


def choice(generator: torch.Generator, count: int) -> int:
    """Return one integer from {0, ..., count - 1}."""
    return int(torch.randint(count, (1,), generator=generator))


def permutation(generator: torch.Generator, count: int) -> list[int]:
    """Return a random permutation of {0, ..., count - 1}."""
    return torch.randperm(count, generator=generator).tolist()


def pose(x: float, y: float, z: float, yaw_rad: float = 0.0) -> list[float]:
    """Return an (x, y, z, qx, qy, qz, qw) pose with a yaw rotation."""
    return [x, y, z, 0.0, 0.0, math.sin(0.5 * yaw_rad), math.cos(0.5 * yaw_rad)]


def counter_spec(name: str = "counter", size_xyz: tuple[float, float, float] = COUNTER_SIZE_M) -> PartSpec:
    """Return the counter slab."""
    return board(name, size_xyz, colour="white")


def counter_pose(center_x: float = 0.60, center_y: float = 0.0, size_z: float = COUNTER_SIZE_M[2]) -> list[float]:
    """Return the pose that puts the counter's top face at z = 0."""
    return pose(center_x, center_y, -0.5 * size_z)
