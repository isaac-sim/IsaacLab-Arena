# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Camera coordinate helpers for the data-generation pipeline."""

from __future__ import annotations

from isaaclab_arena_datagen.camera_trajectory import CameraViewTrajectory, Coord3D

DEFAULT_CAMERA = CameraViewTrajectory(
    position=(0.0, 0.0, 1.0),
    target=(0.0, 0.0, 0.0),
    focal_length_mm=24.0,
)


def resolve_coord(
    coord: Coord3D | list[Coord3D],
    step: int = 0,
) -> Coord3D:
    """Return the (x, y, z) tuple for a camera coordinate at the given step.

    Supports both static and dynamic (per-step) camera coordinates.  A static
    coordinate is a single ``(x, y, z)`` tuple that is returned unchanged
    regardless of *step*.  A dynamic coordinate is a list of such tuples, one
    per simulation step, from which the entry at *step* is selected.

    Args:
        coord: Either a single ``(x, y, z)`` tuple (static) or a list of
            ``(x, y, z)`` tuples indexed by simulation step (dynamic).
        step: Zero-based simulation step index used to look up dynamic
            coordinates.  Ignored for static coordinates.  Defaults to ``0``.

    Returns:
        The resolved ``(x, y, z)`` coordinate for the requested step.
    """
    if isinstance(coord, tuple):
        return coord
    return coord[step]


def validate_camera_configs(cameras: list[CameraViewTrajectory], num_steps: int) -> None:
    """Verify that every dynamic camera coordinate has the correct step count.

    Delegates to :meth:`CameraViewTrajectory.validate_trajectory_length` for
    each camera.

    Args:
        cameras: Camera configurations to validate.
        num_steps: Expected number of simulation steps.

    Raises:
        ValueError: If a dynamic coordinate list length does not match
            *num_steps*.
    """
    for cam in cameras:
        cam.validate_trajectory_length(num_steps)
