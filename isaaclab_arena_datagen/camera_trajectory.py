# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Camera trajectory data types for the IsaacLab Arena data-generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

Coord3D = tuple[float, float, float]


@dataclass(frozen=True)
class CameraViewTrajectory:
    """Camera configuration describing a static view or a dynamic trajectory.

    The camera points at ``target`` (look-at). Both ``position`` and ``target``
    may be a single ``(x, y, z)`` tuple (static) or a list of ``num_steps``
    tuples -- one per simulation step -- so the camera moves through the scene.

    Resolution (width/height) is specified globally because the HDF5 dataset
    format requires all cameras to share the same spatial dimensions.

    Call :meth:`validate_trajectory_length` with the simulation step count to
    verify that dynamic coordinate lists have the expected length.
    """

    position: Coord3D | list[Coord3D]  # world frame
    target: Coord3D | list[Coord3D]  # world frame (look-at)
    focal_length_mm: float  # mm

    def __post_init__(self) -> None:
        if self.focal_length_mm <= 0:
            raise ValueError(f"focal_length_mm must be positive, got {self.focal_length_mm}")

    def validate_trajectory_length(self, num_steps: int) -> None:
        """Check that dynamic coordinate lists have exactly *num_steps* entries.

        Static coordinates (plain tuples) are accepted unconditionally.

        Raises:
            ValueError: If a dynamic coordinate list length does not match
                *num_steps*.
        """
        for key in ("position", "target"):
            val = getattr(self, key)
            if not isinstance(val, tuple) and len(val) != num_steps:
                raise ValueError(
                    f"'{key}' has {len(val)} entries but num_steps={num_steps}. Dynamic coordinates must match."
                )
