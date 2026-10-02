# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Share scale-aware UV degeneracy tolerances between Blender and USD validation."""

from collections.abc import Iterable, Sequence

UV_DOUBLED_AREA_TOLERANCE = 1e-12
"""Minimum noncollapsed triangle area in normalized texture coordinates, doubled."""


def surface_area_tolerance(points: Iterable[Sequence[float]]) -> float:
    """Bound float32 bevel slivers by machine precision times the squared mesh diagonal."""
    lower = [float("inf")] * 3
    upper = [-float("inf")] * 3
    for point in points:
        for axis in range(3):
            lower[axis] = min(lower[axis], point[axis])
            upper[axis] = max(upper[axis], point[axis])
    diagonal_squared = sum((high - low) ** 2 for low, high in zip(lower, upper, strict=True))
    return max(1e-16, 2**-23 * diagonal_squared)
