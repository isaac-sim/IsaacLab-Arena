# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Camera placement helpers for tightly framed asset snapshots."""

from __future__ import annotations

import math


def fit_camera_to_bounds(
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
    view_direction: tuple[float, float, float],
    *,
    vertical_fov_degrees: float = 60.0,
    margin: float = 1.1,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return an eye and target that fit an axis-aligned bound in a perspective camera."""
    target = tuple((low + high) * 0.5 for low, high in zip(minimum, maximum))
    diagonal = math.sqrt(sum((high - low) ** 2 for low, high in zip(minimum, maximum)))
    direction_length = math.sqrt(sum(component**2 for component in view_direction))
    assert diagonal > 0.0, "Camera bounds must have a positive diagonal."
    assert direction_length > 0.0, "Camera view direction must be nonzero."
    assert 0.0 < vertical_fov_degrees < 180.0, "Camera field of view must be between 0 and 180 degrees."
    assert margin >= 1.0, "Camera framing margin must be at least 1."

    # Enclose the AABB in a sphere. Fitting that sphere to the vertical field of view
    # keeps arbitrary object orientations in frame while avoiding Kit's large default margin.
    radius = diagonal * 0.5
    half_fov_radians = math.radians(vertical_fov_degrees) * 0.5
    distance = margin * radius / math.sin(half_fov_radians)
    unit_direction = tuple(component / direction_length for component in view_direction)
    eye = tuple(center + direction * distance for center, direction in zip(target, unit_direction))
    return eye, target
