# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared placement-test helpers."""

from __future__ import annotations

from isaaclab_arena.relations.placement_result import PlacementResult


def layout_signature(result: PlacementResult):
    """Name-keyed (positions, orientations, validation) tuple for comparing layouts across instances."""
    return (
        {obj.name: tuple(pos) for obj, pos in result.positions.items()},
        {obj.name: yaw for obj, yaw in result.orientations.items()},
        dict(result.validation.checks),
    )
