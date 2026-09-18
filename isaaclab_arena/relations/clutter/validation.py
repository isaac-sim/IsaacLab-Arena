# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Rest detection and support-containment checks for clutter."""

from __future__ import annotations

import torch
from dataclasses import dataclass, field

from isaaclab.utils.math import quat_error_magnitude

from isaaclab_arena.relations.clutter.geometry import ClutterRegion
from isaaclab_arena.relations.clutter.settle_params import ClutterSettleParams
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


@dataclass
class ClutterRestVerdict:
    """Indices of clutter members that failed containment checks."""

    diverged: list[int] = field(default_factory=list)
    """Indices of non-finite poses."""

    fell_through: list[int] = field(default_factory=list)
    """Indices below the support surface."""

    fell_off: list[int] = field(default_factory=list)
    """Indices outside the support footprint."""

    @property
    def ok(self) -> bool:
        """Whether every member satisfies the containment checks."""
        return not (self.diverged or self.fell_through or self.fell_off)

    def describe(self, names: list[str]) -> str:
        """Return a human-readable summary naming the offending members."""
        parts = []
        for label, indices in (
            ("diverged", self.diverged),
            ("fell through", self.fell_through),
            ("fell off", self.fell_off),
        ):
            if indices:
                offenders = ", ".join(names[index] for index in indices)
                parts.append(f"{label}: {offenders}")
        return "; ".join(parts) if parts else "all members within support"


class SettleTracker:
    """Consecutive quiet pose windows for N objects. Motion resets the quiet streak.

    A pile can pause before toppling, so one quiet sample is insufficient.
    """

    def __init__(self, params: ClutterSettleParams):
        self._params = params
        self._previous: tuple[torch.Tensor, torch.Tensor] | None = None
        """Previous positions (N, 3) and xyzw quaternions (N, 4), or None before the first finite sample."""
        self._quiet_windows = 0
        self._diverged: list[int] = []
        self._moving: list[int] = []

    @property
    def settled(self) -> bool:
        """Whether enough consecutive quiet polls have been seen."""
        return self._quiet_windows >= self._params.required_quiet_windows

    @property
    def diverged(self) -> bool:
        """Whether the latest sample contains non-finite poses."""
        return bool(self._diverged)

    def failure_reason(self, names: list[str]) -> str | None:
        """Describe why the current window is not settled, naming affected objects."""
        if self._diverged:
            return "non-finite poses: " + ", ".join(names[i] for i in self._diverged)
        if self.settled:
            return None
        if self._moving:
            return "still moving: " + ", ".join(names[i] for i in self._moving)
        return f"insufficient quiet windows: {self._quiet_windows}/{self._params.required_quiet_windows}"

    def update(self, positions: torch.Tensor, rotations: torch.Tensor) -> bool:
        """Record a snapshot and return whether enough quiet windows have elapsed.

        Args:
            positions: Object positions, shape (N, 3).
            rotations: Object quaternions (x, y, z, w), shape (N, 4).
        """
        finite = torch.isfinite(positions).all(dim=-1) & torch.isfinite(rotations).all(dim=-1)
        self._diverged = (~finite).nonzero().flatten().tolist()
        self._moving = []
        if self._diverged:
            self._quiet_windows = 0
            self._previous = None
            return False
        if self._previous is None:
            self._previous = (positions.clone(), rotations.clone())
            return False
        previous_positions, previous_rotations = self._previous
        distance = (positions - previous_positions).norm(dim=-1)
        angle = torch.rad2deg(quat_error_magnitude(rotations, previous_rotations))
        moving = (distance > self._params.move_thresh_m) | (angle > self._params.turn_thresh_deg)
        self._moving = moving.nonzero().flatten().tolist()
        self._quiet_windows = 0 if self._moving else self._quiet_windows + 1
        self._previous = (positions.clone(), rotations.clone())
        return self.settled


def check_resting_poses(
    bounds: AxisAlignedBoundingBox,
    region: ClutterRegion,
    params: ClutterSettleParams,
) -> ClutterRestVerdict:
    """Return containment failures for N members.

    Args:
        bounds: Rotated object bounds in the environment frame, min/max shape (N, 3).
        region: Full support footprint and surface height, without the release spread scaling.
        params: Containment and fall-through tolerances.
    """
    verdict = ClutterRestVerdict()
    margin = params.containment_margin_m
    floor = region.floor_z - params.fall_through_tolerance_m
    for index, (lower, upper) in enumerate(zip(bounds.min_point, bounds.max_point, strict=True)):
        if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()):
            verdict.diverged.append(index)
            continue
        if lower[2] < floor:
            verdict.fell_through.append(index)
        if not (
            lower[0] >= region.min_x - margin
            and upper[0] <= region.max_x + margin
            and lower[1] >= region.min_y - margin
            and upper[1] <= region.max_y + margin
        ):
            verdict.fell_off.append(index)
    return verdict
