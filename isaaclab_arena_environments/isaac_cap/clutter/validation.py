# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Rest detection and support-containment checks for clutter."""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass, field

from isaaclab.utils.math import quat_error_magnitude

from isaaclab_arena_environments.isaac_cap.clutter.drop_poses import ClutterRegion


@dataclass
class ClutterSettleParams:
    """Physics time budget, rest thresholds and support-containment tolerances."""

    timeout_s: float = 10.0
    """Maximum simulated seconds allowed for each candidate layout to settle."""

    poll_interval_s: float = 0.4
    """Simulated seconds between rest checks, independent of control decimation."""

    move_thresh_m: float = 0.002
    """Maximum translation between pose samples, in metres."""

    turn_thresh_deg: float = 2.0
    """Maximum rotation between pose samples, in degrees."""

    required_quiet_windows: int = 2
    """Required consecutive quiet pose samples."""

    fall_through_tolerance_m: float = 0.01
    """How far below the support surface an object may rest before it counts as tunnelled."""

    containment_margin_m: float = 0.0
    """How far outside the region an object may rest before it counts as fallen off."""

    passive_move_thresh_m: float = 0.002
    """Maximum passive-body displacement from its reset pose over the offline trial."""

    passive_turn_thresh_deg: float = 2.0
    """Maximum passive-body rotation from its reset pose over the offline trial."""

    def __post_init__(self) -> None:
        assert math.isfinite(self.timeout_s) and self.timeout_s > 0, "timeout_s must be finite and positive"
        assert (
            math.isfinite(self.poll_interval_s) and self.poll_interval_s > 0
        ), "poll_interval_s must be finite and positive"
        assert self.required_quiet_windows >= 1, "required_quiet_windows must be positive"
        assert (
            self.timeout_s >= (self.required_quiet_windows + 1) * self.poll_interval_s
        ), "timeout_s must cover a baseline sample and all required quiet windows"
        for name, value in (
            ("move_thresh_m", self.move_thresh_m),
            ("turn_thresh_deg", self.turn_thresh_deg),
            ("fall_through_tolerance_m", self.fall_through_tolerance_m),
            ("containment_margin_m", self.containment_margin_m),
            ("passive_move_thresh_m", self.passive_move_thresh_m),
            ("passive_turn_thresh_deg", self.passive_turn_thresh_deg),
        ):
            assert math.isfinite(value) and value >= 0, f"{name} must be finite and non-negative"


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

    def __init__(self, params: ClutterSettleParams | None = None):
        """Track rest using the supplied thresholds.

        Args:
            params: Thresholds for quiet windows. Defaults to ``ClutterSettleParams()``.
        """
        self._params = params or ClutterSettleParams()
        self._previous: tuple[torch.Tensor, torch.Tensor] | None = None
        self._quiet_windows = 0
        self._diverged: list[int] = []
        self._moving: list[int] = []

    @property
    def settled(self) -> bool:
        """Whether enough consecutive quiet polls have been seen."""
        return self._quiet_windows >= self._params.required_quiet_windows

    @property
    def quiet_windows(self) -> int:
        """How many consecutive quiet polls have been seen."""
        return self._quiet_windows

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
    positions: torch.Tensor,
    region: ClutterRegion,
    params: ClutterSettleParams | None = None,
    extents: list[tuple[float, float, float, float, float]] | None = None,
) -> ClutterRestVerdict:
    """Return containment failures for N members.

    Args:
        positions: Member positions in the region frame, shape (N, 3).
        region: Full support footprint and surface height, without the release spread scaling.
        params: Containment and fall-through tolerances.
        extents: N (min_x, min_y, max_x, max_y, min_z) offsets, each shape (5,).
            Omit to check origins only.
    """
    params = params or ClutterSettleParams()
    verdict = ClutterRestVerdict()
    margin = params.containment_margin_m
    floor = region.floor_z - params.fall_through_tolerance_m

    for index in range(positions.shape[0]):
        position = positions[index]
        if not bool(torch.isfinite(position).all()):
            verdict.diverged.append(index)
            continue
        x, y, z = (float(value) for value in position)
        offset_min_x, offset_min_y, offset_max_x, offset_max_y, offset_min_z = (
            extents[index] if extents is not None else (0.0, 0.0, 0.0, 0.0, 0.0)
        )
        if z + offset_min_z < floor:
            verdict.fell_through.append(index)
        if not (region.min_x - margin <= x + offset_min_x and x + offset_max_x <= region.max_x + margin) or not (
            region.min_y - margin <= y + offset_min_y and y + offset_max_y <= region.max_y + margin
        ):
            verdict.fell_off.append(index)
    return verdict
