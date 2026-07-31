# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verdicts on a poured clutter pile.

Physics already guarantees that settled objects do not interpenetrate, so this checks what
physics does not: that nothing fell off the support, fell through it, or diverged, and that
the pile stopped moving. Overlap is deliberately not tested. A resting object is tangent to
its support, a tilted object's axis-aligned box encloses far more than its geometry, and a
concave asset's box encloses more still, so box intersection reports contact that is either
expected or imaginary.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field

from isaaclab_arena.relations.clutter_drop_poses import ClutterRegion


@dataclass(frozen=True)
class ClutterSettleParams:
    """Thresholds deciding when a pile has stopped and whether it came to rest correctly."""

    move_thresh_m: float = 0.002
    """No object may translate more than this between polls for the pile to count as quiet."""

    turn_thresh_deg: float = 2.0
    """No object may rotate more than this between polls for the pile to count as quiet."""

    required_quiet_windows: int = 2
    """Consecutive quiet polls before the pile is called settled.

    Settling is not monotonic: a pile can go quiet, avalanche, then settle. One quiet poll is
    therefore not evidence that it has stopped.
    """

    fall_through_tolerance_m: float = 0.01
    """How far below the support surface an object may rest before it counts as tunnelled."""

    containment_margin_m: float = 0.0
    """How far outside the region an object may rest before it counts as fallen off."""


@dataclass
class ClutterRestVerdict:
    """Which members of a pile came to rest badly, by failure mode."""

    diverged: list[int] = field(default_factory=list)
    """Indices whose pose is not finite."""

    fell_through: list[int] = field(default_factory=list)
    """Indices resting below the support surface."""

    fell_off: list[int] = field(default_factory=list)
    """Indices resting outside the region."""

    @property
    def ok(self) -> bool:
        """Whether every member came to rest acceptably."""
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
        return "; ".join(parts) if parts else "all members at rest"


def quaternion_angle_deg(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Return the angle in degrees between two batches of quaternions.

    Args:
        first: Quaternions, shape ``(N, 4)``.
        second: Quaternions, shape ``(N, 4)``.
    """
    # A quaternion and its negation are the same rotation, so compare magnitudes.
    dots = (first * second).sum(dim=-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.acos(dots))


class SettleTracker:
    """Decides when a pile has stopped, from a sequence of pose snapshots.

    Velocity cannot decide this. Objects in stable contact keep micro-rocking indefinitely,
    and some assets report angular speeds irreconcilable with their own pose history, so a
    velocity threshold either never fires or fires while the pile is still rearranging.
    """

    def __init__(self, params: ClutterSettleParams | None = None):
        """
        Args:
            params: Thresholds for quiet windows. Defaults to ``ClutterSettleParams()``.
        """
        self._params = params or ClutterSettleParams()
        self._previous: tuple[torch.Tensor, torch.Tensor] | None = None
        self._quiet_windows = 0

    @property
    def settled(self) -> bool:
        """Whether enough consecutive quiet polls have been seen."""
        return self._quiet_windows >= self._params.required_quiet_windows

    @property
    def quiet_windows(self) -> int:
        """How many consecutive quiet polls have been seen."""
        return self._quiet_windows

    def update(self, positions: torch.Tensor, rotations: torch.Tensor) -> bool:
        """Record a pose snapshot and return whether the pile now counts as settled.

        A non-finite snapshot resets the streak, so a diverged pile can never be called settled.

        Args:
            positions: Member positions, shape ``(N, 3)``.
            rotations: Member rotations as ``(x, y, z, w)``, shape ``(N, 4)``.
        """
        if not bool(torch.isfinite(positions).all() and torch.isfinite(rotations).all()):
            self._quiet_windows = 0
            self._previous = None
            return False

        if self._previous is None:
            self._previous = (positions.clone(), rotations.clone())
            return False

        previous_positions, previous_rotations = self._previous
        moved = float((positions - previous_positions).norm(dim=-1).max())
        turned = float(quaternion_angle_deg(rotations, previous_rotations).max())
        quiet = moved <= self._params.move_thresh_m and turned <= self._params.turn_thresh_deg

        self._quiet_windows = self._quiet_windows + 1 if quiet else 0
        self._previous = (positions.clone(), rotations.clone())
        return self.settled


def check_resting_poses(
    positions: torch.Tensor,
    region: ClutterRegion,
    params: ClutterSettleParams | None = None,
) -> ClutterRestVerdict:
    """Return which members came to rest badly.

    Args:
        positions: Member positions, shape ``(N, 3)``, in the same frame as ``region``.
        region: The region the pile was poured into, whose floor is the support surface.
        params: Tolerances. Defaults to ``ClutterSettleParams()``.
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
        if z < floor:
            verdict.fell_through.append(index)
        if not (region.min_x - margin <= x <= region.max_x + margin) or not (
            region.min_y - margin <= y <= region.max_y + margin
        ):
            verdict.fell_off.append(index)
    return verdict
