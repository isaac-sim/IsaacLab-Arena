# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Physics-settling configuration for clutter generation."""

import math
from dataclasses import dataclass


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
