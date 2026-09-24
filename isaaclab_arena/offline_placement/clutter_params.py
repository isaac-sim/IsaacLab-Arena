# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Physics-settling configuration for clutter generation."""

import math
from dataclasses import dataclass


@dataclass
class ClutterSettleParams:
    """Physics duration and sampling interval for offline clutter generation."""

    timeout_s: float = 10.0
    """Maximum simulated seconds allowed for each candidate layout to settle."""

    poll_interval_s: float = 0.4
    """Simulated seconds between rest checks, independent of control decimation."""

    def __post_init__(self) -> None:
        assert math.isfinite(self.timeout_s) and self.timeout_s > 0, "timeout_s must be finite and positive"
        assert (
            math.isfinite(self.poll_interval_s) and self.poll_interval_s > 0
        ), "poll_interval_s must be finite and positive"
        assert self.timeout_s >= self.poll_interval_s, "timeout_s must cover one sample interval"
