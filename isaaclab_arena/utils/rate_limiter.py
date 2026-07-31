# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Callable


class RateLimiter:
    """Limit a loop to a requested period."""

    def __init__(self, period_seconds: float) -> None:
        """Initialize the limiter for a period in seconds.

        Args:
            period_seconds: Minimum time between loop iterations.
        """
        assert period_seconds > 0.0, "period_seconds must be greater than zero"
        self._period = period_seconds
        self._next_iteration_time = time.monotonic()
        self._callback_period = min(0.033, self._period)

    def sleep(self, wait_callback: Callable[[], None] | None = None) -> None:
        """Wait for the next iteration, optionally invoking a callback while waiting.

        Args:
            wait_callback: Function to invoke periodically while waiting, such as
                rendering a GUI frame.
        """
        self._next_iteration_time += self._period
        current_time = time.monotonic()

        while current_time < self._next_iteration_time:
            remaining_time = self._next_iteration_time - current_time
            sleep_duration = remaining_time if wait_callback is None else min(self._callback_period, remaining_time)
            time.sleep(sleep_duration)
            if wait_callback is not None:
                wait_callback()
            current_time = time.monotonic()

        if current_time > self._next_iteration_time:
            self._next_iteration_time = current_time
