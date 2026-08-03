# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from isaaclab_arena.utils import rate_limiter


class FakeClock:
    """Provide deterministic monotonic time and sleep behavior."""

    def __init__(self) -> None:
        self.current_time = 0.0
        self.sleep_durations: list[float] = []

    def monotonic(self) -> float:
        return self.current_time

    def sleep(self, duration: float) -> None:
        self.sleep_durations.append(duration)
        self.current_time += duration


def test_rate_limiter_waits_for_the_remaining_period(monkeypatch):
    fake_clock = FakeClock()
    monkeypatch.setattr(rate_limiter.time, "monotonic", fake_clock.monotonic)
    monkeypatch.setattr(rate_limiter.time, "sleep", fake_clock.sleep)
    limiter = rate_limiter.RateLimiter(period_seconds=0.1)

    fake_clock.current_time += 0.025
    limiter.sleep()

    assert fake_clock.current_time == pytest.approx(0.1)
    assert fake_clock.sleep_durations == pytest.approx([0.075])


def test_rate_limiter_invokes_callback_while_waiting(monkeypatch):
    fake_clock = FakeClock()
    callback_times: list[float] = []
    monkeypatch.setattr(rate_limiter.time, "monotonic", fake_clock.monotonic)
    monkeypatch.setattr(rate_limiter.time, "sleep", fake_clock.sleep)
    limiter = rate_limiter.RateLimiter(period_seconds=0.1)

    limiter.sleep(wait_callback=lambda: callback_times.append(fake_clock.current_time))

    assert fake_clock.current_time == pytest.approx(0.1)
    assert len(callback_times) == 4
    assert max(fake_clock.sleep_durations) <= 0.033


def test_rate_limiter_resynchronizes_after_an_overrun(monkeypatch):
    fake_clock = FakeClock()
    monkeypatch.setattr(rate_limiter.time, "monotonic", fake_clock.monotonic)
    monkeypatch.setattr(rate_limiter.time, "sleep", fake_clock.sleep)
    limiter = rate_limiter.RateLimiter(period_seconds=0.1)

    fake_clock.current_time = 0.2
    limiter.sleep()
    limiter.sleep()

    assert fake_clock.sleep_durations == pytest.approx([0.1])
    assert fake_clock.current_time == pytest.approx(0.3)


def test_rate_limiter_rejects_non_positive_period():
    with pytest.raises(AssertionError, match="greater than zero"):
        rate_limiter.RateLimiter(period_seconds=0.0)
