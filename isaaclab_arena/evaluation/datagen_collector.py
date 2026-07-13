# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Interface for datagen data collectors driven by the evaluation runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DatagenCollectorBase(ABC):
    """Interface the evaluation runners use to drive a datagen data collector.

    Implementations record per-step data during a policy rollout (see rollout_policy in
    policy_runner.py). The runner calls on_step after every env.step, on_episode_end at each
    episode boundary before its explicit reset, finalize when the rollout finishes, and
    close when the job is done. How a collector is constructed and configured is up to the
    implementing package; the runners depend only on this interface.
    """

    @abstractmethod
    def on_step(self, env: Any, obs: Any, actions: Any, step_idx: int) -> None:
        """Record one frame after an env.step."""

    @abstractmethod
    def on_episode_end(self, env: Any, outcome: str = "timeout") -> None:
        """Flush the in-progress episode. outcome is "success", "failure", or "timeout".

        This is also the place to prepare the next episode's cameras (e.g. re-randomize
        their placement): the runner calls on_episode_end before the explicit env.reset()
        that starts the next episode, and it is that reset's RTX rerenders that flush new
        camera poses into the renderer. Re-aimed after the reset, the next episode's
        first frame would still render from the previous layout.
        """

    @abstractmethod
    def finalize(self, env: Any | None = None) -> None:
        """Flush any in-progress episode and stop recording. Idempotent."""

    @abstractmethod
    def close(self, env: Any | None = None) -> None:
        """Finalize, then release resources such as spawned cameras. Idempotent."""
