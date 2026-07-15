# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Interfaces for datagen data collection driven by the evaluation runners.

DatagenRunManagerBase is the single object injected into eval_runner.main. It spawns
and owns one DatagenCollectorBase per job and keeps whatever run-level records it
needs. The runners drive both objects only through these interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

# How an episode ended, as classified by episode_outcome.classify_outcome.
EpisodeOutcome = Literal["success", "failure", "timeout"]


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
    def on_episode_end(self, env: Any, outcome: EpisodeOutcome = "timeout") -> None:
        """Flush the in-progress episode with the outcome that ended it.

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

    def cap_episode_length(self, env_max_episode_length: int) -> int:
        """Return the per-episode step cap the rollout should use.

        Collectors with a recording budget return a smaller cap. The default keeps
        the env's own limit.
        """
        return env_max_episode_length


class DatagenRunManagerBase(ABC):
    """Interface for the run-level datagen object injected into eval_runner.main.

    The manager owns the collectors it creates: on_job_finished must close the job's
    collector (its cameras must be released before the stage teardown) and may record
    any bookkeeping it needs for the run.
    """

    @abstractmethod
    def start_run(self, eval_jobs_config: dict, description: str | None, device: str) -> None:
        """Called once before any job runs."""

    @abstractmethod
    def create_collector(self, job_name: str, datagen_job_dict: dict, env: Any) -> DatagenCollectorBase:
        """Build, retain, and return the collector for one job's live env."""

    @abstractmethod
    def on_job_finished(self, job: Any, env: Any) -> None:
        """Called once per collecting job after its rollout, while the env is still alive."""

    @abstractmethod
    def finish_run(self) -> None:
        """Called once after all jobs ran."""
