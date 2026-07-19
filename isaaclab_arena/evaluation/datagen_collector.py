# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Interfaces for datagen data collection driven by the evaluation runners.

DatagenRunManagerBase is the single object injected into eval_runner.main. It drives one
DatagenCollectorBase per job and keeps whatever run-level records it needs. The runners
drive both objects only through these interfaces. NoOpDatagenRunManager is the default
when none is injected, so the runners never special-case the no-collection path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

# How an episode ended.
EpisodeOutcome = Literal["success", "failure", "timeout"]


class DatagenCollectorBase(ABC):
    """Interface the evaluation runners use to drive a datagen data collector.

    Implementations record per-step data during a policy rollout (see rollout_policy in
    policy_runner.py). The runner calls on_step after every env.step, on_episode_end when the
    env resets between episodes, finalize when the rollout finishes, and close when the job is
    done. How a collector is constructed and configured is up to the implementing package. The
    runners depend only on this interface.
    """

    @abstractmethod
    def on_step(self, env: Any, obs: Any, actions: Any, step_idx: int) -> None:
        """Record one frame after an env.step."""

    @abstractmethod
    def on_episode_end(self, env: Any, outcome: EpisodeOutcome = "timeout") -> None:
        """Flush the in-progress episode with the outcome that ended it.

        Also the place to prepare the next episode's cameras (e.g. re-randomize their
        placement). The runner calls this once the env has auto-reset, so any re-aimed poses
        take effect over the next episode's leading frames, which the collector is expected
        to drop.
        """

    @abstractmethod
    def finalize(self, env: Any | None = None) -> None:
        """Flush any in-progress episode and stop recording. Idempotent."""

    @abstractmethod
    def close(self, env: Any | None = None) -> None:
        """Finalize, then release resources such as spawned cameras. Idempotent."""


class DatagenRunManagerBase(ABC):
    """Interface for the run-level datagen object injected into eval_runner.main.

    on_job_finished runs while the job's env is still alive, so an implementation can release
    per-job resources such as spawned cameras before the stage teardown and record whatever
    run-level bookkeeping it needs.
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

    def needs_cameras(self) -> bool:
        """Whether collection requires the sim to start with camera support enabled."""
        return True

    def prepare_env_cfg(self, env_cfg: Any) -> None:
        """Adjust an env cfg before its env is built. Called once per job.

        The default drops the env's metrics and their recorder terms so the collector's own
        dataset is the only one written.
        """
        if hasattr(env_cfg, "metrics"):
            env_cfg.metrics = None
        if hasattr(env_cfg, "recorders"):
            env_cfg.recorders = None


class NoOpDatagenCollector(DatagenCollectorBase):
    """Collector that records nothing, used when no datagen collection is configured."""

    def on_step(self, env: Any, obs: Any, actions: Any, step_idx: int) -> None:
        pass

    def on_episode_end(self, env: Any, outcome: EpisodeOutcome = "timeout") -> None:
        pass

    def finalize(self, env: Any | None = None) -> None:
        pass

    def close(self, env: Any | None = None) -> None:
        pass


class NoOpDatagenRunManager(DatagenRunManagerBase):
    """Run manager that collects nothing. eval_runner defaults to it when none is injected."""

    def start_run(self, eval_jobs_config: dict, description: str | None, device: str) -> None:
        pass

    def create_collector(self, job_name: str, datagen_job_dict: dict, env: Any) -> DatagenCollectorBase:
        return NoOpDatagenCollector()

    def on_job_finished(self, job: Any, env: Any) -> None:
        pass

    def finish_run(self) -> None:
        pass

    def needs_cameras(self) -> bool:
        return False

    def prepare_env_cfg(self, env_cfg: Any) -> None:
        pass
