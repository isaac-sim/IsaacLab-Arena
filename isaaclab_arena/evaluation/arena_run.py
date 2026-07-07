# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed declarations and results for Arena evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.policy.policy_base import PolicyCfg

if TYPE_CHECKING:
    from isaaclab_arena.metrics.metric_data import MetricsDataCollection


@dataclass
class RolloutCfg:
    """Configure the stopping condition for one evaluation rollout."""

    num_steps: int | None = None
    """Number of environment steps, or ``None`` for an episode-driven rollout."""

    num_episodes: int | None = None
    """Number of completed episodes, or ``None`` for a step-driven rollout."""

    def __post_init__(self) -> None:
        assert not (
            self.num_steps is not None and self.num_episodes is not None
        ), "num_steps and num_episodes are mutually exclusive"
        assert self.num_steps is None or self.num_steps > 0, "num_steps must be greater than zero"
        assert self.num_episodes is None or self.num_episodes > 0, "num_episodes must be greater than zero"


@dataclass
class ArenaRunCfg:
    """Declare one independently dispatchable Arena evaluation run."""

    name: str
    """Name used to identify the run and its recorded output."""

    environment: ArenaEnvironmentCfg
    """Concrete configuration for the environment under evaluation."""

    policy: PolicyCfg
    """Concrete configuration for the policy under evaluation."""

    environment_builder: ArenaEnvBuilderCfg = field(default_factory=ArenaEnvBuilderCfg)
    """Configuration used to compile the Arena environment for Isaac Lab."""

    rollout: RolloutCfg = field(default_factory=RolloutCfg)
    """Stopping condition for the policy rollout."""

    num_rebuilds: int = 1
    """Number of fresh environment constructions over which metrics are aggregated."""

    variations: dict[str, Any] = field(default_factory=dict)
    """Variation values applied when the environment is compiled."""

    def __post_init__(self) -> None:
        assert self.name, "run name must not be empty"
        assert self.num_rebuilds > 0, "num_rebuilds must be greater than zero"
        if self.rollout.num_episodes is not None:
            assert self.rollout.num_episodes >= self.num_rebuilds, (
                f"Run '{self.name}': num_episodes ({self.rollout.num_episodes}) must be >= num_rebuilds "
                f"({self.num_rebuilds}) so each rebuild runs at least one episode"
            )


class RunStatus(Enum):
    """Describe whether a run completed or failed."""

    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ArenaRunResult:
    """Record the outcome of executing one Arena run."""

    run_name: str
    """Name of the run that produced this result."""

    status: RunStatus
    """Final execution status."""

    metrics: MetricsDataCollection | None = None
    """Metrics aggregated over all successful rebuilds, when provided by the task."""
