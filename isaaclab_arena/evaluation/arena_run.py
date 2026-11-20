# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed declarations and results for Arena evaluation runs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from prettytable import PrettyTable
from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.registries import PolicyRegistry
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation.report_data import success_rate_from_values
from isaaclab_arena.evaluation.run_status import RunStatus
from isaaclab_arena.policy.policy_base import PolicyCfg

if TYPE_CHECKING:
    from isaaclab_arena.evaluation.arena_experiment_result import ArenaExperimentResult, ArenaExperimentRunData
    from isaaclab_arena.relations.placement_diagnostics import AggregatedPlacementDiagnostics


@dataclass
class RolloutLimitCfg:
    """Configure the stopping limit for one evaluation rollout."""

    num_steps: int | None = None
    """Number of environment steps, or ``None`` for an episode-driven rollout."""

    num_episodes: int | None = None
    """Total number of episodes to start and finish, split evenly across rebuilds, or ``None`` for another limit."""

    num_episodes_per_rebuild: list[int] | None = None
    """Number of episodes to start and finish in each rebuild, in rebuild order, or ``None`` for another limit.
    Its length must equal the Run's ``num_rebuilds``. It cannot be combined with ``num_steps``, and ``num_episodes``
    may also be set only if it equals the sum."""

    def __post_init__(self) -> None:
        has_episode_limit = self.num_episodes is not None or self.num_episodes_per_rebuild is not None
        assert (
            self.num_steps is None or not has_episode_limit
        ), "num_steps is mutually exclusive with num_episodes and num_episodes_per_rebuild"
        assert self.num_steps is None or self.num_steps > 0, "num_steps must be greater than zero"
        assert self.num_episodes is None or self.num_episodes > 0, "num_episodes must be greater than zero"
        if self.num_episodes_per_rebuild is not None:
            assert self.num_episodes_per_rebuild, "num_episodes_per_rebuild must not be empty"
            assert all(
                num_episodes > 0 for num_episodes in self.num_episodes_per_rebuild
            ), f"num_episodes_per_rebuild entries must be greater than zero; got {self.num_episodes_per_rebuild}"
            total_num_episodes = sum(self.num_episodes_per_rebuild)
            assert self.num_episodes is None or self.num_episodes == total_num_episodes, (
                f"num_episodes ({self.num_episodes}) must equal the sum of num_episodes_per_rebuild "
                f"({total_num_episodes}) when both are set"
            )


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

    rollout_limit: RolloutLimitCfg = field(default_factory=RolloutLimitCfg)
    """Step or episode limit for the policy rollout."""

    num_rebuilds: int = 1
    """Number of fresh environment constructions that the Run's episodes are split across. When
    ``rollout_limit.num_episodes_per_rebuild`` is set, this must equal its length."""

    recorded_variation_samples_paths: list[str] | None = None
    """Ordered replay JSONLs, one per environment rebuild."""

    variations: dict[str, Any] = field(default_factory=dict)
    """Variation values applied when the environment is compiled."""

    def __post_init__(self) -> None:
        assert self.name, "run name must not be empty"
        if self.recorded_variation_samples_paths is not None:
            assert self.recorded_variation_samples_paths, "recorded_variation_samples_paths must not be empty"
            assert all(
                isinstance(path, str) and path for path in self.recorded_variation_samples_paths
            ), "recorded_variation_samples_paths must contain nonempty paths"
            assert self.environment_builder.recorded_variation_samples_path is None, (
                "recorded_variation_samples_paths cannot be combined with "
                "environment_builder.recorded_variation_samples_path"
            )
            num_replay_paths = len(self.recorded_variation_samples_paths)
            assert self.num_rebuilds in (1, num_replay_paths), (
                f"num_rebuilds ({self.num_rebuilds}) conflicts with the {num_replay_paths} "
                "recorded_variation_samples_paths; omit num_rebuilds or set it to the number of paths"
            )
            self.num_rebuilds = num_replay_paths
        if (
            self.recorded_variation_samples_paths is not None
            or self.environment_builder.recorded_variation_samples_path is not None
        ):
            assert (
                self.rollout_limit.num_steps is None
            ), "Replay from recorded variation samples does not support a num_steps rollout limit"
            assert self.rollout_limit.num_episodes_per_rebuild is None, (
                "Replay from recorded variation samples does not support num_episodes_per_rebuild; "
                "use num_episodes or leave the limit unset to replay every recorded episode"
            )
        if self.rollout_limit.num_episodes_per_rebuild is not None:
            num_listed_rebuilds = len(self.rollout_limit.num_episodes_per_rebuild)
            assert self.num_rebuilds == num_listed_rebuilds, (
                f"Run '{self.name}': num_rebuilds ({self.num_rebuilds}) must equal the {num_listed_rebuilds} "
                "entries of num_episodes_per_rebuild"
            )
        assert self.num_rebuilds > 0, "num_rebuilds must be greater than zero"
        if self.rollout_limit.num_episodes is not None:
            assert self.rollout_limit.num_episodes >= self.num_rebuilds, (
                f"Run '{self.name}': num_episodes ({self.rollout_limit.num_episodes}) must be >= num_rebuilds "
                f"({self.num_rebuilds}) so each rebuild runs at least one episode"
            )


@dataclass
class ArenaRunResult:
    """Record the outcome of executing one Arena run."""

    run_name: str
    """Name of the run that produced this result."""

    status: RunStatus
    """Final execution status."""

    start_timestamp_ns: int
    """Unix epoch nanoseconds (UTC) at which the run started."""

    end_timestamp_ns: int
    """Unix epoch nanoseconds (UTC) at which the run finished or failed."""

    placement_diagnostics: AggregatedPlacementDiagnostics | None = None
    """Diagnostics of the placement pool fills that built the rebuilds' environments. Rebuilds that solve no
    relations, or that replay recorded layouts, are left out, and the field is None when every rebuild is."""


def build_runs_info_table(
    run_cfgs: Iterable[ArenaRunCfg],
    results: Iterable[ArenaRunResult],
    experiment_result: ArenaExperimentResult | None = None,
) -> PrettyTable:
    """Build a table with the configuration and status of every Run.

    Args:
        run_cfgs: The configured Runs, one row each.
        results: The Runs' execution results; Runs without one are shown as pending.
        experiment_result: The finished Experiment's result. When given, the table also shows each Run's recorded
            episode count and success rate.
    """
    results_by_name = {result.run_name: result for result in results}
    field_names = ["Run Name", "Status", "Policy Type", "Num Envs", "Num Steps", "Num Episodes", "Num Rebuilds"]
    if experiment_result is not None:
        field_names += ["Recorded Episodes", "Success Rate"]
    table = PrettyTable(field_names=field_names)
    policy_registry = PolicyRegistry()
    for run_cfg in run_cfgs:
        result = results_by_name.get(run_cfg.name)
        policy_type = policy_registry.get_policy_type_for_cfg(run_cfg.policy)
        rollout_limit = run_cfg.rollout_limit
        if rollout_limit.num_episodes_per_rebuild is not None:
            num_episodes = rollout_limit.num_episodes_per_rebuild
        else:
            num_episodes = rollout_limit.num_episodes
        row = [
            run_cfg.name,
            result.status.value if result is not None else "pending",
            policy_type.name,
            run_cfg.environment_builder.num_envs,
            rollout_limit.num_steps,
            num_episodes,
            run_cfg.num_rebuilds,
        ]
        if experiment_result is not None:
            row += _recorded_episodes_and_success_rate(experiment_result.runs[run_cfg.name])
        table.add_row(row)
    return table


def _recorded_episodes_and_success_rate(run_data: ArenaExperimentRunData) -> list[int | str]:
    """Return a Run's number of recorded episodes and its success rate, formatted for the Run table."""
    episodes = []
    for rebuild in run_data["rebuilds"]:
        episodes.extend(rebuild["episodes"])
    run_success_rate = success_rate_from_values(episode["success"] for episode in episodes)
    return [len(episodes), "n/a" if run_success_rate is None else f"{run_success_rate:.4f}"]
