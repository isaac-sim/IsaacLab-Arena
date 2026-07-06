# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation.arena_experiment import (
    ArenaExperimentCfg,
    ArenaExperimentResult,
    ExperimentStatus,
    RolloutCfg,
)
from isaaclab_arena.policy.policy_base import PolicyCfg


@dataclass
class _EnvironmentCfg(ArenaEnvironmentCfg):
    value: int = 1


@dataclass
class _PolicyCfg(PolicyCfg):
    value: int = 2


def _experiment(**overrides) -> ArenaExperimentCfg:
    values = {
        "name": "test_experiment",
        "environment": _EnvironmentCfg(),
        "policy": _PolicyCfg(),
        "rollout": RolloutCfg(num_steps=10),
    }
    values.update(overrides)
    return ArenaExperimentCfg(**values)


def test_experiment_contains_only_evaluation_intent():
    """Keep execution state and results separate from the experiment declaration."""
    experiment = _experiment()

    assert experiment.name == "test_experiment"
    assert experiment.environment == _EnvironmentCfg()
    assert experiment.policy == _PolicyCfg()
    assert not hasattr(experiment, "status")
    assert not hasattr(experiment, "metrics")


def test_rollout_limits_are_mutually_exclusive_and_positive():
    with pytest.raises(AssertionError, match="mutually exclusive"):
        RolloutCfg(num_steps=1, num_episodes=1)
    with pytest.raises(AssertionError, match="num_steps must be greater"):
        RolloutCfg(num_steps=0)
    with pytest.raises(AssertionError, match="num_episodes must be greater"):
        RolloutCfg(num_episodes=0)


def test_episode_budget_gives_every_rebuild_an_episode():
    with pytest.raises(AssertionError, match="each rebuild runs at least one episode"):
        _experiment(rollout=RolloutCfg(num_episodes=2), num_rebuilds=3)


def test_experiment_result_records_outcome_separately():
    result = ArenaExperimentResult(
        experiment_name="test_experiment",
        status=ExperimentStatus.COMPLETED,
        started_at=10.0,
        ended_at=12.5,
    )

    assert result.duration_seconds == 2.5
    assert result.metrics is None
    assert result.error is None
