# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest

from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg, ArenaRunResult, RolloutLimitCfg, RunStatus
from isaaclab_arena.policy.policy_base import PolicyCfg


@dataclass
class _EnvironmentCfg(ArenaEnvironmentCfg):
    value: int = 1


@dataclass
class _PolicyCfg(PolicyCfg):
    value: int = 2


def _run(**overrides) -> ArenaRunCfg:
    values = {
        "name": "test_run",
        "environment": _EnvironmentCfg(),
        "policy": _PolicyCfg(),
        "rollout_limit": RolloutLimitCfg(num_steps=10),
    }
    values.update(overrides)
    return ArenaRunCfg(**values)


def test_run_contains_only_evaluation_intent():
    """Keep execution state and results separate from the run declaration."""
    run = _run()

    assert run.name == "test_run"
    assert run.environment == _EnvironmentCfg()
    assert run.policy == _PolicyCfg()
    assert not hasattr(run, "status")
    assert not hasattr(run, "metrics")


def test_rollout_limits_are_mutually_exclusive_and_positive():
    with pytest.raises(AssertionError, match="mutually exclusive"):
        RolloutLimitCfg(num_steps=1, num_episodes=1)
    with pytest.raises(AssertionError, match="num_steps must be greater"):
        RolloutLimitCfg(num_steps=0)
    with pytest.raises(AssertionError, match="num_episodes must be greater"):
        RolloutLimitCfg(num_episodes=0)


def test_episode_budget_gives_every_rebuild_an_episode():
    with pytest.raises(AssertionError, match="each rebuild runs at least one episode"):
        _run(rollout_limit=RolloutLimitCfg(num_episodes=2), num_rebuilds=3)


def test_replay_paths_determine_rebuild_count():
    run = _run(
        rollout_limit=RolloutLimitCfg(),
        num_rebuilds=99,
        recorded_variation_samples_paths=["rebuild0.jsonl", "rebuild1.jsonl"],
    )

    assert run.num_rebuilds == 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"recorded_variation_samples_paths": []}, "must not be empty"),
        (
            {
                "recorded_variation_samples_paths": ["rebuild0.jsonl"],
                "environment_builder": ArenaEnvBuilderCfg(recorded_variation_samples_path="legacy.jsonl"),
            },
            "cannot be combined",
        ),
        (
            {
                "recorded_variation_samples_paths": ["rebuild0.jsonl"],
                "rollout_limit": RolloutLimitCfg(num_episodes=1),
            },
            "rollout limits are not supported",
        ),
    ],
)
def test_replay_paths_reject_conflicting_configuration(overrides, message):
    with pytest.raises(AssertionError, match=message):
        _run(**overrides)


def test_run_result_records_outcome_separately():
    result = ArenaRunResult(
        run_name="test_run",
        status=RunStatus.COMPLETED,
    )

    assert result.run_name == "test_run"
    assert result.metrics is None
