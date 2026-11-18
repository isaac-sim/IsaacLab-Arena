# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest

from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg, ArenaRunResult, RolloutLimitCfg
from isaaclab_arena.evaluation.run_status import RunStatus
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
    with pytest.raises(AssertionError, match="mutually exclusive"):
        RolloutLimitCfg(num_steps=2, num_episodes_per_rebuild=[1, 1])
    with pytest.raises(
        AssertionError, match="num_episodes \\(3\\) must equal the sum of num_episodes_per_rebuild \\(2\\)"
    ):
        RolloutLimitCfg(num_episodes=3, num_episodes_per_rebuild=[1, 1])
    with pytest.raises(AssertionError, match="must not be empty"):
        RolloutLimitCfg(num_episodes_per_rebuild=[])
    with pytest.raises(AssertionError, match="entries must be greater than zero"):
        RolloutLimitCfg(num_episodes_per_rebuild=[3, 0])


def test_episodes_per_rebuild_accept_an_equal_episode_total():
    rollout_limit = RolloutLimitCfg(num_episodes=30, num_episodes_per_rebuild=[20, 5, 5])

    assert rollout_limit.num_episodes == 30
    assert rollout_limit.num_episodes_per_rebuild == [20, 5, 5]


def test_episode_budget_gives_every_rebuild_an_episode():
    with pytest.raises(AssertionError, match="each rebuild runs at least one episode"):
        _run(rollout_limit=RolloutLimitCfg(num_episodes=2), num_rebuilds=3)


@pytest.mark.parametrize("num_rebuilds", [1, 2])
def test_episodes_per_rebuild_require_matching_rebuild_count(num_rebuilds):
    with pytest.raises(AssertionError, match=f"num_rebuilds \\({num_rebuilds}\\) must equal the 3 entries"):
        _run(rollout_limit=RolloutLimitCfg(num_episodes_per_rebuild=[20, 5, 5]), num_rebuilds=num_rebuilds)


def test_replay_paths_determine_rebuild_count():
    run = _run(
        rollout_limit=RolloutLimitCfg(num_episodes=5),
        recorded_variation_samples_paths=["rebuild0.jsonl", "rebuild1.jsonl"],
    )

    assert run.num_rebuilds == 2
    assert run.rollout_limit.num_episodes == 5


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
                "rollout_limit": RolloutLimitCfg(num_steps=1),
            },
            "does not support a num_steps",
        ),
        (
            {
                "environment_builder": ArenaEnvBuilderCfg(recorded_variation_samples_path="rebuild0.jsonl"),
                "rollout_limit": RolloutLimitCfg(num_steps=1),
            },
            "does not support a num_steps",
        ),
        (
            {
                "recorded_variation_samples_paths": ["rebuild0.jsonl", "rebuild1.jsonl"],
                "num_rebuilds": 99,
                "rollout_limit": RolloutLimitCfg(num_episodes=5),
            },
            "num_rebuilds \\(99\\) conflicts with the 2 recorded_variation_samples_paths",
        ),
        (
            {
                "recorded_variation_samples_paths": ["rebuild0.jsonl", "rebuild1.jsonl"],
                "num_rebuilds": 2,
                "rollout_limit": RolloutLimitCfg(num_episodes_per_rebuild=[1, 2]),
            },
            "does not support num_episodes_per_rebuild",
        ),
        (
            {
                "environment_builder": ArenaEnvBuilderCfg(recorded_variation_samples_path="rebuild0.jsonl"),
                "num_rebuilds": 2,
                "rollout_limit": RolloutLimitCfg(num_episodes_per_rebuild=[1, 2]),
            },
            "does not support num_episodes_per_rebuild",
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
        start_timestamp_ns=1792799000000000000,
        end_timestamp_ns=1792800000000000000,
    )

    assert result.run_name == "test_run"
    assert result.start_timestamp_ns == 1792799000000000000
    assert result.end_timestamp_ns == 1792800000000000000


def _run_data(episodes_by_rebuild: list[list[dict]]) -> dict:
    return {
        "environment": {"type": "test"},
        "policy_variant": "test",
        "status": RunStatus.COMPLETED.value,
        "start_timestamp_ns": 0,
        "end_timestamp_ns": 1,
        "rebuilds": [{"index": index, "episodes": episodes} for index, episodes in enumerate(episodes_by_rebuild)],
    }


def test_runs_info_table_shows_recorded_episodes_once_the_experiment_finished(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch

    from isaaclab_arena.assets.registries import PolicyRegistry
    from isaaclab_arena.evaluation.arena_experiment_result import ArenaExperimentResult
    from isaaclab_arena.evaluation.arena_run import build_runs_info_table
    from isaaclab_arena.utils.git_stamp import GitStamp

    run_cfgs = [_run(name="scored_run"), _run(name="unscored_run")]
    run_results = [
        ArenaRunResult(run_name=run_cfg.name, status=RunStatus.COMPLETED, start_timestamp_ns=0, end_timestamp_ns=1)
        for run_cfg in run_cfgs
    ]
    experiment_result = ArenaExperimentResult(
        tmp_path,
        "test_experiment",
        GitStamp(git_sha="unknown", git_branch="unknown"),
        {
            # An episode without a success value is recorded but not scored.
            "scored_run": _run_data([[{"success": True}, {"success": False}], [{"success": None}]]),
            "unscored_run": _run_data([[{"success": None}]]),
        },
    )

    with patch.object(PolicyRegistry, "get_policy_type_for_cfg", return_value=SimpleNamespace(name="test_policy")):
        table_before_runs = build_runs_info_table(run_cfgs, [])
        table_after_runs = build_runs_info_table(run_cfgs, run_results, experiment_result)

    assert "Success Rate" not in table_before_runs.field_names
    assert table_after_runs.field_names[-2:] == ["Recorded Episodes", "Success Rate"]
    assert [row[-2:] for row in table_after_runs.rows] == [[3, "0.5000"], [1, "n/a"]]
