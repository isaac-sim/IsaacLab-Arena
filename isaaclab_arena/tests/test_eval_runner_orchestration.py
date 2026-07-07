# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test experiment orchestration independently of Isaac Sim execution."""

from dataclasses import dataclass

import pytest

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation import eval_runner
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg, ArenaRunResult, RolloutLimitCfg, RunStatus
from isaaclab_arena.policy.policy_base import PolicyCfg


@dataclass
class _EnvironmentCfg(ArenaEnvironmentCfg):
    pass


@dataclass
class _PolicyCfg(PolicyCfg):
    pass


def _run(name: str) -> ArenaRunCfg:
    return ArenaRunCfg(
        name=name,
        environment=_EnvironmentCfg(),
        policy=_PolicyCfg(),
        rollout_limit=RolloutLimitCfg(num_steps=2),
    )


def test_execute_experiment_runs_in_declaration_order(monkeypatch, tmp_path):
    received = []

    def build_and_run(run_cfg, output_dir, video_cfg):
        received.append((run_cfg.name, output_dir, video_cfg.video_base_dir))
        return ArenaRunResult(run_name=run_cfg.name, status=RunStatus.COMPLETED)

    monkeypatch.setattr(eval_runner, "build_and_run", build_and_run)

    results = eval_runner.execute_experiment([_run("first"), _run("second")], output_dir=tmp_path)

    assert [result.run_name for result in results] == ["first", "second"]
    assert received == [
        ("first", str(tmp_path / "first"), str(tmp_path / "first")),
        ("second", str(tmp_path / "second"), str(tmp_path / "second")),
    ]


def test_execute_experiment_records_failure_and_continues(monkeypatch, tmp_path):
    attempted = []

    def build_and_run(run_cfg, output_dir, video_cfg):
        attempted.append(run_cfg.name)
        if run_cfg.name == "failing":
            raise RuntimeError("rollout failed")
        return ArenaRunResult(run_name=run_cfg.name, status=RunStatus.COMPLETED)

    monkeypatch.setattr(eval_runner, "build_and_run", build_and_run)

    results = eval_runner.execute_experiment(
        [_run("failing"), _run("passing")],
        output_dir=tmp_path,
        continue_on_error=True,
    )

    assert attempted == ["failing", "passing"]
    assert [(result.run_name, result.status) for result in results] == [
        ("failing", RunStatus.FAILED),
        ("passing", RunStatus.COMPLETED),
    ]


def test_execute_experiment_stops_on_failure_by_default(monkeypatch, tmp_path):
    attempted = []

    def build_and_run(run_cfg, output_dir, video_cfg):
        attempted.append(run_cfg.name)
        raise RuntimeError("rollout failed")

    monkeypatch.setattr(eval_runner, "build_and_run", build_and_run)

    with pytest.raises(RuntimeError, match="rollout failed"):
        eval_runner.execute_experiment([_run("failing"), _run("not_attempted")], output_dir=tmp_path)

    assert attempted == ["failing"]
