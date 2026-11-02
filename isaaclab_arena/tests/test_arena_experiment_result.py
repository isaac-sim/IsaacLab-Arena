# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for collecting the canonical Arena Experiment result."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from isaaclab_arena.assets.registries import EnvironmentRegistry, PolicyRegistry
from isaaclab_arena.environment_spec import arena_env_graph_yaml_loader
from isaaclab_arena.evaluation.arena_experiment_result import (
    ARENA_EXPERIMENT_RESULT_FILENAME,
    ArenaExperimentResult,
    all_runs_completed,
    build_arena_run_result_metadata,
    get_benchmark_id,
    get_git_stamp,
    get_latest_run_end_timestamp_ns,
    read_experiment_result,
)
from isaaclab_arena.utils.git_stamp import GitStamp

START_TIMESTAMP_NS = 1792799000000000000
END_TIMESTAMP_NS = 1792800000000000000
BENCHMARK_ID = "pnp_benchmark"


def _metadata(
    environment_name: str,
    *,
    policy_variant: str = "pi05",
    status: str = "completed",
) -> dict:
    return {
        "environment": {
            "name": environment_name,
            "definition": f"environments/{environment_name}.yaml",
        },
        "policy_variant": policy_variant,
        "status": status,
        "start_timestamp_ns": START_TIMESTAMP_NS,
        "end_timestamp_ns": END_TIMESTAMP_NS,
    }


def _write_jsonl(path: Path, records: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")


def test_collects_declared_runs_rebuilds_and_rank_files_without_changing_episode_objects(tmp_path, monkeypatch):
    rank_zero_records = [
        {
            "env_id": 7,
            "episode_in_env": 3,
            "success": True,
            "custom": {"score": 0.75, "labels": ["kept", "raw"]},
        },
        {"anything": "is preserved", "nested": [{"value": None}]},
    ]
    rank_one_records = [{"rank": 1, "line": 1}]
    _write_jsonl(tmp_path / "completed/episode_results_rebuild1_rank1.jsonl", rank_one_records)
    _write_jsonl(tmp_path / "completed/episode_results_rebuild1_rank0.jsonl", rank_zero_records)
    _write_jsonl(tmp_path / "completed/nested/episode_results_rebuild0.jsonl", [{"rebuild": 0}])
    _write_jsonl(tmp_path / "completed/episode_results_rebuild2.jsonl", [])
    _write_jsonl(tmp_path / "failed/episode_results.jsonl", [{"partial": True}])
    _write_jsonl(tmp_path / "undeclared/episode_results.jsonl", [{"ignored": True}])

    monkeypatch.delenv("ARENA_GIT_SHA", raising=False)
    monkeypatch.delenv("ARENA_GIT_BRANCH", raising=False)
    result = ArenaExperimentResult(
        tmp_path,
        {
            "completed": _metadata("completed_task"),
            "failed": _metadata("failed_task", status="failed"),
        },
        benchmark_id=BENCHMARK_ID,
    )

    assert result.to_dict() == {
        "git_stamp": {"git_sha": "unknown", "git_branch": "unknown"},
        "benchmark_id": BENCHMARK_ID,
        "runs": {
            "completed": {
                "environment": {
                    "name": "completed_task",
                    "definition": "environments/completed_task.yaml",
                },
                "policy_variant": "pi05",
                "status": "completed",
                "start_timestamp_ns": START_TIMESTAMP_NS,
                "end_timestamp_ns": END_TIMESTAMP_NS,
                "rebuilds": [
                    {"index": 0, "episodes": [{"rebuild": 0}]},
                    {"index": 1, "episodes": [*rank_zero_records, *rank_one_records]},
                    {"index": 2, "episodes": []},
                ],
            },
            "failed": {
                "environment": {
                    "name": "failed_task",
                    "definition": "environments/failed_task.yaml",
                },
                "policy_variant": "pi05",
                "status": "failed",
                "start_timestamp_ns": START_TIMESTAMP_NS,
                "end_timestamp_ns": END_TIMESTAMP_NS,
                "rebuilds": [{"index": 0, "episodes": [{"partial": True}]}],
            },
        },
    }

    result_path = result.write()
    assert result_path == tmp_path / ARENA_EXPERIMENT_RESULT_FILENAME
    assert json.loads(result_path.read_text(encoding="utf-8")) == result.to_dict()


def test_result_records_git_stamp_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_GIT_SHA", "abc123")
    monkeypatch.setenv("ARENA_GIT_BRANCH", "main")

    result = ArenaExperimentResult(tmp_path, {"run": _metadata("task", status="failed")}, benchmark_id=BENCHMARK_ID)

    assert result.to_dict()["git_stamp"] == {"git_sha": "abc123", "git_branch": "main"}


@pytest.mark.parametrize("value", [None, ""])
def test_git_stamp_is_unknown_when_unset_or_empty(tmp_path, monkeypatch, value):
    for variable in ("ARENA_GIT_SHA", "ARENA_GIT_BRANCH"):
        if value is None:
            monkeypatch.delenv(variable, raising=False)
        else:
            monkeypatch.setenv(variable, value)

    result = ArenaExperimentResult(tmp_path, {"run": _metadata("task", status="failed")}, benchmark_id=BENCHMARK_ID)

    assert result.to_dict()["git_stamp"] == {"git_sha": "unknown", "git_branch": "unknown"}


def test_result_records_given_git_stamp(tmp_path):
    git_stamp = GitStamp(git_sha="abc123-dirty", git_branch="feature")

    result = ArenaExperimentResult(
        tmp_path, {"run": _metadata("task", status="failed")}, git_stamp, benchmark_id=BENCHMARK_ID
    )

    assert result.to_dict()["git_stamp"] == {"git_sha": "abc123-dirty", "git_branch": "feature"}
    assert GitStamp.from_dict(result.to_dict()["git_stamp"]) == git_stamp


@pytest.mark.parametrize(
    "git_stamp",
    [
        {"git_sha": "abc123"},
        {"git_sha": "abc123", "git_branch": "main", "extra": "x"},
    ],
)
def test_git_stamp_rejects_unexpected_keys(git_stamp):
    with pytest.raises(AssertionError, match="git stamp"):
        GitStamp.from_dict(git_stamp)


def test_run_metadata_requires_integer_start_timestamp(tmp_path):
    metadata = _metadata("task", status="failed")
    metadata["start_timestamp_ns"] = "1792799000000000000"

    with pytest.raises(AssertionError, match="start_timestamp_ns for Run 'run' must be an int"):
        ArenaExperimentResult(tmp_path, {"run": metadata}, benchmark_id=BENCHMARK_ID)


def test_run_metadata_requires_integer_end_timestamp(tmp_path):
    metadata = _metadata("task", status="failed")
    metadata["end_timestamp_ns"] = "1792800000000000000"

    with pytest.raises(AssertionError, match="end_timestamp_ns for Run 'run' must be an int"):
        ArenaExperimentResult(tmp_path, {"run": metadata}, benchmark_id=BENCHMARK_ID)


def test_failed_runs_may_have_empty_or_missing_output_directories(tmp_path):
    (tmp_path / "empty").mkdir()

    result = ArenaExperimentResult(
        tmp_path,
        {
            "empty": _metadata("task", status="failed"),
            "missing": _metadata("task", status="failed"),
        },
        benchmark_id=BENCHMARK_ID,
    )

    assert result.to_dict()["runs"]["empty"]["rebuilds"] == []
    assert result.to_dict()["runs"]["missing"]["rebuilds"] == []


def test_completed_run_requires_an_output_directory(tmp_path):
    with pytest.raises(AssertionError, match="completed Run 'missing' is missing its output directory"):
        ArenaExperimentResult(tmp_path, {"missing": _metadata("task")}, benchmark_id=BENCHMARK_ID)


def test_malformed_and_nonobject_jsonl_records_fail_collection(tmp_path):
    run_output_directory = tmp_path / "run"
    run_output_directory.mkdir()
    (run_output_directory / "episode_results.jsonl").write_text(
        '{"valid": true}\n{bad json\n[1, 2, 3]\n',
        encoding="utf-8",
    )

    with pytest.raises(AssertionError) as error:
        ArenaExperimentResult(tmp_path, {"run": _metadata("task")}, benchmark_id=BENCHMARK_ID)

    assert "run/episode_results.jsonl: line 2: invalid JSON" in str(error.value)
    assert "run/episode_results.jsonl: line 3: expected a JSON object" in str(error.value)


@pytest.mark.parametrize("run_name", ["", " ", "../outside", "nested/run", "nested\\run", ".", "C:run"])
def test_rejects_unsafe_run_paths(tmp_path, run_name):
    with pytest.raises(AssertionError):
        ArenaExperimentResult(tmp_path, {run_name: _metadata("task", status="failed")}, benchmark_id=BENCHMARK_ID)


def test_run_named_like_timing_totals_is_rejected_up_front():
    with pytest.raises(AssertionError, match="'all' is reserved"):
        ArenaExperimentResult.assert_run_name_is_safe_path_component("all")


@dataclass
class _EnvironmentCfg:
    pass


@dataclass
class _GraphEnvironmentCfg:
    env_spec_path: str


@dataclass
class _PolicyCfg:
    policy_variant: object = None


@dataclass
class _RunCfg:
    environment: object
    policy: object


def test_builds_graph_environment_metadata_and_preserves_an_explicit_policy_variant(monkeypatch):
    monkeypatch.setattr(
        arena_env_graph_yaml_loader,
        "load_env_graph_spec_dict",
        lambda path: {"env_name": "banana_in_bowl"},
    )
    run_cfg = _RunCfg(
        environment=_GraphEnvironmentCfg("configs/../tasks/banana_in_bowl.yaml"),
        policy=_PolicyCfg(policy_variant="Pi0.5-Exact"),
    )

    assert build_arena_run_result_metadata(run_cfg) == {
        "environment": {
            "name": "banana_in_bowl",
            "definition": "configs/../tasks/banana_in_bowl.yaml",
        },
        "policy_variant": "Pi0.5-Exact",
    }


def test_builds_registered_environment_and_policy_metadata(monkeypatch):
    class _EnvironmentFactory:
        name = "registered_environment"

    class _Policy:
        name = "registered_policy"

    monkeypatch.setattr(
        EnvironmentRegistry,
        "get_factory_type_for_cfg",
        lambda registry, environment_cfg: _EnvironmentFactory,
    )
    monkeypatch.setattr(
        PolicyRegistry,
        "get_policy_type_for_cfg",
        lambda registry, policy_cfg: _Policy,
    )

    assert build_arena_run_result_metadata(_RunCfg(_EnvironmentCfg(), _PolicyCfg())) == {
        "environment": {
            "name": "registered_environment",
            "definition": "registered_environment",
        },
        "policy_variant": "registered_policy",
    }


def _write_result(output_dir: Path, runs: dict, **extra) -> None:
    result = {
        "git_stamp": {"git_sha": "abc1234", "git_branch": "main"},
        "benchmark_id": BENCHMARK_ID,
        "runs": runs,
        **extra,
    }
    (output_dir / ARENA_EXPERIMENT_RESULT_FILENAME).write_text(json.dumps(result), encoding="utf-8")


def _run(status: str = "completed", end_timestamp_ns: int = END_TIMESTAMP_NS) -> dict:
    return {**_metadata("env", status=status), "end_timestamp_ns": end_timestamp_ns, "rebuilds": []}


def test_read_experiment_result_rejects_missing_end_timestamp(tmp_path):
    run = _run()
    del run["end_timestamp_ns"]
    _write_result(tmp_path, {"run_a": run})

    with pytest.raises(AssertionError, match="no integer end_timestamp_ns"):
        read_experiment_result(tmp_path)


def test_get_latest_run_end_timestamp_ns_is_the_maximum(tmp_path):
    _write_result(tmp_path, {"run_a": _run(end_timestamp_ns=5), "run_b": _run(end_timestamp_ns=9)})

    assert get_latest_run_end_timestamp_ns(read_experiment_result(tmp_path)) == 9


def test_all_runs_completed_is_false_with_one_failed_run(tmp_path):
    _write_result(tmp_path, {"run_a": _run(), "run_b": _run(status="failed")})
    assert not all_runs_completed(read_experiment_result(tmp_path))

    _write_result(tmp_path, {"run_a": _run()})
    assert all_runs_completed(read_experiment_result(tmp_path))


def test_get_git_stamp_builds_stamp_from_result(tmp_path):
    _write_result(tmp_path, {"run_a": _run()})

    assert get_git_stamp(read_experiment_result(tmp_path)) == GitStamp(git_sha="abc1234", git_branch="main")


def test_get_benchmark_id_reads_id_from_result(tmp_path):
    _write_result(tmp_path, {"run_a": _run()})

    assert get_benchmark_id(read_experiment_result(tmp_path)) == BENCHMARK_ID


def test_result_round_trip_keeps_benchmark_id(tmp_path):
    ArenaExperimentResult(tmp_path, {"run": _metadata("task", status="failed")}, benchmark_id=BENCHMARK_ID).write()

    result = read_experiment_result(tmp_path)

    assert result["benchmark_id"] == BENCHMARK_ID
    assert get_benchmark_id(result) == BENCHMARK_ID


@pytest.mark.parametrize("benchmark_id", ["", "a\nb", None, 3])
def test_result_rejects_invalid_benchmark_id(tmp_path, benchmark_id):
    with pytest.raises(AssertionError, match="benchmark_id"):
        ArenaExperimentResult(tmp_path, {"run": _metadata("task", status="failed")}, benchmark_id=benchmark_id)


@pytest.mark.parametrize("benchmark_id", ["", "a\nb", 3])
def test_read_experiment_result_rejects_invalid_benchmark_id(tmp_path, benchmark_id):
    _write_result(tmp_path, {"run_a": _run()}, benchmark_id=benchmark_id)

    with pytest.raises(AssertionError, match="benchmark_id"):
        read_experiment_result(tmp_path)


def test_read_experiment_result_rejects_missing_benchmark_id(tmp_path):
    result = {"git_stamp": {"git_sha": "abc1234", "git_branch": "main"}, "runs": {"run_a": _run()}}
    (tmp_path / ARENA_EXPERIMENT_RESULT_FILENAME).write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(AssertionError, match="no benchmark_id"):
        read_experiment_result(tmp_path)
