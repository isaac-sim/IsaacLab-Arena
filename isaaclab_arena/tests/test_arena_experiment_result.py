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
    build_arena_run_result_metadata,
)


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
    }


def _write_jsonl(path: Path, records: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")


def test_collects_declared_runs_rebuilds_and_rank_files_without_changing_episode_objects(tmp_path):
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

    result = ArenaExperimentResult(
        tmp_path,
        {
            "completed": _metadata("completed_task"),
            "failed": _metadata("failed_task", status="failed"),
        },
    )

    assert result.to_dict() == {
        "runs": {
            "completed": {
                "environment": {
                    "name": "completed_task",
                    "definition": "environments/completed_task.yaml",
                },
                "policy_variant": "pi05",
                "status": "completed",
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
                "rebuilds": [{"index": 0, "episodes": [{"partial": True}]}],
            },
        },
    }

    result_path = result.write()
    assert result_path == tmp_path / ARENA_EXPERIMENT_RESULT_FILENAME
    assert json.loads(result_path.read_text(encoding="utf-8")) == result.to_dict()


def test_failed_runs_may_have_empty_or_missing_output_directories(tmp_path):
    (tmp_path / "empty").mkdir()

    result = ArenaExperimentResult(
        tmp_path,
        {
            "empty": _metadata("task", status="failed"),
            "missing": _metadata("task", status="failed"),
        },
    )

    assert result.to_dict()["runs"]["empty"]["rebuilds"] == []
    assert result.to_dict()["runs"]["missing"]["rebuilds"] == []


def test_completed_run_requires_an_output_directory(tmp_path):
    with pytest.raises(AssertionError, match="completed Run 'missing' is missing its output directory"):
        ArenaExperimentResult(tmp_path, {"missing": _metadata("task")})


def test_malformed_and_nonobject_jsonl_records_fail_collection(tmp_path):
    run_output_directory = tmp_path / "run"
    run_output_directory.mkdir()
    (run_output_directory / "episode_results.jsonl").write_text(
        '{"valid": true}\n{bad json\n[1, 2, 3]\n',
        encoding="utf-8",
    )

    with pytest.raises(AssertionError) as error:
        ArenaExperimentResult(tmp_path, {"run": _metadata("task")})

    assert "run/episode_results.jsonl: line 2: invalid JSON" in str(error.value)
    assert "run/episode_results.jsonl: line 3: expected a JSON object" in str(error.value)


@pytest.mark.parametrize("run_name", ["", " ", "../outside", "nested/run", "nested\\run", ".", "C:run"])
def test_rejects_unsafe_run_paths(tmp_path, run_name):
    with pytest.raises(AssertionError):
        ArenaExperimentResult(tmp_path, {run_name: _metadata("task", status="failed")})


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
