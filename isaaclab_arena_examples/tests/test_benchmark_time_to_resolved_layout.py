# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the time-to-resolved-layout benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from contextlib import nullcontext
from types import ModuleType

import pytest

import isaaclab_arena_examples.agentic_environment_generation.benchmark.benchmark_time_to_resolved_layout as benchmark
import isaaclab_arena_examples.agentic_environment_generation.benchmark.run_time_to_resolved_layout_benchmarks as batch_benchmark


def test_checkpoint_rejects_changed_environment_spec(tmp_path):
    env_spec = tmp_path / "environment.yaml"
    env_spec.write_text("env_name: original\n", encoding="utf-8")
    output_path = tmp_path / "result.json"
    args = argparse.Namespace(
        resume=True,
        output_path=output_path,
        env_spec=env_spec,
        placement_seed=42,
        warmup_runs=1,
        num_runs=3,
    )
    results = {"1": {"summary": {}, "samples": []}}
    output = benchmark._output_payload(args, results)
    output_path.write_text(json.dumps(output), encoding="utf-8")

    assert benchmark._load_checkpoint(args) == results
    assert batch_benchmark._validate_checkpoint(output, args, env_spec, 1) == []

    env_spec.write_text("env_name: changed\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="contents do not match"):
        benchmark._load_checkpoint(args)


@pytest.mark.parametrize(
    ("argument", "incompatible_value", "error"),
    (
        ("placement_seed", 7, "placement seed"),
        ("warmup_runs", 2, "warmup count"),
        ("num_runs", 4, "requested sample count"),
    ),
)
def test_checkpoint_rejects_incompatible_arguments(tmp_path, argument, incompatible_value, error):
    env_spec = tmp_path / "environment.yaml"
    env_spec.write_text("env_name: test\n", encoding="utf-8")
    output_path = tmp_path / "result.json"
    args = argparse.Namespace(
        resume=True,
        output_path=output_path,
        env_spec=env_spec,
        placement_seed=42,
        warmup_runs=1,
        num_runs=3,
    )
    output = benchmark._output_payload(args, {})
    output_path.write_text(json.dumps(output), encoding="utf-8")
    setattr(args, argument, incompatible_value)

    with pytest.raises(AssertionError, match=error):
        benchmark._load_checkpoint(args)
    with pytest.raises(AssertionError, match=error):
        batch_benchmark._validate_checkpoint(output, args, env_spec, 1)


def test_batch_command_forwards_benchmark_arguments(tmp_path):
    args = argparse.Namespace(
        num_runs=100,
        warmup_runs=1,
        placement_seed=42,
        resume=True,
    )

    command = batch_benchmark._command(args, tmp_path / "environment.yaml", 64, tmp_path / "result.json")

    assert command[0] == batch_benchmark.sys.executable
    expected_arguments = {
        "--num_envs": "64",
        "--num_runs": "100",
        "--warmup_runs": "1",
        "--placement_seed": "42",
    }
    for argument, expected_value in expected_arguments.items():
        assert command[command.index(argument) + 1] == expected_value
    assert command[-1] == "--resume"


def test_failed_sample_makes_single_benchmark_exit_nonzero(monkeypatch, tmp_path):
    env_spec = tmp_path / "environment.yaml"
    env_spec.write_text("env_name: test\n", encoding="utf-8")
    args = argparse.Namespace(
        env_spec=env_spec,
        num_envs=[1],
        num_runs=1,
        warmup_runs=0,
        placement_seed=42,
        output_path=tmp_path / "result.json",
        resume=False,
    )

    def fake_simulation_app_context(_args):
        return nullcontext()

    simulation_app = ModuleType("isaaclab_arena.utils.isaaclab_utils.simulation_app")
    simulation_app.SimulationAppContext = fake_simulation_app_context
    monkeypatch.setitem(sys.modules, simulation_app.__name__, simulation_app)
    monkeypatch.setattr(benchmark, "_parse_args", lambda: args)
    monkeypatch.setattr(
        benchmark,
        "_run_sample",
        lambda *_args: benchmark.SampleResult(1, 0, 0, None, "failure"),
    )

    with pytest.raises(SystemExit) as exit_info:
        benchmark.main()
    assert exit_info.value.code == 1


def test_batch_marks_failed_samples_as_failed_configuration(monkeypatch, tmp_path):
    env_specs_dir = tmp_path / "env_specs"
    env_specs_dir.mkdir()
    (env_specs_dir / "test_case.yaml").write_text("env_name: test_case\n", encoding="utf-8")
    output_dir = tmp_path / "results"
    args = argparse.Namespace(
        env_specs_dir=env_specs_dir,
        case=["test_case"],
        num_envs=[1],
        num_runs=1,
        warmup_runs=0,
        placement_seed=42,
        output_dir=output_dir,
        resume=False,
    )

    def fake_run(command, check):
        assert not check
        output_path = batch_benchmark.Path(command[command.index("--output_path") + 1])
        result = {
            "results_by_num_envs": {
                "1": {
                    "summary": {
                        "successful_samples": 0,
                        "failed_samples": 1,
                        "p50_ms": None,
                        "p95_ms": None,
                        "p99_ms": None,
                    },
                }
            },
        }
        output_path.write_text(json.dumps(result), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(batch_benchmark, "_parse_args", lambda: args)
    monkeypatch.setattr(batch_benchmark.subprocess, "run", fake_run)

    assert batch_benchmark.main() == 1
    combined = json.loads((output_dir / "all_results.json").read_text(encoding="utf-8"))
    assert combined["failed_configurations"] == ["test_case_envs_1"]


def test_batch_ignores_result_left_by_an_earlier_run(monkeypatch, tmp_path):
    env_specs_dir = tmp_path / "env_specs"
    env_specs_dir.mkdir()
    (env_specs_dir / "test_case.yaml").write_text("env_name: test_case\n", encoding="utf-8")
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    (output_dir / "test_case_envs_1.json").write_text(json.dumps({"results_by_num_envs": {}}), encoding="utf-8")
    args = argparse.Namespace(
        env_specs_dir=env_specs_dir,
        case=["test_case"],
        num_envs=[1],
        num_runs=1,
        warmup_runs=0,
        placement_seed=42,
        output_dir=output_dir,
        resume=False,
    )

    monkeypatch.setattr(batch_benchmark, "_parse_args", lambda: args)
    monkeypatch.setattr(
        batch_benchmark.subprocess, "run", lambda command, check: subprocess.CompletedProcess(command, 1)
    )

    assert batch_benchmark.main() == 1
    combined = json.loads((output_dir / "all_results.json").read_text(encoding="utf-8"))
    assert combined["failed_configurations"] == ["test_case_envs_1"]
    assert combined["results"] == []


def _test_run_sample_resolves_a_benchmark_spec(simulation_app):
    sample = benchmark._run_sample(
        batch_benchmark._DEFAULT_ENV_SPECS_DIR / "tabletop_banana_plate_distractors_0.yaml",
        num_envs=1,
        run_index=0,
        placement_seed=42,
    )
    assert sample.error is None, sample.error
    assert sample.time_to_resolved_layout_ms is not None
    assert sample.requested_layouts > 0
    return True


def test_run_sample_resolves_a_benchmark_spec():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_run_sample_resolves_a_benchmark_spec)


def test_find_benchmark_cases_names_each_case_after_its_spec_file(tmp_path):
    (tmp_path / "case_a.yaml").write_text("env_name: case_a\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("", encoding="utf-8")

    assert batch_benchmark.find_benchmark_cases(tmp_path) == {"case_a": tmp_path / "case_a.yaml"}
