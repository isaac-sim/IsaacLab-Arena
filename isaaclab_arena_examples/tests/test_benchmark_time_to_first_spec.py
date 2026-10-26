# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the time-to-first-environment-spec benchmark."""

from __future__ import annotations

import json
import subprocess

import pytest

import isaaclab_arena_examples.agentic_environment_generation.benchmark_time_to_first_spec as benchmark
import isaaclab_arena_examples.agentic_environment_generation.run_time_to_first_spec_benchmarks as batch_benchmark
import isaaclab_arena_examples.agentic_environment_generation.run_time_to_resolved_layout_benchmarks as resolved_layout_batch
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.tests.utils.agentic_environment_generation import minimal_spec_dict
from isaaclab_arena_examples.agentic_environment_generation.cli_runner import add_agent_inference_cli_args


def test_percentile_uses_nearest_rank():
    values = [float(value) for value in range(1, 101)]

    assert benchmark.percentile(values, 50) == 50.0
    assert benchmark.percentile(values, 95) == 95.0
    assert benchmark.percentile(values, 99) == 99.0
    assert benchmark.percentile([], 99) is None


def test_benchmark_reuses_cli_runner_inference_arguments():
    parser = benchmark.argparse.ArgumentParser()
    add_agent_inference_cli_args(parser, include_prompt=False)

    args = parser.parse_args(["--inference_endpoint", "openai", "--model", "gpt-test", "--temperature", "0.1"])

    assert args.inference_endpoint == "openai"
    assert args.model == "gpt-test"
    assert args.temperature == 0.1


def test_batch_command_forwards_case_and_inference_arguments(tmp_path):
    args = batch_benchmark.argparse.Namespace(
        cases_file=tmp_path / "cases.yaml",
        num_runs=3,
        temperature=0.1,
        inference_endpoint="internal",
        model="model-id",
        spec_output_dir=tmp_path / "specs",
    )

    command = batch_benchmark._benchmark_command(args, "case-name", tmp_path / "case-name.json")

    assert command[0] == batch_benchmark.sys.executable
    assert command[1] == str(batch_benchmark._BENCHMARK_SCRIPT)
    assert command[command.index("--case") + 1] == "case-name"
    assert command[command.index("--num_runs") + 1] == "3"
    assert command[command.index("--inference_endpoint") + 1] == "internal"
    assert command[command.index("--model") + 1] == "model-id"
    assert command[command.index("--spec_output_dir") + 1] == str(tmp_path / "specs" / "case-name")


def test_batch_no_save_specs_disables_generated_yaml_output(monkeypatch):
    monkeypatch.setattr(
        batch_benchmark.sys,
        "argv",
        ["run_time_to_first_spec_benchmarks.py", "--num_runs", "100", "--no_save_specs"],
    )

    args = batch_benchmark.parse_args()

    assert args.spec_output_dir is None


def test_batch_saves_specs_under_output_dir_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(
        batch_benchmark.sys,
        "argv",
        ["run_time_to_first_spec_benchmarks.py", "--num_runs", "1", "--output_dir", str(tmp_path)],
    )

    args = batch_benchmark.parse_args()

    assert args.spec_output_dir == tmp_path / "specs"


def test_batch_writes_only_combined_json(monkeypatch, tmp_path):
    cases_path = tmp_path / "cases.yaml"
    output_dir = tmp_path / "results"
    args = batch_benchmark.argparse.Namespace(
        cases_file=cases_path,
        num_runs=3,
        temperature=0.1,
        inference_endpoint="internal",
        model="model-id",
        output_dir=output_dir,
        spec_output_dir=tmp_path / "specs",
    )
    cases = {"first": benchmark.BenchmarkCase("first", "First prompt")}
    result = {
        "case": "first",
        "summary": {
            "successful_samples": 3,
            "failed_samples": 0,
            "p50_ms": 1.0,
            "p95_ms": 2.0,
            "p99_ms": 2.0,
        },
    }

    def fake_run(command, check):
        assert not check
        output_path = command[command.index("--output_path") + 1]
        batch_benchmark.Path(output_path).write_text(json.dumps(result), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(batch_benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(batch_benchmark, "load_benchmark_cases", lambda path: (cases, "first"))
    monkeypatch.setattr(batch_benchmark.subprocess, "run", fake_run)

    assert batch_benchmark.main() == 0
    assert [path.name for path in output_dir.iterdir()] == ["all_results.json"]


def test_run_sample_stops_after_generate_spec_returns(monkeypatch):
    spec = ArenaEnvGraphSpec.model_validate(minimal_spec_dict())

    class FakeAgent:
        traces: tuple[str, ...] = ()

        @staticmethod
        def generate_spec(prompt, *, asset_catalog, relation_catalog, task_catalog):
            assert prompt == "prompt"
            assert (asset_catalog, relation_catalog, task_catalog) == ("assets", "relations", "tasks")
            return spec, None

    times = iter((10.0, 10.25))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))

    result = benchmark._run_sample(
        FakeAgent(),
        "prompt",
        1,
        asset_catalog="assets",
        relation_catalog="relations",
        task_catalog="tasks",
    )

    assert result.time_to_first_spec_ms == pytest.approx(250.0)
    assert result.final_spec_accepted
    assert result.error is None


def test_run_sample_writes_first_generated_spec_after_measurement(monkeypatch, tmp_path):
    spec = ArenaEnvGraphSpec.model_validate(minimal_spec_dict())

    class FakeAgent:
        traces: tuple[str, ...] = ()

        @staticmethod
        def generate_spec(*args, **kwargs):
            return spec, None

    times = iter((10.0, 10.25))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    output_dir = tmp_path / "case"

    result = benchmark._run_sample(
        FakeAgent(),
        "prompt",
        1,
        asset_catalog="assets",
        relation_catalog="relations",
        task_catalog="tasks",
        spec_output_dir=output_dir,
    )

    output_path = output_dir / f"{spec.env_name}.yaml"
    assert result.time_to_first_spec_ms == pytest.approx(250.0)
    assert result.generated_spec_path == str(output_path)
    assert ArenaEnvGraphSpec.from_yaml(output_path) == spec


def test_every_benchmark_case_has_one_environment_spec_fixture():
    cases, default_case_name = benchmark.load_benchmark_cases(benchmark.DEFAULT_CASES_PATH)

    assert default_case_name in cases
    assert set(cases) == set(resolved_layout_batch.find_benchmark_cases(resolved_layout_batch._DEFAULT_ENV_SPECS_DIR))


def test_load_benchmark_cases_from_manually_defined_file(tmp_path):
    cases_path = tmp_path / "cases.yaml"
    cases_path.write_text(
        """\
default_case: first
cases:
  first:
    prompt: First prompt
  kitchen:
    prompt: Kitchen prompt
    object_references_expected: true
    enable_simready_search: true
    simready_sources: [service]
    simready_max_results_per_object: 2
""",
        encoding="utf-8",
    )

    cases, default_case_name = benchmark.load_benchmark_cases(cases_path)

    assert default_case_name == "first"
    assert cases == {
        "first": benchmark.BenchmarkCase("first", "First prompt"),
        "kitchen": benchmark.BenchmarkCase(
            "kitchen",
            "Kitchen prompt",
            object_references_expected=True,
            enable_simready_search=True,
            simready_sources=["service"],
            simready_max_results_per_object=2,
        ),
    }


def test_load_benchmark_cases_rejects_unknown_fields(tmp_path):
    cases_path = tmp_path / "cases.yaml"
    cases_path.write_text(
        "default_case: first\ncases:\n  first:\n    prompt: First prompt\n    simready_source: service\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="unknown fields"):
        benchmark.load_benchmark_cases(cases_path)


def test_load_benchmark_cases_rejects_unknown_default(tmp_path):
    cases_path = tmp_path / "cases.yaml"
    cases_path.write_text("default_case: missing\ncases:\n  first:\n    prompt: First prompt\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="default_case must name one"):
        benchmark.load_benchmark_cases(cases_path)


def test_run_sample_rejects_a_kitchen_result_that_skips_object_references(monkeypatch):
    spec = ArenaEnvGraphSpec.model_validate(minimal_spec_dict())

    class FakeAgent:
        traces: tuple[str, ...] = ()

        @staticmethod
        def generate_spec(*args, **kwargs):
            return spec, None

    times = iter((10.0, 10.25))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))

    result = benchmark._run_sample(
        FakeAgent(),
        "kitchen prompt",
        1,
        asset_catalog="assets",
        relation_catalog="relations",
        task_catalog="tasks",
        object_references_expected=True,
    )

    assert result.time_to_first_spec_ms is None
    assert result.final_spec_accepted
    assert result.error == "workload expected a spec with object_references, got one without them"


def test_summary_excludes_failed_samples_from_percentiles():
    results = [
        benchmark.SampleResult(1, 10.0, final_spec_accepted=True),
        benchmark.SampleResult(2, None, final_spec_accepted=False, error="no spec"),
        benchmark.SampleResult(3, 30.0, final_spec_accepted=False),
    ]

    assert benchmark._summary(results) == {
        "requested_samples": 3,
        "successful_samples": 2,
        "failed_samples": 1,
        "p50_ms": 10.0,
        "p95_ms": 30.0,
        "p99_ms": 30.0,
    }
