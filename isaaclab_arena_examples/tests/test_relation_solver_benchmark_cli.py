# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the relation benchmark example CLI and distributed protocol."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

import isaaclab_arena.relations.benchmark.multi_gpu as benchmark_multi_gpu
from isaaclab_arena.relations.benchmark.models import BenchmarkScenario
from isaaclab_arena.relations.benchmark.multi_gpu import run_multi_gpu, validate_gpu_selectors
from isaaclab_arena.relations.benchmark.reporting import build_distributed_run, requested_scenario_ids
from isaaclab_arena.relations.benchmark.synthetic_benchmark import run_solver_benchmark
from isaaclab_arena_examples.relations import relation_solver_benchmark as benchmark_cli


def _tiny_scenario(**overrides) -> BenchmarkScenario:
    values = {
        "name": "tiny",
        "num_objects": 3,
        "num_envs": 1,
        "max_iters": 1,
        "max_placement_attempts": 1,
        "warmup_runs": 0,
        "timed_runs": 1,
        "final_loss_threshold": 1e9,
    }
    values.update(overrides)
    return BenchmarkScenario(**values)


class _SuccessfulProcess:
    def poll(self):
        return 0

    def terminate(self):
        raise AssertionError("completed process must not be terminated")

    def wait(self):
        return 0


def test_cli_defaults_do_not_write_reports():
    args = benchmark_cli._parse_args([])
    assert args.output_dir is None
    assert args.targets is None


def test_environment_cli_honors_explicit_collision_mode():
    args = benchmark_cli._parse_args([
        "--targets",
        "environment",
        "--collision-mode",
        "bbox",
        "--environment-spec",
        benchmark_cli.DEFAULT_MESH_SPEC,
        "--num-envs",
        "1",
    ])
    scenarios = benchmark_cli._environment_scenarios(args)
    assert {scenario.collision_mode for scenario in scenarios} == {"bbox"}


def test_custom_environment_requires_explicit_collision_mode():
    args = benchmark_cli._parse_args([
        "--targets",
        "environment",
        "--environment-spec",
        "custom.yaml",
    ])
    with pytest.raises(ValueError, match="requires --collision-mode"):
        benchmark_cli._environment_scenarios(args)


def test_environment_cli_expands_mode_robot_matrix():
    args = benchmark_cli._parse_args([
        "--targets",
        "environment",
        "--compare-modes",
        "--robot-mode",
        "both",
        "--num-envs",
        "1",
    ])
    benchmark_cli._validate_args(args)
    scenarios = benchmark_cli._environment_scenarios(args)
    assert len(scenarios) == 8
    assert {scenario.collision_mode for scenario in scenarios} == {"bbox", "mesh"}
    assert {scenario.include_robot for scenario in scenarios} == {True, False}


@pytest.mark.parametrize(
    "argv",
    [
        ["--targets", "environment", "--collision-mode", "bbox"],
        ["--suite", "comprehensive"],
    ],
    ids=("environment", "comprehensive"),
)
def test_environment_cli_reports_before_simulation_teardown(monkeypatch, argv):
    import isaaclab_arena.utils.isaaclab_utils.simulation_app as simulation_app

    lifecycle = []

    class _SimulationContext:
        def __init__(self, args):
            pass

        def __enter__(self):
            lifecycle.append("enter")
            return self

        def __exit__(self, *args):
            lifecycle.append("exit")

    def finish_run(*args):
        lifecycle.append("report")
        return 0

    def run_benchmarks(*args, **kwargs):
        assert lifecycle == ["enter"]
        return []

    monkeypatch.setattr(simulation_app, "SimulationAppContext", _SimulationContext)
    monkeypatch.setattr(benchmark_cli.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark_cli, "run_benchmarks", run_benchmarks)
    monkeypatch.setattr(benchmark_cli, "build_run", lambda *args, **kwargs: object())
    monkeypatch.setattr(benchmark_cli, "build_distributed_run", lambda *args, **kwargs: object())
    monkeypatch.setattr(benchmark_cli, "_finish_run", finish_run)

    assert benchmark_cli.main(argv) == 0
    assert lifecycle == ["enter", "report", "exit"]


def test_comprehensive_plan_has_separate_unique_matrices_and_honors_knobs():
    args = benchmark_cli._parse_args([
        "--suite",
        "comprehensive",
        "--max-iters",
        "17",
        "--convergence-threshold",
        "0.25",
        "--num-spheres",
        "9",
        "--seed",
        "4",
        "--warmup",
        "0",
        "--repeat",
        "2",
    ])
    benchmark_cli._validate_args(args)
    synthetic, environment = benchmark_cli._comprehensive_scenario_groups(args)

    assert len(synthetic) == 12
    assert {(scenario.num_objects, scenario.num_envs, scenario.collision_mode) for scenario in synthetic} == {
        (num_objects, num_envs, mode)
        for num_objects, num_envs in ((3, 1), (5, 1), (10, 1), (3, 8), (3, 32), (3, 128))
        for mode in ("bbox", "mesh")
    }
    assert all(
        (
            scenario.max_iters,
            scenario.convergence_threshold,
            scenario.num_spheres,
            scenario.placement_seed,
            scenario.warmup_runs,
            scenario.timed_runs,
        )
        == (17, 0.25, 9, 4, 0, 2)
        for scenario in synthetic
    )
    assert len(environment) == 8
    assert {scenario.graph_spec_path for scenario in environment} == {
        benchmark_cli.DEFAULT_BBOX_SPEC,
        benchmark_cli.DEFAULT_MESH_SPEC,
    }
    assert {scenario.collision_mode for scenario in environment} == {"bbox", "mesh"}
    assert {scenario.include_robot for scenario in environment} == {True, False}
    assert {scenario.num_envs for scenario in environment} == {1}
    expected = requested_scenario_ids(synthetic, ("solver", "placer")) + requested_scenario_ids(
        environment,
        ("environment",),
    )
    assert len(expected) == len(set(expected)) == 32


@pytest.mark.parametrize(
    "option",
    [
        ["--gpus", "0"],
        ["--capacity-search"],
        ["--targets", "solver"],
        ["--environment-spec", "custom.yaml"],
        ["--collision-mode", "bbox"],
        ["--compare-modes"],
        ["--robot-mode", "with"],
        ["--num-envs", "1"],
        ["--capacity-max-envs", "128"],
        ["--memory-headroom-gib", "1"],
    ],
)
def test_comprehensive_rejects_matrix_overrides(option):
    args = benchmark_cli._parse_args(["--suite", "comprehensive", *option])
    with pytest.raises(ValueError, match="owns its execution matrix"):
        benchmark_cli._validate_args(args)


def test_comprehensive_requires_cuda(monkeypatch, capsys):
    monkeypatch.setattr(benchmark_cli.torch.cuda, "is_available", lambda: False)
    assert benchmark_cli.main(["--suite", "comprehensive"]) == 2
    assert "requires a CUDA GPU" in capsys.readouterr().err


def test_object_suite_keeps_object_sweep_with_num_envs_override():
    args = benchmark_cli._parse_args(["--suite", "objects", "--num-envs", "4"])
    scenarios = benchmark_cli._base_scenarios(args)
    assert {scenario.num_objects for scenario in scenarios} == {3, 5, 6, 10}
    assert {scenario.num_envs for scenario in scenarios} == {4}


def test_multi_gpu_results_are_complete_and_aggregated(monkeypatch, tmp_path):
    scenario = _tiny_scenario()
    base_result = run_solver_benchmark(scenario)

    def launch_worker(scenarios, targets, physical_gpu, directory, worker_id, script_path):
        output_path = tmp_path / f"{worker_id}.json"
        result = base_result.to_dict()
        result["throughput_envs_per_second"] = 10.0 + int(physical_gpu)
        output_path.write_text(json.dumps([result]), encoding="utf-8")
        return _SuccessfulProcess(), output_path

    monkeypatch.setattr(benchmark_multi_gpu, "_launch_worker", launch_worker)
    rows, assignments, exit_codes, worker_errors = run_multi_gpu(
        (scenario,),
        ("solver",),
        ("0", "1"),
        Path(benchmark_cli.__file__).resolve(),
    )

    assert len(rows) == 2
    assert len({row.scenario_id for row in rows}) == 2
    assert {row.worker_id for row in rows} == {"gpu-0", "gpu-1"}
    assert {row.aggregate_throughput_envs_per_second for row in rows} == {21.0}
    assert all(len(ids) == 1 for ids in assignments.values())
    assert exit_codes == {"gpu-0": 0, "gpu-1": 0}
    assert worker_errors == {}


def test_multi_gpu_records_worker_that_writes_no_output(monkeypatch):
    scenario = _tiny_scenario()

    def launch_worker(scenarios, targets, physical_gpu, directory, worker_id, script_path):
        return _SuccessfulProcess(), directory / "missing.json"

    monkeypatch.setattr(benchmark_multi_gpu, "_launch_worker", launch_worker)
    rows, assignments, exit_codes, worker_errors = run_multi_gpu(
        (scenario,),
        ("solver",),
        ("0",),
        Path(benchmark_cli.__file__).resolve(),
    )
    run = build_distributed_run(rows, assignments, exit_codes, worker_errors)

    assert rows == []
    assert not run.succeeded
    assert run.missing_scenario_ids == tuple(assignments["gpu-0"])
    assert "without writing" in worker_errors["gpu-0"]


def test_multi_gpu_omits_partial_aggregate_throughput(monkeypatch):
    scenario = _tiny_scenario()
    base_result = run_solver_benchmark(scenario)

    def launch_worker(scenarios, targets, physical_gpu, directory, worker_id, script_path):
        output_path = directory / f"{worker_id}.json"
        if physical_gpu == "0":
            output_path.write_text(json.dumps([base_result.to_dict()]), encoding="utf-8")
        return _SuccessfulProcess(), output_path

    monkeypatch.setattr(benchmark_multi_gpu, "_launch_worker", launch_worker)
    rows, _, _, worker_errors = run_multi_gpu(
        (scenario,),
        ("solver",),
        ("0", "1"),
        Path(benchmark_cli.__file__).resolve(),
    )
    assert len(rows) == 1
    assert rows[0].aggregate_throughput_envs_per_second is None
    assert "gpu-1" in worker_errors


def test_multi_gpu_cleans_up_started_workers_when_launch_fails(monkeypatch):
    scenario = _tiny_scenario()

    class _RunningProcess:
        terminated = False
        reaped = False

        def poll(self):
            return 0 if self.reaped else None

        def terminate(self):
            self.terminated = True

        def wait(self):
            assert self.terminated
            self.reaped = True
            return 0

    process = _RunningProcess()

    def launch_worker(scenarios, targets, physical_gpu, directory, worker_id, script_path):
        if physical_gpu == "1":
            raise RuntimeError("launch failed")
        return process, directory / "unused.json"

    monkeypatch.setattr(benchmark_multi_gpu, "_launch_worker", launch_worker)

    with pytest.raises(RuntimeError, match="launch failed"):
        run_multi_gpu((scenario,), ("solver",), ("0", "1"), Path(benchmark_cli.__file__).resolve())
    assert process.terminated
    assert process.reaped


@pytest.mark.with_subprocess
def test_worker_cli_writes_result_file(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps({"scenarios": [asdict(_tiny_scenario())]}), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(benchmark_cli.__file__).resolve()),
            "--targets",
            "solver",
            "--worker-input",
            str(input_path),
            "--worker-output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    [measurement] = json.loads(output_path.read_text(encoding="utf-8"))
    assert measurement["status"] == "ok"


def test_gpu_selector_validation_rejects_unknown_device(monkeypatch):
    class QueryResult:
        stdout = "0, GPU-aaaa\n1, GPU-bbbb\n"

    monkeypatch.setattr(benchmark_multi_gpu.subprocess, "run", lambda *args, **kwargs: QueryResult())
    validate_gpu_selectors(("0", "GPU-bbbb"))
    with pytest.raises(ValueError, match="different physical devices"):
        validate_gpu_selectors(("0", "GPU-aaaa"))
    with pytest.raises(ValueError, match="unknown GPU selector"):
        validate_gpu_selectors(("9",))


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--convergence-threshold", "nan"], "convergence_threshold must be finite"),
        (["--final-loss-threshold", "inf"], "final_loss_threshold must be finite"),
        (["--memory-headroom-gib", "nan"], "memory-headroom-gib must be finite"),
    ],
)
def test_cli_rejects_non_finite_values_with_exit_code_two(argv, message, capsys):
    assert benchmark_cli.main(argv) == 2
    assert message in capsys.readouterr().err


def test_cli_writes_reports_and_returns_success(tmp_path):
    exit_code = benchmark_cli.main([
        "--targets",
        "solver",
        "--num-envs",
        "1",
        "--max-iters",
        "1",
        "--warmup",
        "0",
        "--repeat",
        "1",
        "--final-loss-threshold",
        "1e9",
        "--output-dir",
        str(tmp_path),
    ])
    assert exit_code == 0
    assert (tmp_path / "benchmark.json").is_file()
    assert (tmp_path / "benchmark.csv").is_file()


def test_cli_help_uses_a_child_process():
    result = subprocess.run(
        [sys.executable, str(Path(benchmark_cli.__file__).resolve()), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--capacity-search" in result.stdout
    assert "--worker-input" not in result.stdout
