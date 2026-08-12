# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import subprocess
import torch
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import isaaclab_arena.relations.benchmark.metadata as benchmark_metadata
import isaaclab_arena.relations.benchmark.reporting as benchmark_reporting
from isaaclab_arena.relations.benchmark import (
    BenchmarkScenario,
    SoftwareMetadata,
    build_distributed_run,
    build_run,
    collect_software_metadata,
    default_scenarios,
    env_count_sweep,
    format_scaling_summary,
    object_count_sweep,
    requested_scenario_ids,
    run_benchmarks,
    run_environment_benchmark,
    run_placer_benchmark,
    run_solver_benchmark,
    scenarios_for_modes,
    search_capacity,
    write_results_csv,
    write_results_json,
)
from isaaclab_arena.relations.benchmark.solver import _sample_child_origin, build_clutter_scene, build_solve_inputs
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relation_solver_state import RelationSolverState
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena_examples.relations import relation_solver_benchmark as benchmark_cli


class _InspectableRelationSolver(RelationSolver):
    def compute_total_loss(self, state: RelationSolverState) -> torch.Tensor:
        return self._compute_total_loss(state)


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
    def wait(self):
        return 0


def test_cli_does_not_write_reports_by_default():
    assert benchmark_cli._parse_args([]).output_dir is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_objects": 1}, "num_objects"),
        ({"num_envs": 0}, "num_envs"),
        ({"max_iters": 0}, "max_iters"),
        ({"convergence_threshold": -1.0}, "convergence_threshold"),
        ({"convergence_threshold": float("nan")}, "convergence_threshold"),
        ({"warmup_runs": -1}, "warmup_runs"),
        ({"timed_runs": 0}, "timed_runs"),
        ({"final_loss_threshold": float("inf")}, "final_loss_threshold"),
        ({"min_valid_layout_rate": 1.1}, "min_valid_layout_rate"),
        ({"min_valid_layout_rate": float("nan")}, "min_valid_layout_rate"),
    ],
)
def test_benchmark_scenario_rejects_invalid_values(overrides, message):
    kwargs = {"name": "test", "num_objects": 3, "num_envs": 1, **overrides}
    with pytest.raises(ValueError, match=message):
        BenchmarkScenario(**kwargs)


def test_build_clutter_scene_uses_placeable_assets():
    objects = build_clutter_scene(6)
    assert len(objects) == 6
    assert objects[0].name == "table"
    assert all(obj.get_bounding_box() is not None for obj in objects)


def test_build_clutter_scene_rejects_too_few_objects():
    with pytest.raises(AssertionError, match="at least anchor"):
        build_clutter_scene(1)


def test_mesh_clutter_scene_attaches_collision_meshes():
    objects = build_clutter_scene(3, collision_mode="mesh")
    assert all(obj.get_collision_mesh() is not None for obj in objects)


def test_build_solve_inputs_batch_shape():
    objects = build_clutter_scene(4)
    initial_positions, bboxes = build_solve_inputs(objects, num_envs=3, seed=7)
    assert len(initial_positions) == 3
    assert len(bboxes) == len(objects)


def test_build_solve_inputs_is_seeded():
    objects = build_clutter_scene(4)
    first, _ = build_solve_inputs(objects, num_envs=2, seed=7)
    second, _ = build_solve_inputs(objects, num_envs=2, seed=7)
    assert first == second


def test_sample_child_origin_centers_oversized_child():
    generator = torch.Generator().manual_seed(0)
    assert _sample_child_origin(0.0, 1.0, -2.0, 2.0, generator) == pytest.approx(0.5)


def test_default_scenarios_and_sweeps():
    assert [scenario.name for scenario in default_scenarios()] == ["small", "medium", "large"]
    assert {scenario.num_envs for scenario in object_count_sweep()} == {8}
    assert {scenario.num_objects for scenario in env_count_sweep()} == {6}


def test_scenarios_for_modes_builds_unique_ids():
    scenarios = scenarios_for_modes(
        (BenchmarkScenario(name="small", num_objects=3, num_envs=1),),
        ("bbox", "mesh"),
    )
    ids = requested_scenario_ids(scenarios, ("solver", "placer"))
    assert len(ids) == len(set(ids)) == 4


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


def test_object_suite_keeps_object_sweep_with_num_envs_override():
    args = benchmark_cli._parse_args(["--suite", "objects", "--num-envs", "4"])
    scenarios = benchmark_cli._base_scenarios(args)
    assert {scenario.num_objects for scenario in scenarios} == {3, 5, 6, 10}
    assert {scenario.num_envs for scenario in scenarios} == {4}


def test_solver_timing_uses_injected_clock():
    times = iter((1.0, 1.025))
    scenario = _tiny_scenario(name="timed")
    result = run_solver_benchmark(scenario, clock=lambda: next(times))
    assert result.status == "ok"
    assert result.solve_ms_samples == pytest.approx((25.0,))
    assert result.solve_ms == pytest.approx(25.0)
    assert result.solve_step_ms == pytest.approx(25.0)
    assert result.place_ms is None
    assert result.aabb_pair_count is not None
    assert result.mesh_pair_count == 0


def test_warmup_is_excluded_and_median_uses_all_timed_runs():
    times = iter((0.0, 10.0, 20.0, 20.003, 30.0, 30.001, 40.0, 40.002))
    scenario = _tiny_scenario(warmup_runs=1, timed_runs=3)
    result = run_solver_benchmark(scenario, clock=lambda: next(times))
    assert result.solve_ms_samples == pytest.approx((3.0, 1.0, 2.0))
    assert result.solve_ms == pytest.approx(2.0)


def test_zero_duration_keeps_measurement_without_infinite_throughput():
    result = run_solver_benchmark(_tiny_scenario(), clock=lambda: 1.0)
    assert result.status == "ok"
    assert result.solve_ms == 0.0
    assert result.throughput_envs_per_second is None


def test_solver_convergence_requires_every_environment_below_threshold():
    objects = build_clutter_scene(3)
    initial_positions, bboxes = build_solve_inputs(objects, num_envs=2, seed=0)
    for obj in objects[1:]:
        x, y, z = initial_positions[1][obj]
        initial_positions[1][obj] = (x + 100.0, y + 100.0, z)

    probe = _InspectableRelationSolver(RelationSolverParams(verbose=False))
    state = RelationSolverState(objects, initial_positions, env_bboxes=bboxes)
    probe.compute_total_loss(state)
    assert probe.last_loss_per_env is not None
    mean_loss = probe.last_loss_per_env.mean().item()
    max_loss = probe.last_loss_per_env.max().item()
    threshold = (mean_loss + max_loss) / 2

    solver = RelationSolver(RelationSolverParams(max_iters=2, convergence_threshold=threshold, verbose=False))
    solver.solve(objects, initial_positions, env_bboxes=bboxes)
    assert len(solver.last_loss_history) == 2


def test_last_loss_matches_returned_positions():
    objects = build_clutter_scene(3)
    initial_positions, bboxes = build_solve_inputs(objects, num_envs=2, seed=3)
    solver = _InspectableRelationSolver(RelationSolverParams(max_iters=1, convergence_threshold=0.0, verbose=False))
    final_positions = solver.solve(objects, initial_positions, env_bboxes=bboxes)
    assert solver.last_loss_per_env is not None
    reported_loss = solver.last_loss_per_env.clone()

    final_state = RelationSolverState(objects, final_positions, device=reported_loss.device, env_bboxes=bboxes)
    solver.compute_total_loss(final_state)
    assert solver.last_loss_per_env is not None
    torch.testing.assert_close(reported_loss, solver.last_loss_per_env)


def test_environment_failure_has_a_stable_result_id():
    result = run_environment_benchmark(_tiny_scenario())
    assert result.status == "failed"
    assert "missing-spec" in result.scenario_id
    assert result.error == "AssertionError: environment benchmarks require graph_spec_path"


def test_run_benchmarks_returns_separate_targets():
    scenario = _tiny_scenario(max_iters=2, min_valid_layout_rate=0.0)
    rows = run_benchmarks((scenario,))
    assert [row.target for row in rows] == ["solver", "placer"]
    assert rows[0].place_ms is None
    assert rows[1].solve_ms is None


def test_failed_solver_uses_none_for_unmeasured_fields():
    scenario = _tiny_scenario(name="failed", num_objects=10, final_loss_threshold=0.0)
    result = run_solver_benchmark(scenario)
    assert result.status == "failed"
    assert result.error is not None
    assert result.place_ms is None
    assert result.valid_layout_rate is None


def test_placer_uses_layout_validity_instead_of_loss_threshold():
    scenario = _tiny_scenario(num_objects=10, final_loss_threshold=0.0, min_valid_layout_rate=0.0)
    result = run_placer_benchmark(scenario)
    assert result.status == "ok"
    assert result.final_loss is not None and result.final_loss > 0.0


def test_build_run_records_missing_results_and_failed_workers():
    scenario = _tiny_scenario()
    result = run_solver_benchmark(scenario)
    incomplete = build_run((scenario,), ("solver", "placer"), [result])
    assert not incomplete.succeeded
    assert incomplete.missing_scenario_ids == (scenario.scenario_id("placer"),)

    failed_worker = build_run(
        (scenario,),
        ("solver",),
        [result],
        worker_exit_codes={"gpu-0": 137},
        worker_errors={"gpu-0": "worker exited with code 137"},
    )
    assert not failed_worker.succeeded
    assert failed_worker.worker_errors == {"gpu-0": "worker exited with code 137"}


def test_run_builders_collect_software_for_each_run(monkeypatch):
    software = SoftwareMetadata("abc123", True, "3.11", "2.8", "12.8")
    calls = []

    def collect():
        calls.append(None)
        return software

    monkeypatch.setattr(benchmark_reporting, "collect_software_metadata", collect)
    scenario = _tiny_scenario()
    result = run_solver_benchmark(scenario)
    local = build_run((scenario,), ("solver",), [result])
    distributed = build_distributed_run(
        [result],
        {"local": (result.scenario_id,)},
        {"local": 0},
    )

    assert local.software == distributed.software == software
    assert len(calls) == 2


def test_software_metadata_tolerates_unavailable_git(monkeypatch, tmp_path):
    calls = []

    def unavailable_git(*args, **kwargs):
        calls.append(kwargs)
        raise FileNotFoundError

    monkeypatch.setattr(benchmark_metadata.subprocess, "run", unavailable_git)
    metadata = collect_software_metadata(tmp_path)

    assert metadata.git_commit is None
    assert metadata.git_dirty is None
    assert calls[0]["cwd"] == tmp_path
    assert calls[0]["timeout"] == benchmark_metadata.GIT_TIMEOUT_SECONDS


def test_software_metadata_rejects_an_unrelated_git_root(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark_metadata, "_git_output", lambda *_args: str(tmp_path.parent))

    metadata = collect_software_metadata(tmp_path)

    assert metadata.git_commit is None
    assert metadata.git_dirty is None


def test_measurement_from_dict_reports_missing_fields():
    result = run_solver_benchmark(_tiny_scenario())
    payload = result.to_dict()
    del payload["num_spheres"]

    with pytest.raises(ValueError, match="missing required fields: num_spheres"):
        type(result).from_dict(payload)


def test_json_and_csv_reports_share_run_results(tmp_path):
    scenario = _tiny_scenario()
    result = run_solver_benchmark(scenario)
    run = build_run((scenario,), ("solver",), [result])
    json_path = tmp_path / "benchmark.json"
    csv_path = tmp_path / "benchmark.csv"
    write_results_json(json_path, run)
    write_results_csv(csv_path, run)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert payload["software"]["python_version"]
    assert "git_dirty" in payload["software"]
    assert payload["requested_scenario_ids"] == [result.scenario_id]
    assert rows[0]["scenario_id"] == result.scenario_id
    assert rows[0]["status"] == result.status


def test_scaling_summary_reports_successes_throughput_and_failures():
    base = run_solver_benchmark(_tiny_scenario())
    results = [
        replace(base, num_envs=1, throughput_envs_per_second=10.0),
        replace(base, num_envs=4, throughput_envs_per_second=25.0),
        replace(
            base,
            num_envs=8,
            status="failed",
            throughput_envs_per_second=None,
            error="out of memory",
        ),
    ]

    summary = format_scaling_summary(results)

    assert "highest successful=4" in summary
    assert "best throughput=4 (25.000 env/s)" in summary
    assert "failures=8: out of memory" in summary


def test_scaling_summary_does_not_merge_non_comparable_workloads():
    base = run_solver_benchmark(_tiny_scenario())
    object_results = [
        replace(base, num_envs=1, num_objects=3),
        replace(base, num_envs=8, num_objects=4),
    ]
    graph_results = [
        replace(
            base,
            target="environment",
            num_envs=1,
            graph_spec_path="one.yaml",
            include_robot=True,
        ),
        replace(
            base,
            target="environment",
            num_envs=8,
            graph_spec_path="two.yaml",
            include_robot=True,
        ),
    ]

    assert format_scaling_summary(object_results) == ""
    assert format_scaling_summary(graph_results) == ""


def test_capacity_search_uses_exponential_then_binary_probes():
    probes = []

    def probe(num_envs):
        probes.append(num_envs)
        return num_envs <= 10

    assert search_capacity(probe, max_num_envs=32) == 10
    assert probes[:5] == [1, 2, 4, 8, 16]
    assert probes[5:] == [12, 10, 11]


def test_multi_gpu_results_are_complete_and_aggregated(monkeypatch, tmp_path):
    scenario = _tiny_scenario()
    base_result = run_solver_benchmark(scenario)

    def launch_worker(scenarios, targets, physical_gpu, directory, worker_id):
        output_path = tmp_path / f"{worker_id}.json"
        result = base_result.to_dict()
        result["throughput_envs_per_second"] = 10.0 + int(physical_gpu)
        output_path.write_text(json.dumps([result]), encoding="utf-8")
        return _SuccessfulProcess(), output_path

    monkeypatch.setattr(benchmark_cli, "_launch_worker", launch_worker)
    rows, assignments, exit_codes, worker_errors = benchmark_cli._run_multi_gpu(
        (scenario,),
        ("solver",),
        ("0", "1"),
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

    def launch_worker(scenarios, targets, physical_gpu, directory, worker_id):
        return _SuccessfulProcess(), directory / "missing.json"

    monkeypatch.setattr(benchmark_cli, "_launch_worker", launch_worker)
    rows, assignments, exit_codes, worker_errors = benchmark_cli._run_multi_gpu(
        (scenario,),
        ("solver",),
        ("0",),
    )
    run = benchmark_cli.build_distributed_run(rows, assignments, exit_codes, worker_errors)

    assert rows == []
    assert not run.succeeded
    assert run.missing_scenario_ids == tuple(assignments["gpu-0"])
    assert "without writing" in worker_errors["gpu-0"]


def test_multi_gpu_omits_partial_aggregate_throughput(monkeypatch):
    scenario = _tiny_scenario()
    base_result = run_solver_benchmark(scenario)

    def launch_worker(scenarios, targets, physical_gpu, directory, worker_id):
        output_path = directory / f"{worker_id}.json"
        if physical_gpu == "0":
            output_path.write_text(json.dumps([base_result.to_dict()]), encoding="utf-8")
        return _SuccessfulProcess(), output_path

    monkeypatch.setattr(benchmark_cli, "_launch_worker", launch_worker)
    rows, _, _, worker_errors = benchmark_cli._run_multi_gpu(
        (scenario,),
        ("solver",),
        ("0", "1"),
    )
    assert len(rows) == 1
    assert rows[0].aggregate_throughput_envs_per_second is None
    assert "gpu-1" in worker_errors


@pytest.mark.with_subprocess
def test_worker_cli_writes_result_file(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps({"scenarios": [asdict(_tiny_scenario())]}), encoding="utf-8")
    result = subprocess.run(
        [
            TestConstants.python_path,
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

    monkeypatch.setattr(benchmark_cli.subprocess, "run", lambda *args, **kwargs: QueryResult())
    benchmark_cli._validate_gpu_selectors(("0", "GPU-bbbb"))
    with pytest.raises(ValueError, match="different physical devices"):
        benchmark_cli._validate_gpu_selectors(("0", "GPU-aaaa"))
    with pytest.raises(ValueError, match="unknown GPU selector"):
        benchmark_cli._validate_gpu_selectors(("9",))


@pytest.mark.parametrize(
    "argv",
    [
        ["--convergence-threshold", "nan"],
        ["--final-loss-threshold", "inf"],
        ["--memory-headroom-gib", "nan"],
    ],
)
def test_cli_rejects_non_finite_values(argv):
    args = benchmark_cli._parse_args(argv)
    with pytest.raises(ValueError, match="finite"):
        benchmark_cli._validate_args(args)


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
