# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import torch
from dataclasses import asdict, replace

import pytest

import isaaclab_arena.relations.benchmark.environment_benchmark as benchmark_environment
import isaaclab_arena.relations.benchmark.multi_gpu as benchmark_multi_gpu
import isaaclab_arena.relations.benchmark.provenance as benchmark_provenance
import isaaclab_arena.relations.benchmark.reporting as benchmark_reporting
from isaaclab_arena.relations.benchmark.environment_benchmark import run_environment_benchmark
from isaaclab_arena.relations.benchmark.models import BenchmarkScenario, DeviceMetadata
from isaaclab_arena.relations.benchmark.multi_gpu import search_capacity
from isaaclab_arena.relations.benchmark.provenance import SoftwareMetadata, collect_software_metadata
from isaaclab_arena.relations.benchmark.reporting import (
    build_distributed_run,
    build_run,
    format_results_table,
    format_scaling_summary,
    requested_scenario_ids,
    write_results_csv,
    write_results_json,
)
from isaaclab_arena.relations.benchmark.synthetic_benchmark import (
    _sample_child_origin,
    build_clutter_scene,
    build_solve_inputs,
    default_scenarios,
    env_count_sweep,
    object_count_sweep,
    run_benchmarks,
    run_placer_benchmark,
    run_solver_benchmark,
    scenarios_for_modes,
)
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relation_solver_state import RelationSolverState


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
    assert {scenario.num_envs for scenario in object_count_sweep()} == {1}
    assert {scenario.num_objects for scenario in env_count_sweep()} == {3}
    assert {scenario.num_envs for scenario in env_count_sweep()} == {1, 8, 32, 128}


def test_scenarios_for_modes_builds_unique_ids():
    scenarios = scenarios_for_modes(
        (BenchmarkScenario(name="small", num_objects=3, num_envs=1),),
        ("bbox", "mesh"),
    )
    ids = requested_scenario_ids(scenarios, ("solver", "placer"))
    assert len(ids) == len(set(ids)) == 4


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


def test_environment_bring_up_uses_paired_samples_and_table_median(monkeypatch):
    samples = iter([
        benchmark_environment._EnvironmentSample(1.0, 100.0, 3, True, None),
        benchmark_environment._EnvironmentSample(2.0, 2.0, 3, True, None),
        benchmark_environment._EnvironmentSample(100.0, 1.0, 3, True, None),
    ])
    device = DeviceMetadata(None, None, None, None, None, None, None)
    monkeypatch.setattr(benchmark_environment, "_build_and_reset_environment", lambda *_args: next(samples))
    monkeypatch.setattr(benchmark_environment, "get_device_metadata", lambda: device)
    monkeypatch.setattr(benchmark_environment, "record_free_memory_after", lambda value: value)
    monkeypatch.setattr(benchmark_environment, "get_peak_memory", lambda: (None, None))
    monkeypatch.setattr(benchmark_environment, "reset_peak_memory", lambda: None)

    result = run_environment_benchmark(
        _tiny_scenario(graph_spec_path="existing.yaml", timed_runs=3),
    )

    assert result.status == "ok"
    assert result.build_ms == pytest.approx(2.0)
    assert result.reset_ms == pytest.approx(2.0)
    assert result.bring_up_ms_samples == pytest.approx((101.0, 4.0, 101.0))
    assert result.bring_up_ms == pytest.approx(101.0)
    assert result.throughput_envs_per_second == pytest.approx(1000.0 / 101.0)
    table = format_results_table([result]).splitlines()
    assert "objects" in table[0]
    assert "batch_ms" in table[0]
    assert "layouts/s" in table[0]
    row = table[2]
    assert "101.000" in row


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

    monkeypatch.setattr(benchmark_provenance.subprocess, "run", unavailable_git)
    metadata = collect_software_metadata(tmp_path)

    assert metadata.git_commit is None
    assert metadata.git_dirty is None
    assert calls[0]["cwd"] == tmp_path
    assert calls[0]["timeout"] == benchmark_provenance.GIT_TIMEOUT_SECONDS


def test_software_metadata_rejects_an_unrelated_git_root(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark_provenance, "_git_output", lambda *_args: str(tmp_path.parent))

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
    assert "bring_up_ms_samples" in payload["results"][0]
    assert "bring_up_ms" in rows[0]
    restored = type(result).from_dict({
        **result.to_dict(),
        "bring_up_ms_samples": [3.0, 5.0],
        "bring_up_ms": 4.0,
    })
    assert restored.bring_up_ms_samples == (3.0, 5.0)
    assert restored.bring_up_ms == 4.0


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

    assert "Batch-size scaling (fixed object count)" in summary
    assert "throughput_vs_batch_1" in summary
    assert "4=25.000 layouts/s, 2.50x, ok" in summary
    assert "highest successful batch=4" in summary
    assert "failures=8: out of memory" in summary


def test_scaling_summary_reports_object_count_slowdown():
    base = run_solver_benchmark(_tiny_scenario())
    results = [
        replace(base, num_objects=3, solve_ms=10.0),
        replace(base, num_objects=5, solve_ms=15.0),
        replace(base, num_objects=10, solve_ms=30.0, status="failed"),
    ]

    summary = format_scaling_summary(results)

    assert "Object-count scaling (fixed batch size)" in summary
    assert "[bbox, batch=1]" in summary
    assert "latency_vs_objects_3" in summary
    assert "3=10.000 ms, 1.00x, ok" in summary
    assert "10=30.000 ms, 3.00x, failed" in summary


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


def test_environment_worker_writes_before_simulation_teardown(monkeypatch, tmp_path):
    import isaaclab_arena.utils.isaaclab_utils.simulation_app as simulation_app

    scenario = _tiny_scenario(num_objects=0, graph_spec_path="unused.yaml")
    result = run_solver_benchmark(_tiny_scenario())
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps({"scenarios": [asdict(scenario)]}), encoding="utf-8")
    monkeypatch.setattr(benchmark_multi_gpu, "run_benchmarks", lambda *_args, **_kwargs: [result])

    class _SimulationContext:
        def __init__(self, args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            assert output_path.is_file()

    monkeypatch.setattr(simulation_app, "SimulationAppContext", _SimulationContext)

    assert benchmark_multi_gpu.run_worker(input_path, output_path, None, ("environment",)) == 0
