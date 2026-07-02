# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import torch

import pytest

from isaaclab_arena.relations.relation_solver_benchmark import (
    BenchmarkScenario,
    _sample_child_origin,
    build_clutter_scene,
    build_solve_inputs,
    default_scenarios,
    env_count_sweep,
    format_results_table,
    make_solver_params,
    mesh_collision_available,
    object_count_sweep,
    run_benchmarks,
    run_solver_benchmark,
    scenarios_for_modes,
    write_results_json,
)


def test_build_clutter_scene_counts():
    objects = build_clutter_scene(6)
    assert len(objects) == 6
    assert objects[0].name == "table"


def test_build_clutter_scene_rejects_too_few_objects():
    with pytest.raises(AssertionError, match="need at least anchor"):
        build_clutter_scene(1)


def test_build_mesh_clutter_scene_attaches_meshes():
    pytest.importorskip("trimesh")
    objects = build_clutter_scene(3, "mesh")
    assert all(obj.get_collision_mesh() is not None for obj in objects)


def test_sample_child_origin_uses_parent_center_when_child_wider():
    generator = torch.Generator().manual_seed(0)
    origin = _sample_child_origin(0.0, 1.0, -2.0, 2.0, generator)
    assert origin == pytest.approx(0.5)


def test_build_solve_inputs_batch_shape():
    objects = build_clutter_scene(4)
    initial_positions, bboxes = build_solve_inputs(objects, num_envs=3, seed=7)
    assert len(initial_positions) == 3
    assert len(bboxes) == len(objects)


def test_default_scenarios_preset_names():
    scenarios = default_scenarios()
    assert [s.name for s in scenarios] == ["small", "medium", "large"]
    assert scenarios[0].num_objects == 3 and scenarios[0].num_envs == 1
    assert scenarios[-1].num_objects == 10 and scenarios[-1].num_envs == 32


def test_run_solver_benchmark_positive_timing():
    scenario = BenchmarkScenario(name="tiny", num_objects=3, num_envs=1, max_iters=5, warmup_runs=0, timed_runs=1)
    row = run_solver_benchmark(scenario)
    assert row.solve_ms > 0.0
    assert 0 < row.iters <= scenario.max_iters
    assert row.ms_per_iter == pytest.approx(row.solve_ms / row.iters)
    assert row.overlap_pairs > 0


def test_run_benchmarks_merges_placer_timing():
    scenario = BenchmarkScenario(name="tiny", num_objects=3, num_envs=1, max_iters=3, warmup_runs=0, timed_runs=1)
    solver_row = run_solver_benchmark(scenario)
    (merged,) = run_benchmarks((scenario,), include_placer=True)

    assert merged.scenario_name == scenario.name
    assert merged.collision_mode == scenario.collision_mode
    assert merged.num_objects == scenario.num_objects
    assert merged.num_envs == scenario.num_envs
    assert merged.num_optimizable == scenario.num_objects - 1
    assert merged.device == solver_row.device
    assert merged.iters == solver_row.iters
    assert merged.overlap_pairs == solver_row.overlap_pairs
    assert merged.solve_ms > 0.0
    assert merged.place_ms > 0.0
    assert merged.ms_per_iter == pytest.approx(merged.solve_ms / merged.iters)


def test_object_sweep_holds_envs_fixed():
    scenarios = object_count_sweep()
    assert {s.num_envs for s in scenarios} == {8}
    assert {s.num_objects for s in scenarios} == {3, 5, 6, 10}


def test_env_sweep_holds_objects_fixed():
    scenarios = env_count_sweep()
    assert {s.num_objects for s in scenarios} == {6}
    assert {s.num_envs for s in scenarios} == {1, 8, 32}


def test_scenarios_for_modes_expands():
    (bbox_row, mesh_row) = scenarios_for_modes(
        (BenchmarkScenario(name="small", num_objects=3, num_envs=1),),
        ("bbox", "mesh"),
    )
    assert bbox_row.name == "small_bbox"
    assert mesh_row.collision_mode == "mesh"


def test_format_results_table_includes_header():
    scenario = BenchmarkScenario(name="tiny", num_objects=3, num_envs=1, max_iters=2, warmup_runs=0, timed_runs=1)
    table = format_results_table([run_solver_benchmark(scenario)])
    assert "solve_ms" in table
    assert "tiny" in table


def test_write_results_json_round_trip(tmp_path):
    scenario = BenchmarkScenario(name="tiny", num_objects=3, num_envs=1, max_iters=2, warmup_runs=0, timed_runs=1)
    row = run_solver_benchmark(scenario)
    out_path = tmp_path / "bench.json"
    write_results_json(out_path, [row])
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded == [row.to_dict()]


def test_mesh_mode_solver_params():
    scenario = BenchmarkScenario(name="mesh", num_objects=3, num_envs=1, collision_mode="mesh", max_iters=1)
    if not mesh_collision_available():
        with pytest.raises(AssertionError, match="collision_mode module and Warp"):
            make_solver_params(scenario)
        return

    from isaaclab_arena.relations.collision_mode import CollisionMode

    params = make_solver_params(scenario)
    assert params.collision_mode == CollisionMode.MESH
