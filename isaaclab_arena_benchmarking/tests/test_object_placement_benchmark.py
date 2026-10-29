# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the placement success benchmark: measuring specs, seeding builds, provenance and results files."""

import json

import pytest

from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.relations.placement_diagnostics import ValidationCheckDiagnostics
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena_benchmarking.object_placement.results import (
    EnvironmentRecord,
    EnvironmentStatus,
    PlacementBenchmarkResults,
    PlacementBenchmarkSettings,
    format_results,
    summarize_suites,
)


# Defined first so the SimulationApp starts before placement code in later tests imports pxr through Isaac Lab.
def _test_real_specs_are_scored_or_failed(simulation_app) -> bool:
    from isaaclab_arena_benchmarking.object_placement.measurement import discover_specs, measure_environment
    from isaaclab_arena_environments.cli import ensure_environments_registered

    ensure_environments_registered()
    settings = PlacementBenchmarkSettings(num_envs=2, placement_seeds=[0])
    [(_, scored_spec)] = discover_specs("robolab", ["banana_on_plate"])
    [(_, robot_only_spec)] = discover_specs("kitchen_bench", ["kitchen_bench_lightwheel_open_fridge"])

    scored = measure_environment("robolab", scored_spec, settings, seed_block_index=0)
    robot_only = measure_environment("kitchen_bench", robot_only_spec, settings, seed_block_index=1)
    failed = measure_environment("robolab", scored_spec.with_name("missing_spec.yaml"), settings, seed_block_index=2)

    for record in (scored, robot_only):
        assert record.status == EnvironmentStatus.SCORED, record.reason
        # One normal build of two environments, read from its placement diagnostics.
        assert (record.num_builds, record.num_envs) == (1, 2) and record.num_candidates > 0
        assert record.checks["on_relation"].num_evaluated == record.num_candidates
    assert failed.status == EnvironmentStatus.FAILED and failed.reason
    return True


def test_real_specs_are_scored_or_failed():
    """An object spec and a spec that places only the robot are scored, and a spec that cannot be built fails."""
    assert run_function_with_persistent_simulation_app(_test_real_specs_are_scored_or_failed)


def test_results_file_round_trips_and_reports_suite_rates(tmp_path):
    """Suites report counts for every status, and equal-weight mean rates only when no environment failed."""
    results = _results([
        _scored("robolab", "pick_apple", envs=(45, 50), candidates=(300, 500)),
        _scored("robolab", "pick_banana", envs=(25, 25), candidates=(250, 250)),
        _scored("kitchen_bench", "pick_cup", envs=(50, 50), candidates=(500, 500)),
        EnvironmentRecord("kitchen_bench", "static_scene", EnvironmentStatus.SKIPPED, reason="solves no relations"),
        EnvironmentRecord("kitchen_bench", "place_lime", EnvironmentStatus.FAILED, reason="KeyError: 'lime'"),
    ])
    results.environments[0].checks = {
        "no_overlap": ValidationCheckDiagnostics(500, 350, required=True),
        "face_to": ValidationCheckDiagnostics(350, 300, required=False),
    }
    results.settings = PlacementBenchmarkSettings(
        random_yaw_init=True, collision_mode=CollisionMode.MESH, solver_device="cpu"
    )
    path = tmp_path / "run.json"
    results.save(path)

    assert PlacementBenchmarkResults.load(path) == results
    # A file written by another version names its unknown fields instead of failing deep inside the loader.
    old_version = json.loads(path.read_text())
    old_version["environments"][0]["num_layouts"] = 50
    path.write_text(json.dumps(old_version))
    with pytest.raises(AssertionError, match=r"EnvironmentRecord fields .*\['num_layouts'\]"):
        PlacementBenchmarkResults.load(path)
    robolab, kitchen_bench = summarize_suites(results)
    # Every environment weighs the same, whatever its sample size.
    assert robolab.valid_candidate_rate == pytest.approx(0.8)
    assert robolab.valid_env_rate == pytest.approx(0.95)
    assert (kitchen_bench.num_scored, kitchen_bench.num_skipped, kitchen_bench.num_failed) == (1, 1, 1)
    # A failed environment could have lowered the suite's rates, so they are left out.
    assert kitchen_bench.valid_candidate_rate is None
    assert "| kitchen_bench | 1 | 1 | 1 | n/a | n/a |" in format_results(results)


def test_build_results_count_the_first_pool_batch_and_distinct_fallback_environments():
    """Candidates and checks come from each build's first pool batch; refills and repeated fallbacks are not counted."""
    from isaaclab_arena.relations.placement_diagnostics import BatchValidationDiagnostics, PlacementDiagnostics
    from isaaclab_arena_benchmarking.object_placement.measurement import add_build_results

    # IK runs only on the candidates that pass no_overlap, so it rejects 10 of 70 rather than 40 of 100.
    first_batch = BatchValidationDiagnostics(
        100,
        60,
        {
            "no_overlap": ValidationCheckDiagnostics(100, 70, True),
            "ik_reachable": ValidationCheckDiagnostics(70, 60, True),
        },
    )
    refill_batch = BatchValidationDiagnostics(
        20,
        15,
        {
            "no_overlap": ValidationCheckDiagnostics(20, 16, True),
            "ik_reachable": ValidationCheckDiagnostics(16, 15, True),
        },
    )
    record = EnvironmentRecord("kitchen_bench", "open_fridge", EnvironmentStatus.SCORED)

    add_build_results(record, PlacementDiagnostics([first_batch, refill_batch], fallback_envs=[3, 3]), num_envs=10)
    add_build_results(record, PlacementDiagnostics([first_batch]), num_envs=10)

    assert (record.num_valid_candidates, record.num_candidates) == (120, 200)
    assert (record.num_fallback_envs, record.num_envs, record.num_batches, record.num_builds) == (1, 20, 3, 2)
    assert record.checks == {
        "no_overlap": ValidationCheckDiagnostics(200, 140, True),
        "ik_reachable": ValidationCheckDiagnostics(140, 120, True),
    }
    # Adding the same first batch twice must not change it.
    assert first_batch.checks["no_overlap"] == ValidationCheckDiagnostics(100, 70, True)


def test_build_results_mark_checks_outside_required_checks_as_optional():
    """A check that required_checks leaves out is recorded as optional, so it does not decide valid candidates."""
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena_benchmarking.object_placement.measurement import add_build_results

    params = _placer_params(placement_seed=0, required_checks={"on_relation"})
    pool = PooledObjectPlacer(_desk_and_box(box_width_m=0.2), params, pool_size=2 * 2, num_envs=2)
    record = EnvironmentRecord("robolab", "box_on_desk", EnvironmentStatus.SCORED)

    add_build_results(record, pool.last_fill_diagnostics, num_envs=2)

    assert record.checks["on_relation"].required and not record.checks["no_overlap"].required


def test_placement_seeds_give_every_build_its_own_candidates():
    """Builds of different seeds and different specs never share candidates, so their samples are independent."""
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena_benchmarking.object_placement.measurement import _SEEDS_PER_ENVIRONMENT, get_placement_seed

    desk_and_box = _desk_and_box(box_width_m=0.2)
    box = desk_and_box[1]
    params = _placer_params()
    num_envs = 2

    box_positions_per_build = []
    for seed_block_index, seed in ((0, 0), (0, 1), (1, 0)):
        params.placement_seed = get_placement_seed(params, num_envs, seed_block_index, seed)
        pool = PooledObjectPlacer(desk_and_box, params, pool_size=num_envs * 2, num_envs=num_envs)
        box_positions = set()
        for env_layouts in pool.layouts_per_env():
            for layout in env_layouts:
                box_positions.add(tuple(layout.positions[box]))
        box_positions_per_build.append(box_positions)

    first_seed, second_seed, other_spec = box_positions_per_build
    assert not first_seed & second_seed and not first_seed & other_spec
    # A seed whose builds would run into the next spec's block is refused.
    with pytest.raises(AssertionError):
        get_placement_seed(params, num_envs, 0, _SEEDS_PER_ENVIRONMENT)


def test_a_build_that_solves_every_pool_batch_fills_exactly_its_reserved_seeds():
    """A pool that never finds a valid layout solves every batch, and those candidates exactly fill the build's seeds."""
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena_benchmarking.object_placement.measurement import get_placement_seed

    params = _placer_params(placement_seed=0)
    num_envs = 2

    # A box wider than the desk is never valid.
    pool = PooledObjectPlacer(_desk_and_box(box_width_m=1.5), params, pool_size=num_envs * 2, num_envs=num_envs)

    batches = pool.last_fill_diagnostics.batches
    reserved = get_placement_seed(params, num_envs, 0, 1) - get_placement_seed(params, num_envs, 0, 0)
    assert len(batches) == params.max_placement_attempts
    assert sum(batch.num_candidates for batch in batches) == reserved


def test_git_provenance_falls_back_to_the_stamped_commit(tmp_path, monkeypatch):
    """Without a readable checkout, the commit and branch come from the variables an Arena image can set."""
    from isaaclab_arena_benchmarking.object_placement import measurement

    monkeypatch.setattr(measurement, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("ARENA_GIT_SHA", raising=False)
    assert measurement.get_git_provenance() == (None, None, None)
    # An image built without its commit stamps "unknown", which is no commit and says nothing about changes.
    monkeypatch.setenv("ARENA_GIT_SHA", "unknown")
    monkeypatch.setenv("ARENA_GIT_BRANCH", "unknown")
    assert measurement.get_git_provenance() == (None, None, None)

    monkeypatch.setenv("ARENA_GIT_SHA", "1a2b3c4-dirty")
    monkeypatch.setenv("ARENA_GIT_BRANCH", "main")
    assert measurement.get_git_provenance() == ("1a2b3c4", "main", True)


def test_overridden_placer_params_change_a_copy():
    """Yaw and collision-mode overrides change a copy of the environment's placer parameters."""
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena_benchmarking.object_placement.measurement import get_overridden_placer_params

    settings = PlacementBenchmarkSettings(random_yaw_init=True, collision_mode=CollisionMode.MESH)
    original = ObjectPlacerParams()

    overridden = get_overridden_placer_params(original, settings)

    assert overridden.random_yaw_init and overridden.solver_params.collision_mode == CollisionMode.MESH
    assert not original.random_yaw_init and original.solver_params.collision_mode != CollisionMode.MESH


def test_spec_discovery_filters_by_stem_and_gives_each_spec_its_own_seed_block():
    """Every suite contributes specs, a stem glob keeps only matching specs, and every spec has its own seed block."""
    from isaaclab_arena_benchmarking.object_placement.measurement import discover_specs, get_seed_block_index

    specs = discover_specs("all", None)
    assert {suite for suite, _ in specs} == {"robolab", "kitchen_bench"}
    stem = specs[0][1].stem
    assert [path.stem for _, path in discover_specs("all", [stem])] == [stem]
    seed_block_indices = [get_seed_block_index(suite, path) for suite, path in specs]
    assert sorted(seed_block_indices) == list(range(len(specs)))


def _desk_and_box(box_width_m: float) -> list:
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.tests.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    desk = DummyObject(name="desk", bounding_box=AxisAlignedBoundingBox((0.0, 0.0, 0.0), (1.0, 1.0, 0.1)))
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())
    box = DummyObject(name="box", bounding_box=AxisAlignedBoundingBox((0.0, 0.0, 0.0), (box_width_m, box_width_m, 0.2)))
    box.add_relation(On(desk))
    return [desk, box]


def _placer_params(**overrides):
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

    return ObjectPlacerParams(
        solver_params=RelationSolverParams(max_iters=10, verbose=False),
        max_placement_attempts=2,
        min_unique_layouts_per_env=2,
        **overrides,
    )


def _scored(suite: str, environment: str, envs: tuple[int, int], candidates: tuple[int, int]) -> EnvironmentRecord:
    """Return a scored record from (valid, total) environment and candidate counts, over three single-batch builds."""
    return EnvironmentRecord(
        suite,
        environment,
        EnvironmentStatus.SCORED,
        num_builds=3,
        num_candidates=candidates[1],
        num_valid_candidates=candidates[0],
        num_envs=envs[1],
        num_fallback_envs=envs[1] - envs[0],
        num_batches=3,
    )


def _results(records: list[EnvironmentRecord]) -> PlacementBenchmarkResults:
    return PlacementBenchmarkResults(
        "2026-10-21T00:00:00+00:00", "abc123", "main", False, PlacementBenchmarkSettings(), records
    )
