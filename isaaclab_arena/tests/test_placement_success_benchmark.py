# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for placement-success results files, run comparisons and the benchmark script."""

import json
import sys

import pytest

from isaaclab_arena.offline_placement.placement_success import (
    EnvironmentRecord,
    EnvironmentStatus,
    PlacementBenchmarkResults,
    PlacementBenchmarkSettings,
    compare_results,
    format_comparison,
    format_results,
    summarize_suites,
)
from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


# Defined first so the SimulationApp starts before placement code in later tests imports pxr through Isaac Lab.
def _test_real_specs_are_scored_or_failed(simulation_app) -> bool:
    from isaaclab_arena.scripts.placement_success_benchmark import discover_specs, measure_environment
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
        assert (record.num_builds, record.num_envs) == (1, 2) and record.num_candidates > 0 and record.checks
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


def test_comparison_flags_only_changes_that_survive_holm():
    """Only changes that survive Holm's step-down are flagged, and unpaired environments and settings are listed."""
    before = [_scored("robolab", f"task_{index}", envs=(47, 50), candidates=(400, 500)) for index in range(38)]
    before.append(EnvironmentRecord("robolab", "new_task", EnvironmentStatus.FAILED, reason="missing asset"))
    before.append(EnvironmentRecord("robolab", "static_scene", EnvironmentStatus.SKIPPED))
    after = [_scored("robolab", f"task_{index}", envs=(47, 50), candidates=(400, 500)) for index in range(35)]
    after.append(EnvironmentRecord("robolab", "task_35", EnvironmentStatus.SKIPPED))
    after.append(_scored("robolab", "task_36", envs=(47, 50), candidates=(400, 500)))
    after.append(_scored("robolab", "task_37", envs=(12, 25), candidates=(100, 250)))
    after.append(EnvironmentRecord("robolab", "static_scene", EnvironmentStatus.SKIPPED))
    after_results = _results(after)
    after_results.settings = PlacementBenchmarkSettings(random_yaw_init=True, collision_mode=CollisionMode.MESH)

    comparison = compare_results(_results(before), after_results)

    [suite] = comparison.suites
    assert (suite.num_environments, suite.num_not_compared) == (37, 2)
    # Every environment weighs the same, whatever its sample size.
    assert suite.valid_envs.after == pytest.approx((36 * 0.94 + 0.48) / 37)
    changes = {environment.environment: environment for environment in comparison.environments}
    # Agresti-Caffo for 47/50 -> 12/25: adjusted rates 48/52 and 13/27.
    assert (changes["task_37"].valid_envs.low, changes["task_37"].valid_envs.high) == pytest.approx(
        (-0.64350, -0.23969), abs=1e-4
    )
    assert [name for name, environment in changes.items() if environment.significant] == ["task_37"]
    [new_task, task_35] = comparison.unpaired
    assert (new_task.environment, new_task.before.status, new_task.after) == ("new_task", "failed", None)
    assert (task_35.environment, task_35.before.status, task_35.after.status) == ("task_35", "scored", "skipped")
    assert comparison.settings_differences == {
        "random_yaw_init": (False, True),
        "collision_mode": (None, CollisionMode.MESH),
    }
    assert "- collision_mode: before None, after mesh" in format_comparison(comparison)

    # p = 0.035 and 0.047: both are below 5%, but the smaller misses Holm's first threshold of 2.5%, so neither is flagged.
    moderate = compare_results(
        _results([_scored("robolab", "task_0", envs=(47, 50), candidates=(400, 500))]),
        _results([_scored("robolab", "task_0", envs=(40, 50), candidates=(372, 500))]),
    )
    [environment] = moderate.environments
    assert max(environment.valid_envs.p_value, environment.valid_candidates.p_value) < 0.05
    assert not environment.significant

    # After two near-zero p-values, task_1's p = 0.014 is checked against 5% / 2 rather than Bonferroni's 5% / 4.
    step_down = compare_results(
        _results([_scored("robolab", f"task_{index}", envs=(47, 50), candidates=(400, 500)) for index in range(2)]),
        _results([
            _scored("robolab", "task_0", envs=(20, 50), candidates=(200, 500)),
            _scored("robolab", "task_1", envs=(38, 50), candidates=(400, 500)),
        ]),
    )
    assert [environment.significant for environment in step_down.environments] == [True, True]


def test_suite_interval_is_centered_on_the_observed_change():
    """Identical rates show no change even when the runs differ in size, and the p-value agrees with the interval.

    A difference of two rates cannot pass 100 points, so the printed interval stops there, while the stored interval
    keeps its full width for suite averages.
    """
    before = [_scored("robolab", f"task_{index}", envs=(150, 150), candidates=(1500, 1500)) for index in range(36)]
    after = [_scored("robolab", f"task_{index}", envs=(50, 50), candidates=(500, 500)) for index in range(36)]

    [suite] = compare_results(_results(before), _results(after)).suites
    [single] = compare_results(_results(before[:1]), _results(after[:1])).suites

    assert suite.valid_envs.low == pytest.approx(-suite.valid_envs.high)
    assert suite.valid_envs.p_value == pytest.approx(1.0)
    # Independent environment errors add in quadrature, so 36 equal environments narrow the interval six times.
    single_width = single.valid_envs.high - single.valid_envs.low
    assert suite.valid_envs.high - suite.valid_envs.low == pytest.approx(single_width / 6)

    # Small drops in every environment: the p-value is below 5% exactly when the interval excludes zero.
    before = [_scored("robolab", f"task_{index}", envs=(47, 50), candidates=(400, 500)) for index in range(36)]
    for valid_envs in range(44, 48):
        after = [_scored("robolab", f"task_{index}", (valid_envs, 50), (400, 500)) for index in range(36)]
        [suite] = compare_results(_results(before), _results(after)).suites
        change = suite.valid_envs
        assert (change.high < 0 or change.low > 0) == (change.p_value < 0.05)

    collapse = compare_results(
        _results([_scored("robolab", "task_0", envs=(50, 50), candidates=(500, 500))]),
        _results([_scored("robolab", "task_0", envs=(0, 50), candidates=(0, 500))]),
    )
    assert collapse.environments[0].valid_envs.low < -1
    assert "| robolab | 1 | 0 | 100.0% → 0.0% | [-100.0, " in format_comparison(collapse)


def test_comparison_keeps_environments_whose_required_checks_changed_and_warns():
    """Environments whose required checks differ stay in the comparison and are named; optional checks don't count."""
    before = [_scored("robolab", f"task_{index}", envs=(47, 50), candidates=(400, 500)) for index in range(4)]
    after = [_scored("robolab", f"task_{index}", envs=(25, 50), candidates=(200, 500)) for index in range(4)]
    for record in before + after:
        record.checks = {"no_overlap": True, "ik_reachable": True}
    # task_0 loses ik_reachable, task_1 makes it optional and task_2 gains a required check.
    after[0].checks = {"no_overlap": True}
    after[1].checks = {"no_overlap": True, "ik_reachable": False}
    after[2].checks = {"no_overlap": True, "ik_reachable": True, "on_relation": True}
    # task_3 gains an optional check, which does not change success.
    after[3].checks = {"no_overlap": True, "ik_reachable": True, "face_to": False}

    comparison = compare_results(_results(before), _results(after))

    # A change to the required checks is often the change being measured, so its effect stays in the suite rates.
    assert len(comparison.environments) == 4
    assert [pair.environment for pair in comparison.check_differences] == ["task_0", "task_1", "task_2"]
    assert (
        "- robolab/task_1: before ik_reachable, no_overlap; after ik_reachable (optional), no_overlap"
        in format_comparison(comparison)
    )


def test_comparison_headline_counts_environments_that_failed_after_the_change():
    """Failures after a change open the comparison, and every suite row counts what it could not compare."""
    before = [_scored("robolab", f"task_{index}", envs=(47, 50), candidates=(400, 500)) for index in range(3)]
    before.append(_scored("kitchen_bench", "open_fridge", envs=(50, 50), candidates=(500, 500)))
    after = [_scored("robolab", f"task_{index}", envs=(47, 50), candidates=(400, 500)) for index in range(2)]
    after.append(EnvironmentRecord("robolab", "task_2", EnvironmentStatus.FAILED, reason="KeyError: 'lime'"))
    after.append(EnvironmentRecord("kitchen_bench", "open_fridge", EnvironmentStatus.FAILED, reason="ValueError"))

    text = format_comparison(compare_results(_results(before), _results(after)))

    assert text.startswith("Warning: these environments failed after the change:\n- kitchen_bench/open_fridge")
    assert "| robolab | 2 | 1 | 80.0% → 80.0% |" in text
    # A suite whose every environment failed still has a row.
    assert "| kitchen_bench | 0 | 1 | n/a | n/a | n/a | n/a |" in text


def test_build_results_count_the_first_pool_batch_and_distinct_fallback_environments():
    """Candidates come from each build's first pool batch; refill batches and repeated fallbacks are not recounted."""
    from isaaclab_arena.relations.placement_diagnostics import CheckCounts, PlacementDiagnostics, ValidatedBatch
    from isaaclab_arena.scripts.placement_success_benchmark import add_build_results

    # IK runs only on the candidates that pass no_overlap, so it rejects 10 of 70 rather than 40 of 100.
    first_batch = ValidatedBatch(
        100, 60, {"no_overlap": CheckCounts(100, 70, True), "ik_reachable": CheckCounts(70, 60, True)}
    )
    refill_batch = ValidatedBatch(
        20, 15, {"no_overlap": CheckCounts(20, 16, True), "ik_reachable": CheckCounts(16, 15, True)}
    )
    record = EnvironmentRecord("kitchen_bench", "open_fridge", EnvironmentStatus.SCORED)

    add_build_results(record, PlacementDiagnostics([first_batch, refill_batch], fallback_envs=[3, 3]), num_envs=10)
    add_build_results(record, PlacementDiagnostics([first_batch]), num_envs=10)

    assert (record.num_valid_candidates, record.num_candidates) == (120, 200)
    assert (record.num_fallback_envs, record.num_envs, record.num_batches, record.num_builds) == (1, 20, 3, 2)
    assert record.rejections_per_check == {"no_overlap": 60, "ik_reachable": 20}
    assert record.checks == {"no_overlap": True, "ik_reachable": True}


def test_placement_seeds_give_every_build_its_own_candidates():
    """Builds of different seeds and different specs never share candidates, so their samples are independent."""
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.scripts.placement_success_benchmark import _SEEDS_PER_ENVIRONMENT, get_placement_seed
    from isaaclab_arena.tests.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    desk = DummyObject(name="desk", bounding_box=AxisAlignedBoundingBox((0.0, 0.0, 0.0), (1.0, 1.0, 0.1)))
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())
    box = DummyObject(name="box", bounding_box=AxisAlignedBoundingBox((0.0, 0.0, 0.0), (0.2, 0.2, 0.2)))
    box.add_relation(On(desk))
    params = ObjectPlacerParams(
        solver_params=RelationSolverParams(max_iters=10, verbose=False),
        max_placement_attempts=2,
        min_unique_layouts_per_env=2,
    )
    num_envs = 2

    box_positions_per_build = []
    for seed_block_index, seed in ((0, 0), (0, 1), (1, 0)):
        params.placement_seed = get_placement_seed(params, num_envs, seed_block_index, seed)
        pool = PooledObjectPlacer([desk, box], params, pool_size=num_envs * 2, num_envs=num_envs)
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
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.scripts.placement_success_benchmark import get_placement_seed
    from isaaclab_arena.tests.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    desk = DummyObject(name="desk", bounding_box=AxisAlignedBoundingBox((0.0, 0.0, 0.0), (1.0, 1.0, 0.1)))
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())
    # Wider than the desk, so no candidate is ever valid.
    box = DummyObject(name="box", bounding_box=AxisAlignedBoundingBox((0.0, 0.0, 0.0), (1.5, 1.5, 0.2)))
    box.add_relation(On(desk))
    params = ObjectPlacerParams(
        solver_params=RelationSolverParams(max_iters=10, verbose=False),
        max_placement_attempts=2,
        min_unique_layouts_per_env=2,
        placement_seed=0,
    )
    num_envs = 2

    pool = PooledObjectPlacer([desk, box], params, pool_size=num_envs * 2, num_envs=num_envs)

    solved = sum(batch.num_candidates for batch in pool.diagnostics.batches)
    reserved = get_placement_seed(params, num_envs, 0, 1) - get_placement_seed(params, num_envs, 0, 0)
    assert len(pool.diagnostics.batches) == params.max_placement_attempts
    assert solved == reserved


def test_overridden_placer_params_change_a_copy():
    """Yaw and collision-mode overrides change a copy of the environment's placer parameters."""
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.scripts.placement_success_benchmark import get_overridden_placer_params

    settings = PlacementBenchmarkSettings(random_yaw_init=True, collision_mode=CollisionMode.MESH)
    original = ObjectPlacerParams()

    overridden = get_overridden_placer_params(original, settings)

    assert overridden.random_yaw_init and overridden.solver_params.collision_mode == CollisionMode.MESH
    assert not original.random_yaw_init and original.solver_params.collision_mode != CollisionMode.MESH


def test_spec_discovery_filters_by_stem_and_gives_each_spec_its_own_seed_block():
    """Every suite contributes specs, a stem glob keeps only matching specs, and every spec has its own seed block."""
    from isaaclab_arena.scripts.placement_success_benchmark import discover_specs, get_seed_block_index

    specs = discover_specs("all", None)
    assert {suite for suite, _ in specs} == {"robolab", "kitchen_bench"}
    stem = specs[0][1].stem
    assert [path.stem for _, path in discover_specs("all", [stem])] == [stem]
    seed_block_indices = [get_seed_block_index(suite, path) for suite, path in specs]
    assert sorted(seed_block_indices) == list(range(len(specs)))


def test_compare_cli_prints_suite_table_without_simulator(tmp_path, monkeypatch, capsys):
    """--compare reads two results files and prints the suite comparison."""
    from isaaclab_arena.scripts import placement_success_benchmark

    before_path, after_path = tmp_path / "before.json", tmp_path / "after.json"
    _results([_scored("robolab", "pick_apple", envs=(45, 50), candidates=(300, 500))]).save(before_path)
    _results([_scored("robolab", "pick_apple", envs=(50, 50), candidates=(450, 500))]).save(after_path)
    argv = ["placement_success_benchmark.py", "--compare", str(before_path), str(after_path)]
    monkeypatch.setattr(sys, "argv", argv)

    placement_success_benchmark.main()

    assert "| robolab | 1 | 0 | 60.0% → 90.0% |" in capsys.readouterr().out


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
