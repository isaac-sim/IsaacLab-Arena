# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for placement-success results files, run comparisons and the benchmark script."""

import copy
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
    settings = PlacementBenchmarkSettings(num_layouts=2, placement_seeds=[0])
    [(_, scored_spec)] = discover_specs("robolab", ["banana_on_plate"])
    [(_, robot_only_spec)] = discover_specs("kitchen_bench", ["kitchen_bench_lightwheel_open_fridge"])

    scored = measure_environment("robolab", scored_spec, settings, seed_block_index=0)
    robot_only = measure_environment("kitchen_bench", robot_only_spec, settings, seed_block_index=1)
    failed = measure_environment("robolab", scored_spec.with_name("missing_spec.yaml"), settings, seed_block_index=2)

    assert scored.status == EnvironmentStatus.SCORED, scored.reason
    assert scored.num_layouts == 2 and scored.checks
    assert robot_only.status == EnvironmentStatus.SCORED, robot_only.reason
    assert robot_only.num_layouts == 2
    assert failed.status == EnvironmentStatus.FAILED and failed.reason
    return True


def test_real_specs_are_scored_or_failed():
    """An object spec and a spec that places only the robot are scored, and a spec that cannot be built fails."""
    assert run_function_with_persistent_simulation_app(_test_real_specs_are_scored_or_failed)


def test_results_file_round_trips_and_reports_suite_rates(tmp_path):
    """Suites report counts for every status, and equal-weight mean rates only when no environment failed."""
    results = _results([
        _scored("robolab", "pick_apple", layouts=(45, 50), candidates=(300, 500)),
        _scored("robolab", "pick_banana", layouts=(25, 25), candidates=(250, 250)),
        _scored("kitchen_bench", "pick_cup", layouts=(50, 50), candidates=(500, 500)),
        EnvironmentRecord("kitchen_bench", "static_scene", EnvironmentStatus.SKIPPED, reason="places only anchors"),
        EnvironmentRecord("kitchen_bench", "place_lime", EnvironmentStatus.FAILED, reason="KeyError: 'lime'"),
    ])
    results.settings = PlacementBenchmarkSettings(
        random_yaw_init=True, collision_mode=CollisionMode.MESH, solver_device="cpu"
    )
    path = tmp_path / "run.json"
    results.save(path)

    assert PlacementBenchmarkResults.load(path) == results
    robolab, kitchen_bench = summarize_suites(results)
    # Every environment weighs the same, whatever its layout count.
    assert robolab.layout_success_rate == pytest.approx(0.95)
    assert robolab.valid_candidate_rate == pytest.approx(0.8)
    assert (kitchen_bench.num_scored, kitchen_bench.num_skipped, kitchen_bench.num_failed) == (1, 1, 1)
    # A failed environment could have lowered the suite's rates, so they are left out.
    assert kitchen_bench.layout_success_rate is None
    assert "| kitchen_bench | 1 | 1 | 1 | n/a | n/a |" in format_results(results)


def test_comparison_flags_only_changes_that_survive_holm():
    """Only changes that survive Holm's step-down are flagged, and unpaired environments and settings are listed."""
    before = [_scored("robolab", f"task_{index}", layouts=(47, 50), candidates=(400, 500)) for index in range(38)]
    before.append(EnvironmentRecord("robolab", "new_task", EnvironmentStatus.FAILED, reason="missing asset"))
    before.append(EnvironmentRecord("robolab", "static_scene", EnvironmentStatus.SKIPPED))
    after = [_scored("robolab", f"task_{index}", layouts=(47, 50), candidates=(400, 500)) for index in range(35)]
    after.append(EnvironmentRecord("robolab", "task_35", EnvironmentStatus.SKIPPED))
    after.append(_scored("robolab", "task_36", layouts=(47, 50), candidates=(400, 500)))
    after.append(_scored("robolab", "task_37", layouts=(12, 25), candidates=(100, 250)))
    after.append(EnvironmentRecord("robolab", "static_scene", EnvironmentStatus.SKIPPED))
    after_results = _results(after)
    after_results.settings = PlacementBenchmarkSettings(random_yaw_init=True, collision_mode=CollisionMode.MESH)

    comparison = compare_results(_results(before), after_results)

    [suite] = comparison.suites
    assert suite.num_environments == 37
    # Every environment weighs the same, whatever its layout count.
    assert suite.layout_success.after == pytest.approx((36 * 0.94 + 0.48) / 37)
    changes = {environment.environment: environment for environment in comparison.environments}
    # Agresti-Caffo for 47/50 -> 12/25: adjusted rates 48/52 and 13/27.
    assert (changes["task_37"].layout_success.low, changes["task_37"].layout_success.high) == pytest.approx(
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
        _results([_scored("robolab", "task_0", layouts=(47, 50), candidates=(400, 500))]),
        _results([_scored("robolab", "task_0", layouts=(40, 50), candidates=(372, 500))]),
    )
    [environment] = moderate.environments
    assert max(environment.layout_success.p_value, environment.valid_candidates.p_value) < 0.05
    assert not environment.significant

    # After two near-zero p-values, task_1's p = 0.014 is checked against 5% / 2 rather than Bonferroni's 5% / 4.
    step_down = compare_results(
        _results([_scored("robolab", f"task_{index}", layouts=(47, 50), candidates=(400, 500)) for index in range(2)]),
        _results([
            _scored("robolab", "task_0", layouts=(20, 50), candidates=(200, 500)),
            _scored("robolab", "task_1", layouts=(38, 50), candidates=(400, 500)),
        ]),
    )
    assert [environment.significant for environment in step_down.environments] == [True, True]


def test_suite_interval_is_centered_on_the_observed_change():
    """Identical rates show no change even when the runs differ in size, and the p-value agrees with the interval.

    A difference of two rates cannot pass 100 points, so the printed interval stops there, while the stored interval
    keeps its full width for suite averages.
    """
    before = [_scored("robolab", f"task_{index}", layouts=(150, 150), candidates=(1500, 1500)) for index in range(36)]
    after = [_scored("robolab", f"task_{index}", layouts=(50, 50), candidates=(500, 500)) for index in range(36)]

    [suite] = compare_results(_results(before), _results(after)).suites
    [single] = compare_results(_results(before[:1]), _results(after[:1])).suites

    assert suite.layout_success.low == pytest.approx(-suite.layout_success.high)
    assert suite.layout_success.p_value == pytest.approx(1.0)
    # Independent environment errors add in quadrature, so 36 equal environments narrow the interval six times.
    single_width = single.layout_success.high - single.layout_success.low
    assert suite.layout_success.high - suite.layout_success.low == pytest.approx(single_width / 6)

    # Small drops in every environment: the p-value is below 5% exactly when the interval excludes zero.
    before = [_scored("robolab", f"task_{index}", layouts=(47, 50), candidates=(400, 500)) for index in range(36)]
    for successful_layouts in range(44, 48):
        after = [_scored("robolab", f"task_{index}", (successful_layouts, 50), (400, 500)) for index in range(36)]
        [suite] = compare_results(_results(before), _results(after)).suites
        change = suite.layout_success
        assert (change.high < 0 or change.low > 0) == (change.p_value < 0.05)

    collapse = compare_results(
        _results([_scored("robolab", "task_0", layouts=(50, 50), candidates=(500, 500))]),
        _results([_scored("robolab", "task_0", layouts=(0, 50), candidates=(0, 500))]),
    )
    assert collapse.environments[0].layout_success.low < -1
    assert "| robolab | 1 | 100.0% → 0.0% | [-100.0, " in format_comparison(collapse)


def test_comparison_leaves_out_environments_whose_success_criteria_changed():
    """An environment is left out when a check required in only one run rejected candidates in that run."""
    before = [_scored("robolab", f"task_{index}", layouts=(47, 50), candidates=(400, 500)) for index in range(5)]
    after = [_scored("robolab", f"task_{index}", layouts=(25, 50), candidates=(200, 500)) for index in range(5)]
    for record in before + after:
        record.checks = {"no_overlap": True, "ik_reachable": True}
    for record in before:
        record.candidate_failures_per_check = {"ik_reachable": 40}
    # task_0 loses ik_reachable and task_1 makes it optional; it rejected candidates before.
    after[0].checks = {"no_overlap": True}
    after[1].checks = {"no_overlap": True, "ik_reachable": False}
    # task_2 gains a required check that never rejects a candidate, so its success is unchanged.
    after[2].checks = {"no_overlap": True, "ik_reachable": True, "face_to": True}
    # task_3 gains a required check that rejects candidates after the change.
    after[3].checks = {"no_overlap": True, "ik_reachable": True, "on_relation": True}
    after[3].candidate_failures_per_check = {"on_relation": 10}
    # task_4 gains an optional check, which does not affect success even though it rejects candidates.
    after[4].checks = {"no_overlap": True, "ik_reachable": True, "face_to": False}
    after[4].candidate_failures_per_check = {"face_to": 10}

    comparison = compare_results(_results(before), _results(after))

    assert [environment.environment for environment in comparison.environments] == ["task_2", "task_4"]
    assert [excluded.environment for excluded in comparison.check_differences] == ["task_0", "task_1", "task_3"]
    assert (
        "- robolab/task_1: before ik_reachable, no_overlap; after ik_reachable (optional), no_overlap;"
        " layout success 94.0% → 50.0%, valid candidates 80.0% → 40.0%"
        in format_comparison(comparison)
    )


def test_seed_results_count_each_layout_by_its_best_candidate():
    """Layouts count their best candidate, every candidate counts toward validity, and failures name their checks."""
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.validation.types import PlacementValidationResults
    from isaaclab_arena.scripts.placement_success_benchmark import add_seed_results

    def candidate(no_overlap: bool, on_relation: bool, ik_reachable: bool = True) -> PlacementResult:
        # As in PlacementValidationRunner, IK runs only on candidates that pass the required checks.
        ik_reachable = ik_reachable and no_overlap and on_relation
        checks = {"no_overlap": no_overlap, "on_relation": on_relation, "ik_reachable": ik_reachable}
        validation = PlacementValidationResults(checks, required_checks={"no_overlap", "on_relation"})
        return PlacementResult(validation, positions={}, final_loss=0.0, attempts=1)

    record = EnvironmentRecord("robolab", "pick_apple", EnvironmentStatus.SCORED)
    add_seed_results(
        record,
        [
            [candidate(True, True, ik_reachable=False), candidate(True, True), candidate(False, True)],
            [candidate(False, True), candidate(False, False)],
        ],
    )
    add_seed_results(record, [[candidate(True, False), candidate(True, False)]])

    assert (record.num_successful_layouts, record.num_layouts) == (1, 3)
    assert (record.num_valid_candidates, record.num_candidates) == (2, 7)
    assert record.checks == {"no_overlap": True, "on_relation": True, "ik_reachable": False}
    assert record.candidate_failures_per_check == {"ik_reachable": 6, "no_overlap": 3, "on_relation": 3}

    # Without declared required checks, every check is required.
    default_record = EnvironmentRecord("robolab", "pick_apple", EnvironmentStatus.SCORED)
    validation = PlacementValidationResults({"no_overlap": True, "on_relation": False})
    add_seed_results(default_record, [[PlacementResult(validation, positions={}, final_loss=0.0, attempts=1)]])
    assert default_record.checks == {"no_overlap": True, "on_relation": True}


def test_candidate_placer_matches_object_placer_with_disjoint_seeds():
    """Each layout's best candidate is ObjectPlacer.place()'s, and seeds and environments solve different candidates.

    Two environments with the same scene must not share candidates, or their results would not be independent.
    """
    from isaaclab_arena.environments.arena_env_builder import LivePlacementInputs
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.scripts.placement_success_benchmark import _SEEDS_PER_ENVIRONMENT, CandidatePlacer
    from isaaclab_arena.tests.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    desk = DummyObject(name="desk", bounding_box=AxisAlignedBoundingBox((0.0, 0.0, 0.0), (1.0, 1.0, 0.1)))
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())
    box = DummyObject(name="box", bounding_box=AxisAlignedBoundingBox((0.0, 0.0, 0.0), (0.2, 0.2, 0.2)))
    box.add_relation(On(desk, clearance_m=0.01))
    params = ObjectPlacerParams(
        solver_params=RelationSolverParams(max_iters=10, verbose=False), apply_positions_to_objects=False
    )
    num_layouts = 5
    inputs = LivePlacementInputs([desk, box], params, scene_assets=[])
    settings = PlacementBenchmarkSettings(num_layouts=num_layouts)

    box_positions_per_sample = []
    for seed_block_index, seed in ((0, 0), (0, 1), (1, 0)):
        placer = CandidatePlacer(inputs, settings, seed_block_index)
        reference_params = copy.copy(params)
        reference_params.placement_seed = (
            seed_block_index * _SEEDS_PER_ENVIRONMENT + seed * num_layouts * params.max_placement_attempts
        )
        expected_per_layout = ObjectPlacer(reference_params).place([desk, box], num_envs=num_layouts)
        ranked_per_layout = placer.place(seed)
        box_positions = set()
        for ranked, expected in zip(ranked_per_layout, expected_per_layout, strict=True):
            assert len(ranked) == params.max_placement_attempts
            assert ranked[0].positions == expected.positions
            for result in ranked:
                box_positions.add(tuple(result.positions[box]))
        box_positions_per_sample.append(box_positions)
    overridden_settings = PlacementBenchmarkSettings(
        num_layouts=num_layouts, random_yaw_init=True, collision_mode=CollisionMode.MESH
    )
    overridden = CandidatePlacer(inputs, overridden_settings, 0)
    overridden_params = overridden._placer.params
    assert overridden_params.solver_params.collision_mode == CollisionMode.MESH and overridden_params.random_yaw_init
    assert params.solver_params.collision_mode != CollisionMode.MESH and not params.random_yaw_init

    first_seed, second_seed, other_environment = box_positions_per_sample
    assert not first_seed & second_seed
    assert not first_seed & other_environment
    # A seed whose candidates would run into the next environment's block is refused.
    with pytest.raises(AssertionError):
        placer.place(_SEEDS_PER_ENVIRONMENT // (num_layouts * params.max_placement_attempts))


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
    _results([_scored("robolab", "pick_apple", layouts=(45, 50), candidates=(300, 500))]).save(before_path)
    _results([_scored("robolab", "pick_apple", layouts=(50, 50), candidates=(450, 500))]).save(after_path)
    argv = ["placement_success_benchmark.py", "--compare", str(before_path), str(after_path)]
    monkeypatch.setattr(sys, "argv", argv)

    placement_success_benchmark.main()

    assert "| robolab | 1 | 90.0% → 100.0% |" in capsys.readouterr().out


def _scored(suite: str, environment: str, layouts: tuple[int, int], candidates: tuple[int, int]) -> EnvironmentRecord:
    return EnvironmentRecord(
        suite,
        environment,
        EnvironmentStatus.SCORED,
        num_layouts=layouts[1],
        num_successful_layouts=layouts[0],
        num_candidates=candidates[1],
        num_valid_candidates=candidates[0],
    )


def _results(records: list[EnvironmentRecord]) -> PlacementBenchmarkResults:
    return PlacementBenchmarkResults(
        "2026-10-21T00:00:00+00:00", "abc123", "main", False, PlacementBenchmarkSettings(), records
    )
