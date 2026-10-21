# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for placement-success KPI files, run comparisons and the benchmark script."""

import copy
import subprocess

import pytest

from isaaclab_arena.offline_placement.placement_success import (
    EnvironmentRecord,
    EnvironmentStatus,
    PlacementBenchmarkRun,
    compare_runs,
    format_run,
)
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def test_kpi_file_round_trips_and_reports_suite_and_environment_rates(tmp_path):
    """Suites report counts for every status, and equal-weight mean rates only when no environment failed."""
    run = _run([
        _scored("robolab", "pick_apple", layouts=(45, 50), candidates=(300, 500)),
        _scored("robolab", "pick_banana", layouts=(25, 25), candidates=(250, 250)),
        _scored("kitchen_bench", "pick_cup", layouts=(50, 50), candidates=(500, 500)),
        EnvironmentRecord("kitchen_bench", "open_fridge", EnvironmentStatus.SKIPPED, reason="places only the robot"),
        EnvironmentRecord("kitchen_bench", "place_lime", EnvironmentStatus.ERROR, reason="KeyError: 'lime'"),
    ])
    path = tmp_path / "run.json"
    run.save(path)

    assert PlacementBenchmarkRun.load(path) == run
    kpis = {(kpi["name"], tuple(sorted(kpi["tags"].items()))): kpi["value"] for kpi in run.get_kpis()}
    assert kpis[("layout_success_rate", (("suite", "robolab"),))] == pytest.approx(0.95)
    assert kpis[("valid_candidate_rate", (("suite", "robolab"),))] == pytest.approx(0.8)
    assert kpis[("layout_success_rate", (("environment", "pick_apple"), ("suite", "robolab")))] == pytest.approx(0.9)
    assert kpis[("environments_skipped", (("suite", "kitchen_bench"),))] == 1
    assert kpis[("environments_failed", (("suite", "kitchen_bench"),))] == 1
    # A failed environment could have lowered the suite's rates, so they are left out.
    assert ("layout_success_rate", (("suite", "kitchen_bench"),)) not in kpis
    assert kpis[("layout_success_rate", (("environment", "pick_cup"), ("suite", "kitchen_bench")))] == 1.0
    assert "| kitchen_bench | 1 | 1 | 1 | n/a | n/a |" in format_run(run)


def test_comparison_flags_only_changes_that_survive_holm():
    """Only changes that survive Holm's step-down are flagged, and environments not scored in both runs are listed."""
    baseline = [_scored("robolab", f"task_{index}", layouts=(47, 50), candidates=(400, 500)) for index in range(38)]
    baseline.append(EnvironmentRecord("robolab", "new_task", EnvironmentStatus.ERROR, reason="missing asset"))
    baseline.append(EnvironmentRecord("robolab", "open_fridge", EnvironmentStatus.SKIPPED))
    candidate = [_scored("robolab", f"task_{index}", layouts=(47, 50), candidates=(400, 500)) for index in range(35)]
    candidate.append(EnvironmentRecord("robolab", "task_35", EnvironmentStatus.SKIPPED))
    candidate.append(_scored("robolab", "task_36", layouts=(47, 50), candidates=(400, 500)))
    candidate.append(_scored("robolab", "task_37", layouts=(12, 25), candidates=(100, 250)))
    candidate[-1].checks = ["ik_reachable"]
    candidate.append(EnvironmentRecord("robolab", "open_fridge", EnvironmentStatus.SKIPPED))
    candidate_run = _run(candidate)
    candidate_run.settings = {"num_layouts": 50, "random_yaw_init": True}

    comparison = compare_runs(_run(baseline), candidate_run)

    [suite] = comparison.suites
    assert suite.environments == 37
    # Every environment weighs the same, whatever its layout count.
    assert suite.layout_success.candidate == pytest.approx((36 * 0.94 + 0.48) / 37)
    changes = {environment.environment: environment for environment in comparison.environments}
    # Agresti-Caffo for 47/50 -> 12/25: adjusted rates 48/52 and 13/27.
    assert (changes["task_37"].layout_success.low, changes["task_37"].layout_success.high) == pytest.approx(
        (-0.64350, -0.23969), abs=1e-4
    )
    assert [name for name, environment in changes.items() if environment.significant] == ["task_37"]
    assert comparison.unpaired == [
        ("robolab", "new_task", "error", "missing"),
        ("robolab", "task_35", "scored", "skipped"),
    ]
    assert comparison.settings_differences == [("random_yaw_init", None, True), ("checks", [], ["ik_reachable"])]

    # p = 0.035 and 0.047: both are below 5%, but the smaller misses Holm's first threshold of 2.5%, so neither is flagged.
    moderate = compare_runs(
        _run([_scored("robolab", "task_0", layouts=(47, 50), candidates=(400, 500))]),
        _run([_scored("robolab", "task_0", layouts=(40, 50), candidates=(372, 500))]),
    )
    [environment] = moderate.environments
    assert max(environment.layout_success.p_value, environment.valid_candidates.p_value) < 0.05
    assert not environment.significant


def test_git_provenance_ignores_only_the_output_file(tmp_path, monkeypatch):
    """The run's own KPI file leaves the checkout clean, while a source change marks it dirty."""
    from isaaclab_arena.scripts import placement_success_benchmark

    monkeypatch.setattr(placement_success_benchmark, "_REPO_ROOT", tmp_path.resolve())
    git = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com"]
    (tmp_path / "source.py").write_text("")
    subprocess.run([*git, "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run([*git, "add", "source.py"], cwd=tmp_path, check=True)
    subprocess.run([*git, "commit", "-qm", "source"], cwd=tmp_path, check=True)
    output = tmp_path / "run.json"
    output.write_text("{}")

    assert placement_success_benchmark.get_git_provenance(output)[2] is False
    (tmp_path / "source.py").write_text("changed = True\n")
    assert placement_success_benchmark.get_git_provenance(output)[2] is True


def test_seed_results_count_each_layout_by_its_best_candidate():
    """Layouts count their best candidate, every candidate counts toward validity, and failures name their checks."""
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.validation.types import PlacementValidationResults
    from isaaclab_arena.scripts.placement_success_benchmark import add_seed_results

    def candidate(no_overlap: bool, on_relation: bool, ik_reachable: bool = True) -> PlacementResult:
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

    assert (record.layout_successes, record.num_layouts) == (1, 3)
    assert (record.candidate_successes, record.num_candidates) == (2, 7)
    assert record.failed_checks == {"no_overlap": 1, "on_relation": 1}
    assert record.checks == ["no_overlap", "on_relation", "ik_reachable"]


def test_benchmark_placement_matches_object_placer_with_disjoint_seeds():
    """Each layout's best candidate is ObjectPlacer.place()'s, and consecutive seeds solve different candidates."""
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.scripts.placement_success_benchmark import BenchmarkPlacement
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
    placement = BenchmarkPlacement([desk, box], params, num_layouts=num_layouts)

    box_positions_per_seed = []
    for seed in (0, 1):
        reference_params = copy.copy(params)
        reference_params.placement_seed = seed * num_layouts * params.max_placement_attempts
        expected_per_layout = ObjectPlacer(reference_params).place([desk, box], num_envs=num_layouts)
        ranked_per_layout = placement.place(seed, collision_objects=[])
        box_positions = set()
        for ranked, expected in zip(ranked_per_layout, expected_per_layout, strict=True):
            assert len(ranked) == params.max_placement_attempts
            assert ranked[0].positions == expected.positions
            for result in ranked:
                box_positions.add(tuple(result.positions[box]))
        box_positions_per_seed.append(box_positions)
    assert not box_positions_per_seed[0] & box_positions_per_seed[1]


def _test_real_specs_are_scored_or_skipped(simulation_app) -> bool:
    from isaaclab_arena.scripts.placement_success_benchmark import (
        discover_specs,
        get_benchmark_parser,
        measure_environment,
    )
    from isaaclab_arena_environments.cli import ensure_environments_registered

    ensure_environments_registered()
    args_cli = get_benchmark_parser().parse_args(["--num_layouts", "2", "--placement_seeds", "0"])
    [(_, scored_spec)] = discover_specs("robolab", ["banana_on_plate"])
    [(_, robot_only_spec)] = discover_specs("kitchen_bench", ["kitchen_bench_lightwheel_open_fridge"])

    scored = measure_environment("robolab", scored_spec, args_cli)
    skipped = measure_environment("kitchen_bench", robot_only_spec, args_cli)

    assert scored.status == EnvironmentStatus.SCORED, scored.reason
    assert scored.num_layouts == 2 and scored.checks
    assert skipped.status == EnvironmentStatus.SKIPPED, skipped.reason
    return True


def test_real_specs_are_scored_or_skipped():
    """A RoboLab spec is placed and scored, and a KitchenBench spec that places only the robot is skipped."""
    assert run_function_with_persistent_simulation_app(_test_real_specs_are_scored_or_skipped)


def test_spec_discovery_covers_both_suites_and_filters_by_stem():
    """Every suite contributes specs, and a stem glob keeps only matching specs."""
    from isaaclab_arena.scripts.placement_success_benchmark import discover_specs

    specs = discover_specs("all", None)
    assert {suite for suite, _ in specs} == {"robolab", "kitchen_bench"}
    assert all(path.is_file() for _, path in specs)
    stem = specs[0][1].stem
    assert [path.stem for _, path in discover_specs("all", [stem])] == [stem]


def test_compare_cli_prints_suite_table_without_simulator(tmp_path):
    """--compare reads two KPI files and prints the suite comparison."""
    baseline_path, candidate_path = tmp_path / "baseline.json", tmp_path / "candidate.json"
    _run([_scored("robolab", "pick_apple", layouts=(45, 50), candidates=(300, 500))]).save(baseline_path)
    _run([_scored("robolab", "pick_apple", layouts=(50, 50), candidates=(450, 500))]).save(candidate_path)

    completed = subprocess.run(
        [
            TestConstants.python_path,
            f"{TestConstants.scripts_dir}/placement_success_benchmark.py",
            "--compare",
            str(baseline_path),
            str(candidate_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "| robolab | 1 | 90.0% → 100.0% |" in completed.stdout


def _scored(suite: str, environment: str, layouts: tuple[int, int], candidates: tuple[int, int]) -> EnvironmentRecord:
    return EnvironmentRecord(
        suite,
        environment,
        EnvironmentStatus.SCORED,
        num_layouts=layouts[1],
        layout_successes=layouts[0],
        num_candidates=candidates[1],
        candidate_successes=candidates[0],
    )


def _run(records: list[EnvironmentRecord]) -> PlacementBenchmarkRun:
    return PlacementBenchmarkRun("2026-10-21T00:00:00+00:00", "abc123", "main", False, {"num_layouts": 50}, records)
