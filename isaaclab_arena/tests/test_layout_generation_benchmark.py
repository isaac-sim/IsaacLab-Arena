# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import random
from dataclasses import replace

import pytest

from isaaclab_arena.relations.benchmark import layout_throughput
from isaaclab_arena.relations.benchmark.layout_generation import (
    LayoutGenerationRun,
    LayoutPose,
    LayoutSample,
    _ks_uniform,
    build_arena_controlled_scene,
    canonical_layout_key,
    compare_layout_runs,
    explicit_layout,
    format_layout_markdown,
    make_object_sizes,
    sample_random_rejection,
    summarize_layout_run,
    validate_controlled_layout,
)
from isaaclab_arena.relations.benchmark.layout_throughput import (
    LayoutThroughputRun,
    check_throughput_compatibility,
    format_throughput_markdown,
    run_controlled_throughput,
    run_controlled_throughput_sample,
)
from isaaclab_arena.relations.relations import On
from isaaclab_arena_examples.relations import collision_representation_ablation
from isaaclab_arena_examples.relations import collision_space_coverage_benchmark as coverage_benchmark
from isaaclab_arena_examples.relations import direct_solver_comparison_benchmark as direct_benchmark
from isaaclab_arena_examples.relations import fixed_iteration_batch_scaling_benchmark as batch_scaling_benchmark
from isaaclab_arena_examples.relations import obstacle_layout_distribution_benchmark as obstacle_benchmark
from isaaclab_arena_examples.relations import robolab_layout_generation_benchmark as robolab_benchmark

BOUNDS = (-0.5, 0.5, -0.5, 0.5)
SIZES = {"a": (0.2, 0.2), "b": (0.2, 0.2)}


def _run(samples):
    return LayoutGenerationRun("controlled", "abc", 0, 7, None, BOUNDS, SIZES, tuple(samples))


def test_layout_generation_json_round_trip_and_strict_finite_validation():
    sample = sample_random_rejection(seed=7, table_xy_bounds=BOUNDS, object_xy_sizes=SIZES)
    run = _run([sample])
    assert LayoutGenerationRun.from_dict(json.loads(json.dumps(run.to_dict()))) == run

    payload = run.to_dict()
    payload["samples"][0]["elapsed_ms"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        LayoutGenerationRun.from_dict(payload)

    payload = run.to_dict()
    payload["unknown"] = True
    with pytest.raises(ValueError, match="unknown"):
        LayoutGenerationRun.from_dict(payload)


def test_shared_validator_requires_complete_contained_non_overlapping_layout():
    valid = (LayoutPose("a", -0.2, 0.0), LayoutPose("b", 0.2, 0.0))
    assert validate_controlled_layout(valid, BOUNDS, SIZES)
    assert not validate_controlled_layout(valid[:1], BOUNDS, SIZES)
    assert not validate_controlled_layout((LayoutPose("a", 0.0, 0.0), LayoutPose("b", 0.1, 0.0)), BOUNDS, SIZES)
    assert not validate_controlled_layout((LayoutPose("a", -0.2, 0.0), LayoutPose("b", 0.45, 0.0)), BOUNDS, SIZES)


def test_arena_controlled_scene_uses_zero_edge_margin():
    scene = build_arena_controlled_scene(BOUNDS, SIZES)
    relations = [relation for obj in scene[1:] for relation in obj.get_relations() if isinstance(relation, On)]
    assert relations
    assert all(relation.edge_margin_m == 0.0 for relation in relations)


def test_random_rejection_is_deterministic_diverse_and_bounded():
    first = sample_random_rejection(seed=11, table_xy_bounds=BOUNDS, object_xy_sizes=SIZES)
    second = sample_random_rejection(seed=11, table_xy_bounds=BOUNDS, object_xy_sizes=SIZES)
    third = sample_random_rejection(seed=12, table_xy_bounds=BOUNDS, object_xy_sizes=SIZES)
    assert first == replace(second, elapsed_ms=first.elapsed_ms)
    assert canonical_layout_key(first.poses) != canonical_layout_key(third.poses)

    impossible_sizes = {"a": (0.75, 0.75), "b": (0.75, 0.75)}
    failed = sample_random_rejection(
        seed=0,
        table_xy_bounds=BOUNDS,
        object_xy_sizes=impossible_sizes,
        max_attempts=4,
    )
    assert not failed.success
    assert failed.attempts == 4


def test_random_one_object_sampling_uniformity_and_bias_detection():
    sizes = {"a": (0.1, 0.1)}
    x_values = [
        sample_random_rejection(seed=seed, table_xy_bounds=BOUNDS, object_xy_sizes=sizes).poses[0].x
        for seed in range(500)
    ]
    _statistic, uniform_p = _ks_uniform(x_values, -0.45, 0.45)
    _biased_statistic, biased_p = _ks_uniform([-0.44] * 500, -0.45, 0.45)
    assert uniform_p is not None and uniform_p >= 0.01
    assert biased_p is not None and biased_p < 0.001


def test_layout_summary_reports_latency_determinism_uniqueness_and_caveat():
    sample = sample_random_rejection(seed=7, table_xy_bounds=BOUNDS, object_xy_sizes=SIZES)
    repeated = replace(sample, elapsed_ms=(sample.elapsed_ms or 0.0) + 1.0)
    summary = summarize_layout_run(_run([sample, repeated]))
    report = format_layout_markdown([_run([sample, repeated])])
    assert summary.success_rate == 1.0
    assert summary.same_seed_deterministic is True
    assert summary.unique_layouts == 1
    assert summary.median_successful_attempt_latency_ms is not None
    assert "p >= 0.05 does not prove unbiased sampling" in report


def test_layout_comparison_rejects_workload_geometry_and_seed_mismatch():
    sample = sample_random_rejection(seed=7, table_xy_bounds=BOUNDS, object_xy_sizes=SIZES)
    baseline = _run([sample])
    for mismatch in (
        replace(baseline, workload="other"),
        replace(baseline, master_seed=8),
        replace(baseline, object_xy_sizes={"a": (0.1, 0.1), "b": (0.2, 0.2)}),
    ):
        with pytest.raises(ValueError, match="mismatched"):
            compare_layout_runs([baseline, mismatch])


def test_explicit_is_one_attempt_capacity_baseline_with_na_timing():
    sample = run_controlled_throughput_sample(
        "explicit",
        seed=0,
        target_layouts=1,
        table_xy_bounds=BOUNDS,
        object_xy_sizes=SIZES,
    )
    assert sample.target_reached
    assert sample.attempted_layouts == 1
    assert sample.elapsed_ms is sample.layouts_per_second is None
    assert not sample.timing_applicable

    over_capacity = replace(sample, target_layouts=2, target_reached=False, error="capacity")
    assert over_capacity.attempted_layouts == 1
    assert not over_capacity.target_reached


def test_explicit_requested_above_capacity_is_retained_as_failure():
    sample = run_controlled_throughput_sample(
        "explicit",
        seed=0,
        target_layouts=2,
        table_xy_bounds=BOUNDS,
        object_xy_sizes=SIZES,
    )
    assert sample.unique_layouts == 1
    assert sample.attempted_layouts == sample.accepted_layouts == 1
    assert not sample.target_reached


def test_random_throughput_reaches_exact_k_with_accounting():
    sample = run_controlled_throughput_sample(
        "random_rejection",
        seed=4,
        target_layouts=3,
        table_xy_bounds=BOUNDS,
        object_xy_sizes=SIZES,
        max_attempts_per_layout=10,
    )
    assert sample.target_reached
    assert sample.unique_layouts == 3
    assert sample.accepted_layouts <= sample.attempted_layouts <= 30
    assert sample.validity_rate == sample.accepted_layouts / sample.attempted_layouts


def test_random_throughput_retains_bounded_failure():
    sample = run_controlled_throughput_sample(
        "random_rejection",
        seed=4,
        target_layouts=2,
        table_xy_bounds=BOUNDS,
        object_xy_sizes={"a": (0.75, 0.75), "b": (0.75, 0.75)},
        max_attempts_per_layout=3,
    )
    assert not sample.target_reached
    assert sample.attempted_layouts == 6
    assert sample.accepted_layouts == sample.unique_layouts == 0


def test_arena_throughput_counts_one_candidate_per_attempt(monkeypatch):
    calls = []

    def fake_sample_arena_batch(**kwargs):
        calls.append(kwargs)
        return tuple(
            LayoutSample(
                "arena",
                kwargs["seed"] + index,
                not (kwargs["seed"] == 10 and index == 0),
                1,
                1.0,
                (LayoutPose("a", (kwargs["seed"] + index) / 100.0, 0.0),),
            )
            for index in range(kwargs["num_layouts"])
        )

    monkeypatch.setattr(layout_throughput, "sample_arena_batch", fake_sample_arena_batch)
    clock = iter((0.0, 0.004)).__next__
    sample = layout_throughput.run_controlled_throughput_sample(
        "arena",
        seed=10,
        target_layouts=3,
        table_xy_bounds=BOUNDS,
        object_xy_sizes={"a": (0.01, 0.01)},
        max_attempts_per_layout=4,
        clock=clock,
    )

    assert sample.target_reached
    assert sample.attempted_layouts == 4
    assert sample.elapsed_ms == 4.0
    assert [call["seed"] for call in calls] == [10, 13]
    assert all(call["max_attempts_per_layout"] == 1 for call in calls)


def test_throughput_json_round_trip_and_report_fields():
    run = run_controlled_throughput(
        "explicit",
        target_layout_counts=(1, 2),
        repetitions=2,
        master_seed=5,
        table_xy_bounds=BOUNDS,
        object_xy_sizes=SIZES,
        max_attempts_per_layout=3,
    )
    restored = LayoutThroughputRun.from_dict(json.loads(json.dumps(run.to_dict())))
    assert restored == run
    assert run.seeds == (5, 11)
    legacy_payload = run.to_dict()
    legacy_payload.pop("solver_config")
    assert LayoutThroughputRun.from_dict(legacy_payload).solver_config == {}
    report = format_throughput_markdown([run])
    assert "Targets reached" in report
    assert "Attempted" in report
    assert "Explicit poses are a one-layout capacity baseline" in report
    assert "N/A" in report


@pytest.mark.parametrize(
    "change",
    [
        {"workload": "other"},
        {"object_xy_sizes": {"a": (0.1, 0.1), "b": (0.2, 0.2)}},
        {"max_attempts_per_layout": 4},
        {"repetitions": 1, "seeds": (5,), "samples": ()},
        {"target_layout_counts": (1,), "samples": ()},
    ],
)
def test_throughput_compatibility_rejects_mismatches(change):
    baseline = run_controlled_throughput(
        "explicit",
        target_layout_counts=(1, 2),
        repetitions=2,
        master_seed=5,
        table_xy_bounds=BOUNDS,
        object_xy_sizes=SIZES,
        max_attempts_per_layout=3,
    )
    if "samples" in change:
        with pytest.raises(ValueError):
            replace(baseline, **change)
    else:
        mismatch = replace(baseline, **change)
        with pytest.raises(ValueError, match="mismatched"):
            check_throughput_compatibility([baseline, mismatch])


def test_throughput_compatibility_rejects_seed_mismatch():
    baseline = run_controlled_throughput(
        "explicit",
        target_layout_counts=(1,),
        repetitions=2,
        master_seed=5,
        table_xy_bounds=BOUNDS,
        object_xy_sizes=SIZES,
        max_attempts_per_layout=3,
    )
    mismatch = replace(
        baseline,
        seeds=tuple(seed + 3 for seed in baseline.seeds),
        samples=tuple(replace(sample, seed=sample.seed + 3) for sample in baseline.samples),
    )
    with pytest.raises(ValueError, match="mismatched seeds"):
        check_throughput_compatibility([baseline, mismatch])


def test_throughput_compatibility_rejects_same_method_configuration_mismatch():
    baseline = run_controlled_throughput(
        "explicit",
        target_layout_counts=(1,),
        table_xy_bounds=BOUNDS,
        object_xy_sizes=SIZES,
    )
    mismatch = replace(baseline, solver_config={**baseline.solver_config, "variant": "different"})
    with pytest.raises(ValueError, match="mismatched configuration"):
        check_throughput_compatibility([baseline, mismatch])


def test_authored_layout_validity_depends_on_requested_density():
    assert explicit_layout(table_xy_bounds=BOUNDS, object_xy_sizes=make_object_sizes(4, 0.2)).success
    assert not explicit_layout(table_xy_bounds=BOUNDS, object_xy_sizes=make_object_sizes(4, 0.8)).success


def test_robolab_adapter_calls_spatial_solver(monkeypatch):
    calls = []

    class FakeState:
        def __init__(self, name, predicates):
            self.name = name
            self.predicates = predicates
            self.x = self.y = self.yaw = None

    class FakePredicate:
        def __init__(self, name, yaw):
            self.name = name
            self.yaw = yaw

    class FakeSolver:
        def __init__(self, table_bounds, collision_margin):
            calls.append(("init", table_bounds, collision_margin))

        def solve(self, states, dimensions, max_iterations, allow_relaxation):
            calls.append(("solve", dimensions, max_iterations, allow_relaxation))
            for index, state in enumerate(states.values()):
                state.x = -0.2 + index * 0.4
                state.y = 0.0
                state.yaw = 0.0
            return True, "solved"

    monkeypatch.setattr(
        robolab_benchmark,
        "_load_robolab_solver",
        lambda _root: (FakeState, FakePredicate, FakeSolver),
    )
    solve_layout = robolab_benchmark._make_solve_layout(
        robolab_root=None,
        bounds=BOUNDS,
        sizes=SIZES,
        collision_margin_m=0.0,
        max_iterations=123,
        allow_relaxation=False,
    )

    success, poses, error, elapsed_ms = solve_layout(7)

    assert success
    assert error is None
    assert elapsed_ms >= 0.0
    assert len(poses) == 2
    assert calls == [
        ("init", BOUNDS, 0.0),
        ("solve", {"a": (0.2, 0.2, 0.1), "b": (0.2, 0.2, 0.1)}, 123, False),
    ]
    assert random.Random(7).random() == pytest.approx(random.random())


def test_robolab_throughput_retries_with_distinct_deterministic_seeds(monkeypatch):
    observed_seeds = []

    def solve_layout(seed):
        observed_seeds.append(seed)
        x = 0.0 if seed < 12 else 0.25
        pose = [{"object_name": "a", "x": x, "y": 0.0, "z": 0.05, "yaw": 0.0}]
        return True, pose, None, 1.0

    clock = iter((0.0, 0.003)).__next__
    monkeypatch.setattr(robolab_benchmark.time, "perf_counter", clock)
    sample = robolab_benchmark._run_sample(10, 2, 3, solve_layout)

    assert sample["target_reached"]
    assert sample["attempted_layouts"] == 3
    assert sample["accepted_layouts"] == 3
    assert sample["unique_layouts"] == 2
    assert sample["elapsed_ms"] == 3.0
    assert observed_seeds == [10, 11, 12]


def test_direct_solver_progressive_suite_starts_simple_and_adds_difficulty():
    scenarios = direct_benchmark.progressive_scenarios()

    assert scenarios[0] == direct_benchmark.Scenario("simple", "one-object", 1, 0.12, "random")
    assert {scenario.stage for scenario in scenarios} == {
        "simple",
        "object-scaling",
        "density-scaling",
        "initialization",
    }
    assert any(scenario.num_objects == 10 and scenario.object_size_m == 0.24 for scenario in scenarios)
    assert any(scenario.init_mode == "overlap" for scenario in scenarios)


def test_direct_solver_initial_positions_are_deterministic_and_mode_specific():
    random_scenario = direct_benchmark.Scenario("test", "random", 2, 0.12, "random")
    clustered_scenario = direct_benchmark.Scenario("test", "clustered", 2, 0.12, "clustered")
    overlap_scenario = direct_benchmark.Scenario("test", "overlap", 2, 0.12, "overlap")

    assert direct_benchmark.initial_xy(random_scenario, 2, 7, BOUNDS) == direct_benchmark.initial_xy(
        random_scenario, 2, 7, BOUNDS
    )
    clustered = direct_benchmark.initial_xy(clustered_scenario, 2, 7, BOUNDS)
    assert all(abs(coordinate) <= 0.1 for layout in clustered for xy in layout.values() for coordinate in xy)
    overlap = direct_benchmark.initial_xy(overlap_scenario, 2, 7, BOUNDS)
    assert all(xy == (0.0, 0.0) for layout in overlap for xy in layout.values())


def test_direct_solver_relational_suite_uses_supported_directional_graphs():
    scenarios = direct_benchmark.relational_scenarios()

    assert [scenario.name for scenario in scenarios] == [
        "directional-pair",
        "directional-chain-5",
        "directional-star-5",
        "directional-dual-chain-10",
    ]
    for scenario in scenarios:
        object_names = {f"object-{index}" for index in range(scenario.num_objects)}
        assert scenario.relations
        assert all(
            relation.child in object_names and relation.parent in object_names for relation in scenario.relations
        )


def test_direct_solver_shared_relation_validator_uses_edge_gap():
    relation = direct_benchmark.RelationEdge("object-1", "object-0", "positive_x")
    scenario = direct_benchmark.Scenario("relations", "pair", 2, 0.08, "random", (relation,))
    valid_poses = {"object-0": (0.0, 0.0), "object-1": (0.11, 0.04)}
    invalid_poses = {"object-0": (0.0, 0.0), "object-1": (0.08, 0.04)}

    assert direct_benchmark._relations_valid(valid_poses, scenario)
    assert not direct_benchmark._relations_valid(invalid_poses, scenario)


def test_direct_solver_robolab_seed_is_reproducible():
    direct_benchmark._seed_robolab(7)
    first = random.random()
    direct_benchmark._seed_robolab(7)

    assert random.random() == first


def test_obstacle_distribution_validator_checks_fixed_and_movable_collisions():
    valid = {
        "movable-small": (0.42, 0.40),
        "movable-medium": (-0.42, 0.00),
        "movable-large": (0.42, 0.00),
    }
    obstacle_collision = {**valid, "movable-small": (-0.27, 0.25)}
    movable_collision = {**valid, "movable-medium": (0.42, 0.40)}

    assert obstacle_benchmark.validate_layout(valid)
    assert not obstacle_benchmark.validate_layout(obstacle_collision)
    assert not obstacle_benchmark.validate_layout(movable_collision)


def test_obstacle_feasible_mask_depends_on_object_footprint():
    small = obstacle_benchmark._feasible_mask("movable-small", 32)
    large = obstacle_benchmark._feasible_mask("movable-large", 32)

    assert 0 < large.sum() < small.sum() < small.size


def test_collision_space_physical_validator_distinguishes_box_and_disk_corners():
    box = coverage_benchmark.Scenario("box", 0.16)
    disk = coverage_benchmark.Scenario("disk", 0.16)
    corner_clearance_position = (0.06, 0.08)

    assert not coverage_benchmark._position_valid(corner_clearance_position, box)
    assert coverage_benchmark._position_valid(corner_clearance_position, disk)


def test_collision_space_sweep_balances_shapes_and_sizes():
    scenarios = coverage_benchmark.scenarios()
    expected = tuple(
        coverage_benchmark.Scenario(shape, size)
        for shape in coverage_benchmark.SHAPES
        for size in coverage_benchmark.OBSTACLE_SIZES_M
    )

    assert scenarios == expected


@pytest.mark.parametrize(
    ("second_xy", "expected"),
    [
        pytest.param((0.09, 0.19), True, id="overlapping"),
        pytest.param((0.11, 0.19), False, id="separated-x"),
        pytest.param((0.09, 0.21), False, id="separated-y"),
    ],
)
def test_representation_ablation_aabb_oracle_checks_both_axes(second_xy, expected):
    dims = (0.1, 0.2, 0.1)

    assert collision_representation_ablation._aabb_overlap((0.0, 0.0), second_xy, dims, dims, 0.0) is expected


def test_representation_ablation_uses_identical_seeded_initialization():
    seeded_position = collision_representation_ablation._initial_xy(123)

    assert seeded_position == collision_representation_ablation._initial_xy(123)
    assert seeded_position != collision_representation_ablation._initial_xy(124)


def test_fixed_iteration_batch_scaling_uses_deterministic_overlapping_cluster():
    positions = batch_scaling_benchmark._clustered_xy(5, variant=3)

    assert positions == batch_scaling_benchmark._clustered_xy(5, variant=3)
    assert positions != batch_scaling_benchmark._clustered_xy(5, variant=4)
    assert all(abs(coordinate) <= 0.012 for xy in positions.values() for coordinate in xy)


def test_robolab_fixed_iteration_loop_runs_every_configured_iteration():
    class FakeSolver:
        def __init__(self):
            self.collision_checks = 0
            self.bounds_checks = 0

        def _check_collisions(self, states, dimensions):
            self.collision_checks += 1
            return []

        def _check_table_bounds(self, states, dimensions):
            self.bounds_checks += 1

    solver = FakeSolver()
    batch_scaling_benchmark._run_robolab_fixed_iterations(
        solver,
        states={},
        dimensions={},
        iterations=600,
        seed=0,
    )

    assert solver.collision_checks == 600
    assert solver.bounds_checks == 0
