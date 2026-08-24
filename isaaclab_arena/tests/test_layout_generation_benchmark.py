# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import pytest

from isaaclab_arena.relations.benchmark.layout_generation import (
    LayoutGenerationRun,
    LayoutPose,
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
        seeds=(8, 9),
        samples=tuple(replace(sample, seed=sample.seed + 3) for sample in baseline.samples),
    )
    with pytest.raises(ValueError, match="mismatched seeds"):
        check_throughput_compatibility([baseline, mismatch])


def test_authored_layout_validity_depends_on_requested_density():
    assert explicit_layout(table_xy_bounds=BOUNDS, object_xy_sizes=make_object_sizes(4, 0.2)).success
    assert not explicit_layout(table_xy_bounds=BOUNDS, object_xy_sizes=make_object_sizes(4, 0.8)).success
