# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test solver reproducibility under repeated, reordered, and partitioned execution."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Literal

from isaaclab_arena.relations.benchmark.provenance import collect_source_revision
from isaaclab_arena_examples.relations.background_batch_scaling_benchmark import SCENARIOS, _source_revision
from isaaclab_arena_examples.relations.background_practical_generation_benchmark import (
    Layout,
    _initial_layout,
    _make_counted_robolab_api,
    _sample_arena_batch,
    _sample_robolab,
    _valid_layout,
)

Method = Literal["arena", "robolab"]
TOY_SCENARIO = SCENARIOS[0]


def _sample(
    method: Method,
    seeds: list[int],
    num_objects: int,
    max_iterations: int,
    robolab_api,
) -> dict[int, Layout]:
    if method == "arena":
        layouts, _iterations, _native_successes, _peak_memory_mb = _sample_arena_batch(
            TOY_SCENARIO,
            num_objects,
            seeds,
            max_iterations,
        )
    else:
        layouts = [
            _sample_robolab(
                TOY_SCENARIO,
                num_objects,
                seed,
                max_iterations,
                allow_relaxation=True,
                robolab_api=robolab_api,
            )[0]
            for seed in seeds
        ]
    return dict(zip(seeds, layouts, strict=True))


def _partitioned_sample(
    method: Method,
    seeds: list[int],
    partition_size: int,
    num_objects: int,
    max_iterations: int,
    robolab_api,
) -> dict[int, Layout]:
    layouts = {}
    for start in range(0, len(seeds), partition_size):
        layouts.update(
            _sample(
                method,
                seeds[start : start + partition_size],
                num_objects,
                max_iterations,
                robolab_api,
            )
        )
    return layouts


def _layout_vector(layout: Layout, num_objects: int) -> tuple[float, ...]:
    return tuple(coordinate for index in range(num_objects) for coordinate in layout[f"object-{index}"])


def _select_overlapping_seeds(start_seed: int, count: int, num_objects: int) -> list[int]:
    """Select deterministic seeds whose initial layouts require collision resolution."""
    seeds = []
    for seed in range(start_seed, start_seed + 1_000_000):
        initial_layout = _initial_layout(seed, num_objects)
        if not _valid_layout(initial_layout, num_objects, TOY_SCENARIO):
            seeds.append(seed)
            if len(seeds) == count:
                return seeds
    raise RuntimeError(f"Could not find {count} overlapping layouts in one million seeds")


def _compare(reference: dict[int, Layout], candidate: dict[int, Layout], num_objects: int) -> dict:
    assert reference.keys() == candidate.keys()
    exact_matches = tolerance_matches = 0
    maximum_error = 0.0
    for seed in reference:
        expected = _layout_vector(reference[seed], num_objects)
        observed = _layout_vector(candidate[seed], num_objects)
        errors = [abs(left - right) for left, right in zip(expected, observed, strict=True)]
        error = max(errors, default=0.0)
        exact_matches += int(expected == observed)
        tolerance_matches += int(error <= 1e-7)
        maximum_error = max(maximum_error, error)
    count = len(reference)
    return {
        "layouts": count,
        "exact_match_rate": exact_matches / count,
        "match_rate_at_1e-7_m": tolerance_matches / count,
        "max_abs_position_error_m": maximum_error,
        "reference_shared_valid_rate": (
            sum(_valid_layout(layout, num_objects, TOY_SCENARIO) for layout in reference.values()) / count
        ),
        "candidate_shared_valid_rate": (
            sum(_valid_layout(layout, num_objects, TOY_SCENARIO) for layout in candidate.values()) / count
        ),
    }


def _summarize_method(
    method: Method,
    seeds: list[int],
    num_objects: int,
    max_iterations: int,
    repeats: int,
    partition_sizes: tuple[int, ...],
    robolab_api,
) -> dict:
    initial = {seed: _initial_layout(seed, num_objects) for seed in seeds}
    baseline = _sample(method, seeds, num_objects, max_iterations, robolab_api)
    repeated = [
        _compare(
            baseline,
            _sample(method, seeds, num_objects, max_iterations, robolab_api),
            num_objects,
        )
        for _ in range(repeats)
    ]
    reorderings = {"reversed": list(reversed(seeds))}
    for index in range(3):
        shuffled = seeds.copy()
        random.Random(91_000 + index).shuffle(shuffled)
        reorderings[f"shuffle-{index}"] = shuffled
    reordered = {
        name: _compare(
            baseline,
            _sample(method, order, num_objects, max_iterations, robolab_api),
            num_objects,
        )
        for name, order in reorderings.items()
    }
    partitioned = (
        {
            str(partition_size): _compare(
                baseline,
                _partitioned_sample(
                    method,
                    seeds,
                    partition_size,
                    num_objects,
                    max_iterations,
                    robolab_api,
                ),
                num_objects,
            )
            for partition_size in partition_sizes
        }
        if method == "arena"
        else {"not_applicable": "RoboLab is serial; grouping identical per-seed calls does not change its execution."}
    )
    vectors = {_layout_vector(layout, num_objects) for layout in baseline.values()}
    return {
        "method": method,
        "initial_to_baseline": _compare(initial, baseline, num_objects),
        "same_process_repeats": repeated,
        "reorderings": reordered,
        "partition_sizes": partitioned,
        "different_seed_unique_rate": len(vectors) / len(seeds),
        "shared_valid_rate": sum(
            _valid_layout(layout, num_objects, TOY_SCENARIO) for layout in baseline.values()
        ) / len(seeds),
        "baseline_layouts": {str(seed): layout for seed, layout in baseline.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robolab-root", type=Path, required=True)
    parser.add_argument("--num-objects", type=int, default=5)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--partition-sizes", type=lambda value: tuple(map(int, value.split(","))), default=(1, 2, 4))
    parser.add_argument("--max-iterations", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.num_objects >= 2
    assert min(args.sample_count, args.repeats, args.max_iterations) > 0
    assert args.partition_sizes and all(0 < size <= args.sample_count for size in args.partition_sizes)

    seeds = _select_overlapping_seeds(args.seed, args.sample_count, args.num_objects)
    robolab_api = _make_counted_robolab_api(args.robolab_root, mode="native")
    benchmark_root = Path(__file__).resolve().parents[2]
    payload = {
        "schema_version": 1,
        "experiment": "same-seed reproducibility and batch-partition invariance",
        "benchmark_revision": collect_source_revision(benchmark_root),
        "robolab_revision": _source_revision(args.robolab_root),
        "num_objects": args.num_objects,
        "sample_count": args.sample_count,
        "seeds": seeds,
        "seed_selection": "first seeds at or above --seed with shared-invalid initial layouts",
        "initial_shared_valid_rate": 0.0,
        "repeats": args.repeats,
        "partition_sizes": args.partition_sizes,
        "max_iterations": args.max_iterations,
        "tolerance_m": 1e-7,
        "scope": "same process and hardware",
        "results": [
            _summarize_method(
                method,
                seeds,
                args.num_objects,
                args.max_iterations,
                args.repeats,
                args.partition_sizes,
                robolab_api,
            )
            for method in ("arena", "robolab")
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for result in payload["results"]:
        worst_repeat = max(item["max_abs_position_error_m"] for item in result["same_process_repeats"])
        worst_reordering = max(item["max_abs_position_error_m"] for item in result["reorderings"].values())
        summary = (
            f"{result['method']:7} valid={result['shared_valid_rate']:.3f} "
            f"repeat-error={worst_repeat:.3e} reorder-error={worst_reordering:.3e}"
        )
        if result["method"] == "arena":
            worst_partition = max(item["max_abs_position_error_m"] for item in result["partition_sizes"].values())
            summary += f" partition-error={worst_partition:.3e}"
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
