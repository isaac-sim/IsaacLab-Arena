# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure direct-solver feasibility across object and background complexity."""

from __future__ import annotations

import argparse
import json
import math
import numpy as np
import statistics
from dataclasses import asdict
from pathlib import Path

from isaaclab_arena.relations.benchmark.provenance import collect_source_revision
from isaaclab_arena_examples.relations.background_factorial_benchmark import (
    DEFAULT_EXCLUDED_FRACTIONS,
    FactorialScene,
    _parse_floats,
    _parse_topologies,
    _sample_arena_factorial_batch,
    _sample_robolab_controlled,
    _valid_factorial_layout,
    build_scenes,
)
from isaaclab_arena_examples.relations.background_practical_generation_benchmark import (
    _initial_layout,
    _make_counted_robolab_api,
    _runtime_metadata,
)
from isaaclab_arena_examples.relations.fixed_iteration_batch_scaling_benchmark import (
    OBJECT_SIZE_M,
    TABLE_BOUNDS,
    _parse_ints,
)

Layout = dict[str, tuple[float, float]]
REFERENCE_SEED_OFFSET = 1_000_000_000


def _max_pairwise_penetration(layout: Layout, num_objects: int, scene: FactorialScene) -> float:
    """Return the largest local table violation or pairwise AABB penetration."""
    if set(layout) != {f"object-{index}" for index in range(num_objects)}:
        return math.inf
    half_size = OBJECT_SIZE_M / 2
    maximum = 0.0
    for x, y in layout.values():
        maximum = max(
            maximum,
            TABLE_BOUNDS[0] + half_size - x,
            x - (TABLE_BOUNDS[1] - half_size),
            TABLE_BOUNDS[2] + half_size - y,
            y - (TABLE_BOUNDS[3] - half_size),
        )
        for obstacle in scene.obstacles:
            penetration_x = (OBJECT_SIZE_M + obstacle.width) / 2 - abs(x - obstacle.x)
            penetration_y = (OBJECT_SIZE_M + obstacle.depth) / 2 - abs(y - obstacle.y)
            if penetration_x > 0.0 and penetration_y > 0.0:
                maximum = max(maximum, min(penetration_x, penetration_y))
    positions = list(layout.values())
    for index, first in enumerate(positions):
        for second in positions[index + 1 :]:
            penetration_x = OBJECT_SIZE_M - abs(first[0] - second[0])
            penetration_y = OBJECT_SIZE_M - abs(first[1] - second[1])
            if penetration_x > 0.0 and penetration_y > 0.0:
                maximum = max(maximum, min(penetration_x, penetration_y))
    return maximum


def _mean_displacement(initial: Layout, final: Layout) -> float:
    if initial.keys() != final.keys():
        return math.inf
    return statistics.fmean(math.dist(initial[name], final[name]) for name in initial)


def _summarize_layouts(
    method: str,
    budget: int,
    layouts: list[Layout],
    initial_layouts: list[Layout],
    num_objects: int,
    scene: FactorialScene,
    native_success_rate: float | None,
) -> dict:
    violations = [_max_pairwise_penetration(layout, num_objects, scene) for layout in layouts]
    displacements = [
        _mean_displacement(initial, final) for initial, final in zip(initial_layouts, layouts, strict=True)
    ]
    return {
        "method": method,
        "budget": budget,
        "valid_rate": sum(_valid_factorial_layout(layout, num_objects, scene) for layout in layouts) / len(layouts),
        "mean_max_pairwise_penetration_m": statistics.fmean(violations),
        "p95_max_pairwise_penetration_m": float(np.percentile(violations, 95)),
        "mean_object_displacement_m": statistics.fmean(displacements),
        "native_success_rate": native_success_rate,
    }


def _rejection_successes(
    trial_seeds: range,
    proposal_budget: int,
    proposal_stride: int,
    num_objects: int,
    scene: FactorialScene,
) -> tuple[float, float]:
    """Measure complete-layout rejection success within a proposal budget."""
    successes = attempts = 0
    for trial_seed in trial_seeds:
        for proposal_index in range(proposal_budget):
            attempts += 1
            proposal_seed = REFERENCE_SEED_OFFSET + trial_seed * proposal_stride + proposal_index
            if _valid_factorial_layout(_initial_layout(proposal_seed, num_objects), num_objects, scene):
                successes += 1
                break
    return successes / len(trial_seeds), attempts / len(trial_seeds)


def _condition_rows(
    scene: FactorialScene,
    num_objects: int,
    seeds: range,
    budgets: tuple[int, ...],
    robolab_api,
) -> list[dict]:
    initial_layouts = [_initial_layout(seed, num_objects) for seed in seeds]
    condition = {
        "num_objects": num_objects,
        "scene": scene.name,
        "scene_index": scene.scene_index,
        "topology": scene.topology,
        "target_excluded_fraction": scene.target_excluded_fraction,
        "measured_excluded_fraction": scene.measured_excluded_fraction,
        "background_objects": scene.obstacle_count,
    }
    rows = [
        condition
        | _summarize_layouts(
            "initial",
            0,
            initial_layouts,
            initial_layouts,
            num_objects,
            scene,
            native_success_rate=None,
        )
    ]
    for budget in budgets:
        arena_layouts, arena_steps, arena_successes, _memory = _sample_arena_factorial_batch(
            scene, num_objects, seeds, budget
        )
        arena_summary = _summarize_layouts(
            "arena",
            budget,
            arena_layouts,
            initial_layouts,
            num_objects,
            scene,
            native_success_rate=arena_successes / len(seeds),
        )
        arena_summary["iterations_run"] = arena_steps
        rows.append(condition | arena_summary)
        robolab_results = [_sample_robolab_controlled(scene, num_objects, seed, budget, robolab_api) for seed in seeds]
        robolab_layouts = [result[0] for result in robolab_results]
        robolab_summary = _summarize_layouts(
            "robolab",
            budget,
            robolab_layouts,
            initial_layouts,
            num_objects,
            scene,
            native_success_rate=sum(result[2] for result in robolab_results) / len(seeds),
        )
        robolab_summary["mean_collision_checks"] = statistics.fmean(result[1] for result in robolab_results)
        rows.append(condition | robolab_summary)
        rejection_rate, mean_attempts = _rejection_successes(
            seeds,
            budget,
            max(budgets),
            num_objects,
            scene,
        )
        rows.append(
            condition
            | {
                "method": "random_rejection",
                "budget": budget,
                "valid_rate": rejection_rate,
                "mean_proposals_per_trial": mean_attempts,
                "mean_max_pairwise_penetration_m": None,
                "p95_max_pairwise_penetration_m": None,
                "mean_object_displacement_m": None,
                "native_success_rate": None,
            }
        )
    return rows


def _deduplicated_scenes(
    excluded_fractions: tuple[float, ...],
    topologies: tuple[str, ...],
    scene_instances: int,
) -> tuple[FactorialScene, ...]:
    return build_scenes(excluded_fractions, topologies, scene_instances)


def _write_plot(
    rows: list[dict], object_counts: tuple[int, ...], excluded_fractions: tuple[float, ...], output: Path
) -> None:
    import matplotlib.pyplot as plt

    maximum_budget = max(row["budget"] for row in rows)
    methods = ("initial", "random_rejection", "robolab", "arena")
    figure, axes = plt.subplots(1, len(methods), figsize=(13, 3.2), sharex=True, sharey=True)
    for axis, method in zip(axes, methods, strict=True):
        values = np.full((len(excluded_fractions), len(object_counts)), np.nan)
        for row_index, excluded_fraction in enumerate(excluded_fractions):
            for column_index, num_objects in enumerate(object_counts):
                selected = [
                    row["valid_rate"]
                    for row in rows
                    if row["method"] == method
                    and row["topology"] == "scattered"
                    and row["num_objects"] == num_objects
                    and math.isclose(row["target_excluded_fraction"], excluded_fraction)
                    and (row["budget"] == maximum_budget or method == "initial")
                ]
                if selected:
                    values[row_index, column_index] = statistics.median(selected)
        image = axis.imshow(values, vmin=0.0, vmax=1.0, origin="lower", cmap="viridis", aspect="auto")
        title = method.replace("_", " ").title()
        if method == "random_rejection":
            title = f"Rejection (≤{maximum_budget} proposals)"
        elif method != "initial":
            title = f"{title} (≤{maximum_budget} iterations)"
        axis.set_title(title)
        axis.set_xticks(range(len(object_counts)), object_counts)
        for row_index in range(len(excluded_fractions)):
            for column_index in range(len(object_counts)):
                axis.text(column_index, row_index, f"{values[row_index, column_index]:.2f}", ha="center", va="center")
    axes[0].set_yticks(
        range(len(excluded_fractions)),
        [f"{100 * value:.0f}%" for value in excluded_fractions],
    )
    axes[0].set_ylabel("Excluded center space")
    for axis in axes:
        axis.set_xlabel("Movable objects")
    figure.colorbar(image, ax=axes, label="Success rate", fraction=0.025, pad=0.03)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robolab-root", type=Path, required=True)
    parser.add_argument("--object-counts", type=_parse_ints, default=(5, 10, 20))
    parser.add_argument("--excluded-fractions", type=_parse_floats, default=DEFAULT_EXCLUDED_FRACTIONS)
    parser.add_argument("--topologies", type=_parse_topologies, default=("scattered",))
    parser.add_argument("--scene-instances", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--budgets", type=_parse_ints, default=(10, 50, 100, 300, 600))
    parser.add_argument("--seed", type=int, default=40_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    assert min(*args.object_counts, args.scene_instances, args.batch_size, *args.budgets) > 0
    assert tuple(sorted(set(args.budgets))) == args.budgets
    assert "scattered" in args.topologies, "the primary frontier requires the scattered topology"
    assert all(
        any(math.isclose(value, supported, rel_tol=0.0, abs_tol=1e-9) for supported in DEFAULT_EXCLUDED_FRACTIONS)
        for value in args.excluded_fractions
    ), f"excluded fractions must use supported levels {DEFAULT_EXCLUDED_FRACTIONS}"

    scenes = _deduplicated_scenes(args.excluded_fractions, args.topologies, args.scene_instances)
    robolab_api = _make_counted_robolab_api(args.robolab_root, mode="matched-aabb")
    rows = []
    for num_objects in args.object_counts:
        for scene in scenes:
            condition_rows = _condition_rows(
                scene,
                num_objects,
                range(args.seed, args.seed + args.batch_size),
                args.budgets,
                robolab_api,
            )
            rows.extend(condition_rows)
            final = [row for row in condition_rows if row["budget"] == args.budgets[-1]]
            print(
                f"n={num_objects:2} {scene.name:34} "
                + " ".join(f"{row['method']}={row['valid_rate']:.3f}" for row in final)
            )

    benchmark_root = Path(__file__).resolve().parents[2]
    payload = {
        "schema_version": 1,
        "experiment": "direct-solver-feasibility-frontier",
        "benchmark_revision": collect_source_revision(benchmark_root),
        "robolab_revision": collect_source_revision(args.robolab_root),
        "runtime": _runtime_metadata(),
        "collision_representation": "matched-aabb",
        "movable_size_m": OBJECT_SIZE_M,
        "table_bounds": TABLE_BOUNDS,
        "object_counts": args.object_counts,
        "excluded_fractions": args.excluded_fractions,
        "topologies": args.topologies,
        "scene_instances": args.scene_instances,
        "batch_size": args.batch_size,
        "budgets": args.budgets,
        "seed": args.seed,
        "reference_seed_offset": REFERENCE_SEED_OFFSET,
        "reference_proposal_stride": max(args.budgets),
        "budget_semantics": {
            "arena": "maximum optimizer iterations for one candidate",
            "robolab": "maximum procedural iterations for one candidate",
            "random_rejection": "maximum independent complete-layout proposals",
        },
        "primary_topology": "scattered",
        "corridor_scope": "diagnostic for collision-objective failure modes",
        "scenes": [asdict(scene) | {"name": scene.name} for scene in scenes],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.plot is not None:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        _write_plot(rows, args.object_counts, args.excluded_fractions, args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
