# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare native exact-K valid-layout generation with passive backgrounds."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import numpy as np
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Literal

from isaaclab_arena.relations.benchmark.provenance import collect_source_revision
from isaaclab_arena_examples.relations.background_batch_scaling_benchmark import (
    BACKGROUND_CENTERS,
    BACKGROUND_SIZE_M,
    SCENARIOS,
    BackgroundScenario,
    _backgrounds,
    _load_robolab,
)
from isaaclab_arena_examples.relations.background_batch_scaling_benchmark import (
    _runtime_metadata as _scaling_runtime_metadata,
)
from isaaclab_arena_examples.relations.background_batch_scaling_benchmark import _source_revision
from isaaclab_arena_examples.relations.fixed_iteration_batch_scaling_benchmark import (
    OBJECT_HEIGHT_M,
    OBJECT_SIZE_M,
    TABLE_BOUNDS,
    _build_arena_workload,
    _parse_ints,
)

Algorithm = Literal["arena", "robolab"]
Layout = dict[str, tuple[float, float]]


def _initial_layout(seed: int, num_objects: int) -> Layout:
    """Sample one deterministic layout inside object-center table bounds."""
    generator = random.Random(seed)
    margin = OBJECT_SIZE_M / 2
    return {
        f"object-{index}": (
            generator.uniform(TABLE_BOUNDS[0] + margin, TABLE_BOUNDS[1] - margin),
            generator.uniform(TABLE_BOUNDS[2] + margin, TABLE_BOUNDS[3] - margin),
        )
        for index in range(num_objects)
    }


def _overlap(
    first_xy: tuple[float, float],
    first_size: float,
    second_xy: tuple[float, float],
    second_size: float,
) -> bool:
    return (
        abs(first_xy[0] - second_xy[0]) < (first_size + second_size) / 2
        and abs(first_xy[1] - second_xy[1]) < (first_size + second_size) / 2
    )


def _valid_layout(layout: Layout, num_objects: int, scenario: BackgroundScenario) -> bool:
    """Check exact AABB containment and movable/background non-overlap."""
    expected_names = {f"object-{index}" for index in range(num_objects)}
    if set(layout) != expected_names:
        return False
    margin = OBJECT_SIZE_M / 2
    for xy in layout.values():
        if not (
            TABLE_BOUNDS[0] + margin <= xy[0] <= TABLE_BOUNDS[1] - margin
            and TABLE_BOUNDS[2] + margin <= xy[1] <= TABLE_BOUNDS[3] - margin
        ):
            return False
        if any(_overlap(xy, OBJECT_SIZE_M, center, BACKGROUND_SIZE_M) for center in scenario.centers):
            return False
    items = list(layout.items())
    return not any(
        _overlap(first_xy, OBJECT_SIZE_M, second_xy, OBJECT_SIZE_M)
        for index, (_, first_xy) in enumerate(items)
        for _, second_xy in items[index + 1 :]
    )


def _layout_key(layout: Layout, resolution_m: float = 1e-6) -> tuple[tuple[str, int, int], ...]:
    return tuple((name, round(x / resolution_m), round(y / resolution_m)) for name, (x, y) in sorted(layout.items()))


def _summarize_diversity(layouts: list[Layout], num_objects: int) -> dict[str, float]:
    if not layouts:
        return {
            "unique_1cm_rate": 0.0,
            "mean_marginal_grid_coverage": 0.0,
            "mean_marginal_normalized_entropy": 0.0,
            "mean_intra_layout_pair_distance_m": 0.0,
        }
    unique_1cm = len({_layout_key(layout, 0.01) for layout in layouts}) / len(layouts)
    bins = 20
    coverages = []
    entropies = []
    for object_index in range(num_objects):
        name = f"object-{object_index}"
        counts = np.zeros((bins, bins), dtype=np.int64)
        for layout in layouts:
            x, y = layout[name]
            x_bin = min(bins - 1, int((x - TABLE_BOUNDS[0]) / (TABLE_BOUNDS[1] - TABLE_BOUNDS[0]) * bins))
            y_bin = min(bins - 1, int((y - TABLE_BOUNDS[2]) / (TABLE_BOUNDS[3] - TABLE_BOUNDS[2]) * bins))
            counts[x_bin, y_bin] += 1
        occupied = counts[counts > 0]
        coverages.append(len(occupied) / counts.size)
        probabilities = occupied / occupied.sum()
        entropies.append(float(-(probabilities * np.log(probabilities)).sum() / math.log(counts.size)))
    pair_distances = []
    for layout in layouts:
        positions = list(layout.values())
        pair_distances.extend(
            math.dist(first, second) for index, first in enumerate(positions) for second in positions[index + 1 :]
        )
    return {
        "unique_1cm_rate": unique_1cm,
        "mean_marginal_grid_coverage": statistics.mean(coverages),
        "mean_marginal_normalized_entropy": statistics.mean(entropies),
        "mean_intra_layout_pair_distance_m": statistics.mean(pair_distances) if pair_distances else 0.0,
    }


def _sample_arena_batch(
    scenario: BackgroundScenario,
    num_objects: int,
    seeds: range,
    max_iterations: int,
) -> tuple[list[Layout], int, int, float | None]:
    import torch

    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

    objects, positions = _build_arena_workload(num_objects, len(seeds))
    table = objects[0]
    for env_positions, seed in zip(positions, seeds, strict=True):
        initial_layout = _initial_layout(seed, num_objects)
        for obj in objects:
            if obj is not table:
                env_positions[obj] = (*initial_layout[obj.name], OBJECT_HEIGHT_M / 2)
    solver = RelationSolver(
        RelationSolverParams(
            max_iters=max_iterations,
            clearance_m=0.0,
            verbose=False,
            save_position_history=False,
        )
    )
    solved_positions = solver.solve(objects, positions, collision_objects=_backgrounds(scenario))
    movable_objects = objects[1:]
    layouts = [
        {obj.name: (float(env_positions[obj][0]), float(env_positions[obj][1])) for obj in movable_objects}
        for env_positions in solved_positions
    ]
    assert solver.last_loss_per_env is not None
    native_successes = int((solver.last_loss_per_env < solver.params.convergence_threshold).sum().item())
    peak_memory_mb = float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else None
    return layouts, len(solver.last_loss_history), native_successes, peak_memory_mb


def _make_counted_robolab_api(robolab_root: Path, mode: str):
    if mode == "matched-aabb":
        ObjectState, BaseSolver = _load_robolab(robolab_root)
    else:
        sys.path.insert(0, str(robolab_root.resolve()))
        from robolab.scene_gen.llm_scene_gen.predicates import ObjectState
        from robolab.scene_gen.llm_scene_gen.spatial_solver import SpatialSolver as BaseSolver

    class CountedSolver(BaseSolver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.collision_checks = 0

        def _check_collisions(self, object_states, object_dims):
            self.collision_checks += 1
            return super()._check_collisions(object_states, object_dims)

    return ObjectState, CountedSolver


def _sample_robolab(
    scenario: BackgroundScenario,
    num_objects: int,
    seed: int,
    max_iterations: int,
    allow_relaxation: bool,
    robolab_api,
) -> tuple[Layout, int, bool]:
    ObjectState, SpatialSolver = robolab_api
    random.seed(seed)
    np.random.seed(seed % (2**32))
    initial_layout = _initial_layout(seed, num_objects)
    states = {
        name: ObjectState(name=name, x=xy[0], y=xy[1], yaw=0.0, is_placed=True) for name, xy in initial_layout.items()
    }
    background_names = []
    for index, center in enumerate(scenario.centers):
        name = f"background-{index}"
        states[name] = ObjectState(name=name, x=center[0], y=center[1], yaw=0.0, is_placed=True)
        background_names.append(name)
    dimensions = {f"object-{index}": (OBJECT_SIZE_M, OBJECT_SIZE_M, OBJECT_HEIGHT_M) for index in range(num_objects)}
    dimensions.update({
        f"background-{index}": (BACKGROUND_SIZE_M, BACKGROUND_SIZE_M, OBJECT_HEIGHT_M)
        for index in range(scenario.obstacle_count)
    })
    solver = SpatialSolver(table_bounds=TABLE_BOUNDS, collision_margin=0.0)
    solver._background_names = set(background_names)
    with contextlib.redirect_stdout(io.StringIO()):
        native_success, _ = solver.solve(
            states,
            dimensions,
            max_iterations=max_iterations,
            fixed_objects=background_names,
            allow_relaxation=allow_relaxation,
        )
    for index, center in enumerate(scenario.centers):
        state = states[f"background-{index}"]
        assert (state.x, state.y) == center, "RoboLab moved a fixed background object"
    layout = {
        name: (float(states[name].x), float(states[name].y))
        for name in initial_layout
        if states[name].x is not None and states[name].y is not None
    }
    return layout, solver.collision_checks, native_success


def _generate_repetition(
    algorithm: Algorithm,
    scenario: BackgroundScenario,
    num_objects: int,
    target_layouts: int,
    max_attempts_per_layout: int,
    max_iterations: int,
    seed: int,
    allow_robolab_relaxation: bool,
    robolab_api=None,
) -> dict:
    import torch

    unique: dict[tuple[tuple[str, int, int], ...], Layout] = {}
    attempted = accepted = native_successes = solver_calls = work_unit_layouts = 0
    maximum_attempts = target_layouts * max_attempts_per_layout
    peak_memory_mb = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    while attempted < maximum_attempts and len(unique) < target_layouts:
        if algorithm == "arena":
            batch_size = min(target_layouts - len(unique), maximum_attempts - attempted)
            seeds = range(seed + attempted, seed + attempted + batch_size)
            candidates, iterations, batch_native_successes, peak_memory_mb = _sample_arena_batch(
                scenario, num_objects, seeds, max_iterations
            )
            attempted += batch_size
            native_successes += batch_native_successes
            solver_calls += 1
            work_unit_layouts += iterations * batch_size
        else:
            layout, iterations, native_success = _sample_robolab(
                scenario,
                num_objects,
                seed + attempted,
                max_iterations,
                allow_robolab_relaxation,
                robolab_api,
            )
            attempted += 1
            candidates = [layout]
            native_successes += int(native_success)
            solver_calls += 1
            work_unit_layouts += iterations
        for layout in candidates:
            if _valid_layout(layout, num_objects, scenario):
                accepted += 1
                unique.setdefault(_layout_key(layout), layout)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1e3
    layouts = list(unique.values())
    measurement = {
        "seed": seed,
        "scenario": scenario.name,
        "background_objects": scenario.obstacle_count,
        "num_objects": num_objects,
        "target_layouts": target_layouts,
        "target_reached": len(layouts) == target_layouts,
        "elapsed_ms": elapsed_ms,
        "attempted_layouts": attempted,
        "native_successes": native_successes,
        "shared_valid_layouts": accepted,
        "unique_layouts": len(layouts),
        "solver_calls": solver_calls,
        "native_success_rate": native_successes / attempted,
        "shared_valid_rate": accepted / attempted,
        "unique_valid_layouts_per_second": len(layouts) * 1e3 / elapsed_ms,
        "peak_gpu_memory_mb": peak_memory_mb if algorithm == "arena" else None,
        "diversity": _summarize_diversity(layouts, num_objects),
    }
    work_unit_name = (
        "arena_optimizer_steps_per_attempt" if algorithm == "arena" else "robolab_collision_checks_per_attempt"
    )
    measurement[work_unit_name] = work_unit_layouts / attempted
    return measurement


def _runtime_metadata() -> dict:
    import torch

    return _scaling_runtime_metadata(torch)


def generate(args: argparse.Namespace) -> int:
    benchmark_root = Path(__file__).resolve().parents[2]
    scenarios = tuple(scenario for scenario in SCENARIOS if scenario.name in args.scenario_names)
    allow_robolab_relaxation = (
        args.allow_robolab_relaxation if args.allow_robolab_relaxation is not None else args.robolab_mode == "native"
    )
    robolab_api = (
        _make_counted_robolab_api(args.robolab_root, args.robolab_mode) if args.algorithm == "robolab" else None
    )
    maximum_target = max(args.target_layout_counts)
    seed_stride = maximum_target * args.max_attempts_per_layout
    for scenario in scenarios:
        _generate_repetition(
            args.algorithm,
            scenario,
            args.num_objects,
            min(16, maximum_target),
            args.max_attempts_per_layout,
            args.max_iterations,
            args.seed - seed_stride,
            allow_robolab_relaxation,
            robolab_api,
        )
    measurements = []
    for scenario in scenarios:
        for target_layouts in args.target_layout_counts:
            for repeat in range(args.repetitions):
                measurement = _generate_repetition(
                    args.algorithm,
                    scenario,
                    args.num_objects,
                    target_layouts,
                    args.max_attempts_per_layout,
                    args.max_iterations,
                    args.seed + repeat * seed_stride,
                    allow_robolab_relaxation,
                    robolab_api,
                )
                measurements.append(measurement)
                print(
                    f"{args.algorithm:7} {scenario.name:18} K={target_layouts:4} "
                    f"valid/s={measurement['unique_valid_layouts_per_second']:8.2f} "
                    f"accepted={measurement['shared_valid_rate']:.3f}"
                )
    source_root = args.robolab_root if args.algorithm == "robolab" else benchmark_root
    payload = {
        "schema_version": 2,
        "algorithm": args.algorithm,
        "source_revision": _source_revision(source_root),
        "benchmark_revision": collect_source_revision(benchmark_root),
        "runtime": _runtime_metadata(),
        "scenarios": [{"name": scenario.name, "obstacle_count": scenario.obstacle_count} for scenario in scenarios],
        "num_objects": args.num_objects,
        "target_layout_counts": args.target_layout_counts,
        "repetitions": args.repetitions,
        "max_attempts_per_layout": args.max_attempts_per_layout,
        "max_iterations": args.max_iterations,
        "seed": args.seed,
        "seed_stride": seed_stride,
        "table_bounds": TABLE_BOUNDS,
        "movable_size_m": OBJECT_SIZE_M,
        "background_size_m": BACKGROUND_SIZE_M,
        "background_centers": BACKGROUND_CENTERS,
        "comparison_mode": args.robolab_mode,
        "collision_representation": "matched-aabb" if args.robolab_mode == "matched-aabb" else "method-native",
        "arena_collision_representation": "aabb",
        "robolab_collision_representation": (
            "aabb-adapter" if args.robolab_mode == "matched-aabb" else "native-circumscribed-circle"
        ),
        "constraints": "table-containment-and-pairwise-non-overlap",
        "initialization": "shared-seeded-uniform-center-positions",
        "early_stopping": True,
        "allow_robolab_relaxation": allow_robolab_relaxation,
        "robolab_adaptive_iteration_budget": True,
        "timing_scope": "complete-exact-k-generation",
        "measurements": measurements,
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


def _median_iqr(values: list[float]) -> tuple[float, float]:
    return statistics.median(values), float(np.percentile(values, 75) - np.percentile(values, 25))


def analyze(args: argparse.Namespace) -> int:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    keyed = {run["algorithm"]: run for run in runs}
    if set(keyed) != {"arena", "robolab"}:
        raise ValueError("analysis requires one Arena and one RoboLab run")
    compatibility = (
        "schema_version",
        "benchmark_revision",
        "runtime",
        "scenarios",
        "num_objects",
        "target_layout_counts",
        "repetitions",
        "max_attempts_per_layout",
        "max_iterations",
        "seed",
        "seed_stride",
        "table_bounds",
        "movable_size_m",
        "background_size_m",
        "background_centers",
        "comparison_mode",
        "collision_representation",
        "arena_collision_representation",
        "robolab_collision_representation",
        "constraints",
        "initialization",
        "early_stopping",
        "allow_robolab_relaxation",
        "robolab_adaptive_iteration_budget",
        "timing_scope",
    )
    for field in compatibility:
        if keyed["arena"][field] != keyed["robolab"][field]:
            raise ValueError(f"incompatible {field}")
    rows = []
    scenarios = tuple(BackgroundScenario(**scenario) for scenario in keyed["arena"]["scenarios"])
    for scenario in scenarios:
        for target_layouts in keyed["arena"]["target_layout_counts"]:
            method_measurements = {
                algorithm: [
                    measurement
                    for measurement in run["measurements"]
                    if measurement["scenario"] == scenario.name and measurement["target_layouts"] == target_layouts
                ]
                for algorithm, run in keyed.items()
            }
            if any(len(values) != keyed["arena"]["repetitions"] for values in method_measurements.values()):
                raise ValueError(f"incomplete repetitions for {scenario.name}, K={target_layouts}")
            row = {
                "scenario": scenario.name,
                "background_objects": scenario.obstacle_count,
                "num_objects": keyed["arena"]["num_objects"],
                "target_layouts": target_layouts,
            }
            for algorithm, measurements in method_measurements.items():
                for metric in (
                    "target_reached",
                    "unique_valid_layouts_per_second",
                    "native_success_rate",
                    "shared_valid_rate",
                    "peak_gpu_memory_mb",
                ):
                    values = [
                        float(measurement[metric]) for measurement in measurements if measurement[metric] is not None
                    ]
                    if values:
                        median, iqr = _median_iqr(values)
                        row[f"{algorithm}_{metric}"] = median
                        row[f"{algorithm}_{metric}_iqr"] = iqr
                for metric in (
                    "unique_1cm_rate",
                    "mean_marginal_grid_coverage",
                    "mean_marginal_normalized_entropy",
                    "mean_intra_layout_pair_distance_m",
                ):
                    values = [measurement["diversity"][metric] for measurement in measurements]
                    median, iqr = _median_iqr(values)
                    row[f"{algorithm}_{metric}"] = median
                    row[f"{algorithm}_{metric}_iqr"] = iqr
                work_metric = (
                    "arena_optimizer_steps_per_attempt"
                    if algorithm == "arena"
                    else "robolab_collision_checks_per_attempt"
                )
                work_values = [measurement[work_metric] for measurement in measurements]
                work_median, work_iqr = _median_iqr(work_values)
                row[work_metric] = work_median
                row[f"{work_metric}_iqr"] = work_iqr
            arena_rate = row["arena_unique_valid_layouts_per_second"]
            robolab_rate = row["robolab_unique_valid_layouts_per_second"]
            paired_complete = all(
                measurement["target_reached"]
                for measurements in method_measurements.values()
                for measurement in measurements
            )
            row["paired_exact_k_complete"] = paired_complete
            row["arena_speedup"] = arena_rate / robolab_rate if paired_complete and robolab_rate > 0.0 else None
            rows.append(row)
    output = {
        "schema_version": 2,
        "comparison": "native exact-K shared-valid background-aware layout generation",
        "rows": rows,
        "inputs": [str(path) for path in args.inputs],
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in rows:
        speedup = f"{row['arena_speedup']:.2f}x" if row["arena_speedup"] is not None else "not-comparable"
        print(
            f"{row['scenario']:18} K={row['target_layouts']:4} "
            f"Arena={row['arena_unique_valid_layouts_per_second']:8.2f}/s "
            f"RoboLab={row['robolab_unique_valid_layouts_per_second']:8.2f}/s "
            f"speedup={speedup}"
        )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--algorithm", choices=("arena", "robolab"), required=True)
    generate_parser.add_argument("--robolab-root", type=Path)
    generate_parser.add_argument("--num-objects", type=int, default=10)
    generate_parser.add_argument("--target-layout-counts", type=_parse_ints, default=(128, 512, 2048))
    generate_parser.add_argument("--repetitions", type=int, default=3)
    generate_parser.add_argument("--max-attempts-per-layout", type=int, default=4)
    generate_parser.add_argument("--max-iterations", type=int, default=600)
    generate_parser.add_argument("--seed", type=int, default=10_000)
    generate_parser.add_argument("--robolab-mode", choices=("native", "matched-aabb"), default="native")
    generate_parser.add_argument(
        "--allow-robolab-relaxation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    generate_parser.add_argument(
        "--scenario-names",
        type=lambda value: tuple(value.split(",")),
        default=tuple(scenario.name for scenario in SCENARIOS),
    )
    generate_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("inputs", nargs=2, type=Path)
    analyze_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        if args.algorithm == "robolab" and args.robolab_root is None:
            parser.error("--robolab-root is required for RoboLab")
        if (
            min(
                args.num_objects,
                args.repetitions,
                args.max_attempts_per_layout,
                args.max_iterations,
            )
            <= 0
        ):
            parser.error("generation counts must be positive")
        unknown_scenarios = set(args.scenario_names) - {scenario.name for scenario in SCENARIOS}
        if unknown_scenarios:
            parser.error(f"unknown scenarios: {sorted(unknown_scenarios)}")
        if not args.scenario_names:
            parser.error("--scenario-names must select at least one scenario")
    return args


def main() -> int:
    args = _parse_args()
    return generate(args) if args.command == "generate" else analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
