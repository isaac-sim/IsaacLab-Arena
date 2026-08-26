# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Find the batch size where Arena overtakes serial RoboLab for 600 fixed iterations."""

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from isaaclab_arena.relations.benchmark.provenance import collect_source_revision

Algorithm = Literal["arena", "robolab"]
TABLE_BOUNDS = (-0.5, 0.5, -0.5, 0.5)
OBJECT_SIZE_M = 0.08
OBJECT_HEIGHT_M = 0.1
DEFAULT_OBJECT_COUNTS = (2, 5, 10, 20)
DEFAULT_BATCH_SIZES = tuple(2**exponent for exponent in range(15))


@dataclass(frozen=True)
class Crossover:
    """First measured Arena batch that exceeds projected serial RoboLab throughput."""

    num_objects: int
    robolab_layouts_per_second: float
    crossover_batch_size: int | None
    arena_layouts_per_second_at_crossover: float | None
    speedup_at_crossover: float | None
    largest_measured_batch: int
    arena_layouts_per_second_at_largest_batch: float
    speedup_at_largest_batch: float


def _parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(","))
    assert values and all(item > 0 for item in values), f"expected positive comma-separated integers, got {value}"
    return values


def _clustered_xy(num_objects: int, variant: int = 0) -> dict[str, tuple[float, float]]:
    """Return deterministic, slightly offset overlapping object centers."""
    phase = variant * 0.173
    return {
        f"object-{index}": (
            0.012 * math.cos(index * 2.399 + phase),
            0.012 * math.sin(index * 2.399 + phase),
        )
        for index in range(num_objects)
    }


def _build_arena_workload(num_objects: int, batch_size: int):
    from isaaclab_arena.relations.benchmark.layout_generation import build_arena_controlled_scene

    sizes = {f"object-{index}": (OBJECT_SIZE_M, OBJECT_SIZE_M) for index in range(num_objects)}
    objects = build_arena_controlled_scene(TABLE_BOUNDS, sizes)
    table = objects[0]
    layouts = [_clustered_xy(num_objects, env_index) for env_index in range(batch_size)]
    positions = [
        {obj: (0.0, 0.0, 0.0) if obj is table else (*layout[obj.name], OBJECT_HEIGHT_M / 2) for obj in objects}
        for layout in layouts
    ]
    return objects, positions


def _measure_arena(
    num_objects: int,
    batch_size: int,
    iterations: int,
    repeat: int,
) -> dict:
    import torch

    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

    class CollisionOnlyRelationSolver(RelationSolver):
        """Arena optimizer with only its built-in AABB no-overlap objective."""

        def _compute_total_loss(self, state, debug: bool = False):
            total_loss = self._compute_no_overlap_loss(state, debug)
            self._last_loss_per_env = total_loss.detach().clone()
            return total_loss.mean()

    objects, positions = _build_arena_workload(num_objects, batch_size)
    solver = CollisionOnlyRelationSolver(
        RelationSolverParams(
            max_iters=iterations,
            convergence_threshold=0.0,
            clearance_m=0.0,
            verbose=False,
            save_position_history=False,
        )
    )
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    solver.solve(objects, positions)
    elapsed_ms = solver.last_optimization_elapsed_ms
    iterations_run = len(solver.last_loss_history)
    assert iterations_run == iterations, f"Arena ran {iterations_run}, expected exactly {iterations}"
    peak_memory_mb = float(torch.cuda.max_memory_allocated() / 1024**2) if cuda_available else None
    return {
        "algorithm": "arena",
        "num_objects": num_objects,
        "batch_size": batch_size,
        "repeat": repeat,
        "iterations": iterations_run,
        "optimization_elapsed_ms": elapsed_ms,
        "layouts_per_second": batch_size * 1e3 / elapsed_ms,
        "peak_gpu_memory_mb": peak_memory_mb,
        "pair_count": solver.last_no_overlap_pair_count,
    }


def _measure_arena_batch(
    num_objects: int,
    batch_size: int,
    iterations: int,
    repetitions: int,
) -> list[dict]:
    """Warm and atomically measure every repetition for one Arena batch."""
    _measure_arena(num_objects, batch_size, iterations, repeat=-1)
    return [_measure_arena(num_objects, batch_size, iterations, repeat) for repeat in range(repetitions)]


def _load_robolab(robolab_root: Path):
    sys.path.insert(0, str(robolab_root.resolve()))
    from robolab.scene_gen.llm_scene_gen.predicates import ObjectState
    from robolab.scene_gen.llm_scene_gen.spatial_solver import SpatialSolver

    from isaaclab_arena_examples.relations.collision_representation_ablation import _aabb_overlap

    class AabbSpatialSolver(SpatialSolver):
        """RoboLab iterative pushing with a matched AABB collision oracle."""

        def _check_collisions(self, object_states, object_dims):
            collisions = []
            names = list(object_states)
            for index, first_name in enumerate(names):
                first = object_states[first_name]
                for second_name in names[index + 1 :]:
                    second = object_states[second_name]
                    if _aabb_overlap(
                        (first.x, first.y),
                        (second.x, second.y),
                        object_dims[first_name],
                        object_dims[second_name],
                        self.collision_margin,
                    ):
                        collisions.append((first_name, second_name))
            return collisions

        def _resolve_collision(self, first, second, first_dims, second_dims):
            dx = first.x - second.x
            dy = first.y - second.y
            separation_x = (first_dims[0] + second_dims[0]) / 2 + 2 * self.collision_margin
            separation_y = (first_dims[1] + second_dims[1]) / 2 + 2 * self.collision_margin
            penetration_x = separation_x - abs(dx)
            penetration_y = separation_y - abs(dy)
            average_size = (max(first_dims[0], first_dims[1]) + max(second_dims[0], second_dims[1])) / 2
            extra_buffer = 0.02 if average_size > 0.2 else 0.01
            if penetration_x <= penetration_y:
                direction = 1.0 if dx >= 0 else -1.0
                displacement = penetration_x / 2 + extra_buffer
                first.x += direction * displacement
                second.x -= direction * displacement
            else:
                direction = 1.0 if dy >= 0 else -1.0
                displacement = penetration_y / 2 + extra_buffer
                first.y += direction * displacement
                second.y -= direction * displacement

    return ObjectState, AabbSpatialSolver


def _make_robolab_state(num_objects: int, variant: int, ObjectState):
    return {
        name: ObjectState(name=name, x=xy[0], y=xy[1], yaw=0.0, is_placed=True)
        for name, xy in _clustered_xy(num_objects, variant).items()
    }


def _run_robolab_fixed_iterations(solver, states, dimensions, iterations: int, seed: int) -> None:
    """Run RoboLab's collision-check/push loop without its early return."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    previous_collision_count = float("inf")
    no_progress_count = 0
    for _ in range(iterations):
        collisions = solver._check_collisions(states, dimensions)
        if not collisions:
            continue
        if len(collisions) >= previous_collision_count:
            no_progress_count += 1
            if no_progress_count > 10:
                for state in states.values():
                    if state.x is not None:
                        state.x += random.uniform(-0.05, 0.05)
                        state.y += random.uniform(-0.05, 0.05)
                no_progress_count = 0
        else:
            no_progress_count = 0
        previous_collision_count = len(collisions)
        for first_name, second_name in collisions[:3]:
            solver._resolve_collision(
                states[first_name],
                states[second_name],
                dimensions[first_name],
                dimensions[second_name],
            )


def _measure_robolab(
    num_objects: int,
    calibration_layouts: int,
    iterations: int,
    repeat: int,
    robolab_api,
) -> dict:
    ObjectState, SpatialSolver = robolab_api
    dimensions = {f"object-{index}": (OBJECT_SIZE_M, OBJECT_SIZE_M, OBJECT_HEIGHT_M) for index in range(num_objects)}
    jobs = [
        (
            SpatialSolver(table_bounds=TABLE_BOUNDS, collision_margin=0.0),
            _make_robolab_state(num_objects, variant, ObjectState),
        )
        for variant in range(calibration_layouts)
    ]
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        for variant, (solver, states) in enumerate(jobs):
            _run_robolab_fixed_iterations(
                solver,
                states,
                dimensions,
                iterations,
                seed=repeat * calibration_layouts + variant,
            )
    elapsed_ms = (time.perf_counter() - start) * 1e3
    per_layout_ms = elapsed_ms / calibration_layouts
    return {
        "algorithm": "robolab",
        "num_objects": num_objects,
        "calibration_layouts": calibration_layouts,
        "repeat": repeat,
        "iterations": iterations,
        "optimization_elapsed_ms": elapsed_ms,
        "per_layout_optimization_ms": per_layout_ms,
        "layouts_per_second": 1e3 / per_layout_ms,
        "batch_scaling_model": "serial-linear-projection",
    }


def _source_revision(root: Path) -> str | None:
    from isaaclab_arena_examples.relations.collision_space_coverage_benchmark import _source_revision

    return _source_revision(root)


def generate(args: argparse.Namespace) -> int:
    benchmark_root = Path(__file__).resolve().parents[2]
    measurements = []
    stopped = {}
    if args.algorithm == "arena":
        for num_objects in args.object_counts:
            for batch_size in args.batch_sizes:
                try:
                    measurements.extend(
                        _measure_arena_batch(
                            num_objects,
                            batch_size,
                            args.iterations,
                            args.repetitions,
                        )
                    )
                    print(f"Arena objects={num_objects:2} batch={batch_size:5}: complete")
                except RuntimeError as error:
                    if "out of memory" not in str(error).lower():
                        raise
                    stopped[str(num_objects)] = {
                        "batch_size": batch_size,
                        "reason": "cuda-out-of-memory",
                    }
                    print(f"Arena objects={num_objects:2} batch={batch_size:5}: CUDA OOM, stopping sweep")
                    break
    else:
        robolab_api = _load_robolab(args.robolab_root)
        for num_objects in args.object_counts:
            _measure_robolab(num_objects, args.calibration_layouts, args.iterations, repeat=-1, robolab_api=robolab_api)
            for repeat in range(args.repetitions):
                measurements.append(
                    _measure_robolab(
                        num_objects,
                        args.calibration_layouts,
                        args.iterations,
                        repeat,
                        robolab_api,
                    )
                )
            print(f"RoboLab objects={num_objects:2}: calibrated")
    source_root = args.robolab_root if args.algorithm == "robolab" else benchmark_root
    payload = {
        "schema_version": 1,
        "algorithm": args.algorithm,
        "source_revision": _source_revision(source_root),
        "benchmark_revision": collect_source_revision(benchmark_root),
        "object_counts": args.object_counts,
        "batch_sizes": args.batch_sizes,
        "iterations": args.iterations,
        "repetitions": args.repetitions,
        "calibration_layouts": args.calibration_layouts,
        "object_size_m": OBJECT_SIZE_M,
        "table_bounds": TABLE_BOUNDS,
        "initialization": "deterministic-overlapping-cluster",
        "collision_representation": "matched-aabb",
        "objective": "pairwise-collision-only",
        "timing_scope": "optimization-loop-only",
        "early_stopping": False,
        "robolab_scaling": "measured serial per-layout cost projected linearly across batch size",
        "stopped": stopped,
        "measurements": measurements,
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


def analyze(args: argparse.Namespace) -> int:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    keyed = {run["algorithm"]: run for run in runs}
    if set(keyed) != {"arena", "robolab"}:
        raise ValueError("analysis requires one Arena and one RoboLab run")
    compatibility = (
        "schema_version",
        "benchmark_revision",
        "object_counts",
        "batch_sizes",
        "iterations",
        "repetitions",
        "calibration_layouts",
        "object_size_m",
        "table_bounds",
        "initialization",
        "collision_representation",
        "objective",
        "timing_scope",
        "early_stopping",
    )
    for field in compatibility:
        if keyed["arena"][field] != keyed["robolab"][field]:
            raise ValueError(f"incompatible {field}")
    rows = []
    crossovers = []
    for num_objects in keyed["arena"]["object_counts"]:
        robolab_rates = [
            measurement["layouts_per_second"]
            for measurement in keyed["robolab"]["measurements"]
            if measurement["num_objects"] == num_objects
        ]
        robolab_rate = statistics.median(robolab_rates)
        object_rows = []
        for batch_size in keyed["arena"]["batch_sizes"]:
            arena_measurements = [
                measurement
                for measurement in keyed["arena"]["measurements"]
                if measurement["num_objects"] == num_objects and measurement["batch_size"] == batch_size
            ]
            if len(arena_measurements) != keyed["arena"]["repetitions"]:
                continue
            arena_elapsed = statistics.median(
                measurement["optimization_elapsed_ms"] for measurement in arena_measurements
            )
            arena_rate = batch_size * 1e3 / arena_elapsed
            arena_rates = [
                batch_size * 1e3 / measurement["optimization_elapsed_ms"] for measurement in arena_measurements
            ]
            paired_speedups = [
                arena_repeat_rate / robolab_repeat_rate
                for arena_repeat_rate, robolab_repeat_rate in zip(
                    arena_rates,
                    robolab_rates,
                    strict=True,
                )
            ]
            row = {
                "num_objects": num_objects,
                "batch_size": batch_size,
                "arena_optimization_ms": arena_elapsed,
                "arena_layouts_per_second": arena_rate,
                "robolab_projected_batch_ms": batch_size * 1e3 / robolab_rate,
                "robolab_layouts_per_second": robolab_rate,
                "arena_speedup": arena_rate / robolab_rate,
                "arena_speedup_paired_iqr": float(
                    np.percentile(paired_speedups, 75) - np.percentile(paired_speedups, 25)
                ),
                "arena_speedup_paired_q25": float(np.percentile(paired_speedups, 25)),
                "arena_peak_gpu_memory_mb": statistics.median(
                    measurement["peak_gpu_memory_mb"] for measurement in arena_measurements
                ),
            }
            rows.append(row)
            object_rows.append(row)
        crossover_row = next((row for row in object_rows if row["arena_speedup_paired_q25"] > 1.0), None)
        largest = object_rows[-1]
        crossovers.append(
            Crossover(
                num_objects=num_objects,
                robolab_layouts_per_second=robolab_rate,
                crossover_batch_size=crossover_row["batch_size"] if crossover_row else None,
                arena_layouts_per_second_at_crossover=(
                    crossover_row["arena_layouts_per_second"] if crossover_row else None
                ),
                speedup_at_crossover=crossover_row["arena_speedup"] if crossover_row else None,
                largest_measured_batch=largest["batch_size"],
                arena_layouts_per_second_at_largest_batch=largest["arena_layouts_per_second"],
                speedup_at_largest_batch=largest["arena_speedup"],
            )
        )
    output = {
        "schema_version": 1,
        "comparison": "fixed-iteration optimization-only; Arena measured batches vs serial RoboLab projection",
        "iterations": keyed["arena"]["iterations"],
        "rows": rows,
        "crossovers": [asdict(crossover) for crossover in crossovers],
        "inputs": [str(path) for path in args.inputs],
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for crossover in crossovers:
        print(
            f"objects={crossover.num_objects:2} crossover={str(crossover.crossover_batch_size):>5} "
            f"largest={crossover.largest_measured_batch:5} speedup={crossover.speedup_at_largest_batch:7.2f}x"
        )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--algorithm", choices=("arena", "robolab"), required=True)
    generate_parser.add_argument("--robolab-root", type=Path)
    generate_parser.add_argument("--object-counts", type=_parse_ints, default=DEFAULT_OBJECT_COUNTS)
    generate_parser.add_argument("--batch-sizes", type=_parse_ints, default=DEFAULT_BATCH_SIZES)
    generate_parser.add_argument("--iterations", type=int, default=600)
    generate_parser.add_argument("--repetitions", type=int, default=3)
    generate_parser.add_argument("--calibration-layouts", type=int, default=16)
    generate_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("inputs", nargs=2, type=Path)
    analyze_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        if args.algorithm == "robolab" and args.robolab_root is None:
            parser.error("--robolab-root is required for RoboLab")
        if args.iterations <= 0 or args.repetitions <= 0 or args.calibration_layouts <= 0:
            parser.error("iterations, repetitions, and calibration layouts must be positive")
    return args


def main() -> int:
    args = _parse_args()
    return generate(args) if args.command == "generate" else analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
