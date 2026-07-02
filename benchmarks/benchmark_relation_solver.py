# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark relation-solver collision work across object and candidate counts."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import platform
import statistics
import subprocess
import time
import torch
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relation_solver_state import RelationSolverState
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

PHASE_COLLISION_FORWARD = "collision_forward"
PHASE_COLLISION_FORWARD_BACKWARD = "collision_forward_backward"
PHASE_SOLVE = "solve"
PHASES = (PHASE_COLLISION_FORWARD, PHASE_COLLISION_FORWARD_BACKWARD, PHASE_SOLVE)
LAYOUTS = ("dense", "sparse")


@dataclass
class Measurement:
    """Summary statistics for repeated operation timings."""

    samples_ms: list[float]
    peak_memory_mib_samples: list[float]
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    stddev_ms: float
    median_peak_memory_mib: float | None


@dataclass
class BenchmarkResult:
    """One benchmark result for a phase and scene configuration."""

    phase: str
    layout: str
    num_movable_objects: int
    batch_size: int
    eligible_directed_pairs: int
    initial_selected_directed_pair_instances: int
    iterations_per_sample: int
    warmups: int
    repeats: int
    samples_ms: list[float]
    peak_memory_mib_samples: list[float]
    initial_collision_loss_mean: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    stddev_ms: float
    median_ms_per_iteration: float
    median_peak_memory_mib: float | None
    baseline_median_ms: float | None = None
    speedup_vs_baseline: float | None = None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object-counts",
        type=_positive_int,
        nargs="+",
        default=[8, 16, 32],
        help="Movable-object counts to benchmark (default: 8 16 32).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=_positive_int,
        nargs="+",
        default=[1, 10, 50],
        help="Candidate batch sizes to benchmark (default: 1 10 50).",
    )
    parser.add_argument(
        "--layouts",
        choices=LAYOUTS,
        nargs="+",
        default=list(LAYOUTS),
        help="Initial layouts to benchmark (default: dense sparse).",
    )
    parser.add_argument(
        "--phases",
        choices=PHASES,
        nargs="+",
        default=list(PHASES),
        help="Benchmark phases to run.",
    )
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        default=25,
        help="Fixed solver iterations per solve sample (default: 25).",
    )
    parser.add_argument("--warmups", type=_positive_int, default=3, help="Warmup samples per case (default: 3).")
    parser.add_argument("--repeats", type=_positive_int, default=10, help="Measured samples per case (default: 10).")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--baseline", type=Path, help="Optional prior JSON result to compare against.")
    return parser.parse_args()


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _device_metadata(device: torch.device) -> dict[str, str | bool | None]:
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        cuda_version = torch.version.cuda
    else:
        device_name = platform.processor() or platform.machine()
        cuda_version = None
    return {
        "device": str(device),
        "device_name": device_name,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": cuda_version,
    }


def _metadata(device: torch.device, args: argparse.Namespace) -> dict:
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": platform.node(),
        "git_commit": _run_git("rev-parse", "HEAD"),
        "git_branch": _run_git("branch", "--show-current"),
        "git_dirty": bool(_run_git("status", "--porcelain")),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        **_device_metadata(device),
        "configuration": {
            "object_counts": args.object_counts,
            "batch_sizes": args.batch_sizes,
            "layouts": args.layouts,
            "phases": args.phases,
            "iterations": args.iterations,
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _measure(
    operation: Callable[[], object],
    device: torch.device,
    warmups: int,
    repeats: int,
) -> Measurement:
    for _ in range(warmups):
        result = operation()
        _synchronize(device)
        del result

    elapsed_ms: list[float] = []
    peak_memory_bytes: list[int] = []
    for _ in range(repeats):
        _synchronize(device)
        if device.type == "cuda":
            baseline_memory = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        result = operation()
        _synchronize(device)
        elapsed_ms.append((time.perf_counter() - start) * 1e3)
        del result
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated(device)
            peak_memory_bytes.append(max(0, peak_memory - baseline_memory))

    median_peak_memory_mib = None
    peak_memory_mib_samples = [value / (1024**2) for value in peak_memory_bytes]
    if peak_memory_bytes:
        median_peak_memory_mib = statistics.median(peak_memory_mib_samples)
    return Measurement(
        samples_ms=elapsed_ms,
        peak_memory_mib_samples=peak_memory_mib_samples,
        median_ms=statistics.median(elapsed_ms),
        p95_ms=_percentile(elapsed_ms, 0.95),
        min_ms=min(elapsed_ms),
        max_ms=max(elapsed_ms),
        stddev_ms=statistics.pstdev(elapsed_ms),
        median_peak_memory_mib=median_peak_memory_mib,
    )


def _make_scene(
    num_movable_objects: int,
    batch_size: int,
    layout: str,
) -> tuple[list[DummyObject], list[dict[DummyObject, tuple[float, float, float]]]]:
    box_size = 0.1
    box_half = box_size / 2
    grid_width = math.ceil(math.sqrt(num_movable_objects))
    sparse_spacing = box_size * 2
    table_half_extent = max(1.0, grid_width * sparse_spacing)

    table = DummyObject(
        name="table",
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-table_half_extent, -table_half_extent, 0.0),
            max_point=(table_half_extent, table_half_extent, 0.1),
        ),
    )
    table.set_initial_pose(Pose.identity())
    table.add_relation(IsAnchor())

    movable_objects: list[DummyObject] = []
    for object_idx in range(num_movable_objects):
        obj = DummyObject(
            name=f"box_{object_idx}",
            bounding_box=AxisAlignedBoundingBox(
                min_point=(-box_half, -box_half, -box_half),
                max_point=(box_half, box_half, box_half),
            ),
        )
        obj.add_relation(On(table, clearance_m=0.01))
        movable_objects.append(obj)

    objects = [table, *movable_objects]
    initial_positions: list[dict[DummyObject, tuple[float, float, float]]] = []
    for candidate_idx in range(batch_size):
        positions: dict[DummyObject, tuple[float, float, float]] = {table: (0.0, 0.0, 0.0)}
        candidate_offset = ((candidate_idx % 7) - 3) * 1e-4
        for object_idx, obj in enumerate(movable_objects):
            if layout == "dense":
                angle = 2 * math.pi * object_idx / num_movable_objects
                x = 0.1 * box_size * math.cos(angle) + candidate_offset
                y = 0.1 * box_size * math.sin(angle) - candidate_offset
            else:
                row, column = divmod(object_idx, grid_width)
                x = (column - (grid_width - 1) / 2) * sparse_spacing + candidate_offset
                y = (row - (grid_width - 1) / 2) * sparse_spacing - candidate_offset
            positions[obj] = (x, y, 0.1 + 0.01 + box_half)
        initial_positions.append(positions)
    return objects, initial_positions


def _make_solver(max_iters: int) -> RelationSolver:
    return RelationSolver(
        params=RelationSolverParams(
            max_iters=max_iters,
            convergence_threshold=-1.0,
            verbose=False,
            profile=False,
            save_position_history=False,
        )
    )


def _preflight_case(
    objects: list[DummyObject],
    initial_positions: list[dict[DummyObject, tuple[float, float, float]]],
    device: torch.device,
) -> tuple[int, float]:
    solver = _make_solver(max_iters=0)
    state = RelationSolverState(objects, initial_positions, device=device)
    with torch.no_grad():
        loss = solver._compute_no_overlap_loss(state)  # pyright: ignore[reportPrivateUsage]
    selected_pair_instances = solver._last_selected_no_overlap_pair_count  # pyright: ignore[reportPrivateUsage]
    return selected_pair_instances, loss.mean().item()


def _benchmark_phase(
    phase: str,
    objects: list[DummyObject],
    initial_positions: list[dict[DummyObject, tuple[float, float, float]]],
    device: torch.device,
    iterations: int,
    warmups: int,
    repeats: int,
) -> tuple[Measurement, int]:
    if phase == PHASE_SOLVE:

        def operation() -> object:
            solver = _make_solver(max_iters=iterations)
            return solver.solve(objects, initial_positions)

        return _measure(operation, device, warmups, repeats), iterations

    solver = _make_solver(max_iters=0)
    state = RelationSolverState(objects, initial_positions, device=device)
    if phase == PHASE_COLLISION_FORWARD:

        def operation() -> object:
            with torch.no_grad():
                return solver._compute_no_overlap_loss(state)  # pyright: ignore[reportPrivateUsage]

    elif phase == PHASE_COLLISION_FORWARD_BACKWARD:
        optimizable_positions = state.optimizable_positions
        assert optimizable_positions is not None

        def operation() -> object:
            optimizable_positions.grad = None
            loss = solver._compute_no_overlap_loss(state).sum()  # pyright: ignore[reportPrivateUsage]
            loss.backward()
            return loss.detach()

    else:
        raise ValueError(f"Unknown benchmark phase: {phase}")

    return _measure(operation, device, warmups, repeats), 1


def _result_key(result: BenchmarkResult | dict) -> tuple[str, str, int, int, int, int, int]:
    if isinstance(result, BenchmarkResult):
        return (
            result.phase,
            result.layout,
            result.num_movable_objects,
            result.batch_size,
            result.iterations_per_sample,
            result.warmups,
            result.repeats,
        )
    return (
        result["phase"],
        result["layout"],
        result["num_movable_objects"],
        result["batch_size"],
        result["iterations_per_sample"],
        result["warmups"],
        result["repeats"],
    )


def _add_baseline_comparisons(results: list[BenchmarkResult], baseline_path: Path, current_metadata: dict) -> None:
    baseline_payload = json.loads(baseline_path.read_text())
    baseline_metadata = baseline_payload["metadata"]
    compatibility_fields = ("device", "device_name", "torch_version", "cuda_version")
    mismatches = [
        f"{field}: baseline={baseline_metadata.get(field)!r}, current={current_metadata.get(field)!r}"
        for field in compatibility_fields
        if baseline_metadata.get(field) != current_metadata.get(field)
    ]
    if mismatches:
        raise ValueError("Baseline environment is incompatible:\n  " + "\n  ".join(mismatches))
    baseline_by_key = {_result_key(result): result for result in baseline_payload["results"]}
    for result in results:
        baseline = baseline_by_key.get(_result_key(result))
        if baseline is None:
            continue
        result.baseline_median_ms = baseline["median_ms"]
        result.speedup_vs_baseline = result.baseline_median_ms / result.median_ms


def _print_results(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'phase':<28} {'layout':<7} {'objects':>7} {'batch':>5} {'selected':>8} "
        f"{'median ms':>10} {'p95 ms':>10} {'ms/iter':>10} {'peak MiB':>10} {'speedup':>9}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        peak_memory = "-" if result.median_peak_memory_mib is None else f"{result.median_peak_memory_mib:.2f}"
        speedup = "-" if result.speedup_vs_baseline is None else f"{result.speedup_vs_baseline:.3f}x"
        print(
            f"{result.phase:<28} {result.layout:<7} {result.num_movable_objects:>7} "
            f"{result.batch_size:>5} {result.initial_selected_directed_pair_instances:>8} "
            f"{result.median_ms:>10.3f} {result.p95_ms:>10.3f} "
            f"{result.median_ms_per_iteration:>10.3f} {peak_memory:>10} {speedup:>9}"
        )


def main() -> None:
    """Run the configured benchmark matrix and optionally write JSON results."""
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = _metadata(device, args)
    results: list[BenchmarkResult] = []

    for layout in args.layouts:
        for num_movable_objects in args.object_counts:
            assert num_movable_objects >= 2, "object counts must be at least 2 to benchmark object-object collisions."
            eligible_directed_pairs = num_movable_objects * (num_movable_objects - 1)
            for batch_size in args.batch_sizes:
                objects, initial_positions = _make_scene(num_movable_objects, batch_size, layout)
                initial_selected_pairs, initial_collision_loss = _preflight_case(objects, initial_positions, device)
                for phase in args.phases:
                    measurement, iterations_per_sample = _benchmark_phase(
                        phase,
                        objects,
                        initial_positions,
                        device,
                        args.iterations,
                        args.warmups,
                        args.repeats,
                    )
                    results.append(
                        BenchmarkResult(
                            phase=phase,
                            layout=layout,
                            num_movable_objects=num_movable_objects,
                            batch_size=batch_size,
                            eligible_directed_pairs=eligible_directed_pairs,
                            initial_selected_directed_pair_instances=initial_selected_pairs,
                            iterations_per_sample=iterations_per_sample,
                            warmups=args.warmups,
                            repeats=args.repeats,
                            samples_ms=measurement.samples_ms,
                            peak_memory_mib_samples=measurement.peak_memory_mib_samples,
                            initial_collision_loss_mean=initial_collision_loss,
                            median_ms=measurement.median_ms,
                            p95_ms=measurement.p95_ms,
                            min_ms=measurement.min_ms,
                            max_ms=measurement.max_ms,
                            stddev_ms=measurement.stddev_ms,
                            median_ms_per_iteration=measurement.median_ms / iterations_per_sample,
                            median_peak_memory_mib=measurement.median_peak_memory_mib,
                        )
                    )

    if args.baseline is not None:
        _add_baseline_comparisons(results, args.baseline, metadata)
    _print_results(results)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 2,
            "metadata": metadata,
            "results": [asdict(result) for result in results],
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nWrote benchmark results to {args.output}")


if __name__ == "__main__":
    main()
