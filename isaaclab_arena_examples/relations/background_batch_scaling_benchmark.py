# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure how passive background objects shift the Arena/RoboLab batch crossover."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import io
import json
import numpy as np
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from isaaclab_arena.relations.benchmark.provenance import collect_source_revision
from isaaclab_arena_examples.relations.fixed_iteration_batch_scaling_benchmark import (
    OBJECT_HEIGHT_M,
    OBJECT_SIZE_M,
    TABLE_BOUNDS,
    _build_arena_workload,
    _clustered_xy,
    _parse_ints,
    _source_revision,
)

Algorithm = Literal["arena", "robolab"]
BACKGROUND_SIZE_M = 0.12
BACKGROUND_CENTERS = (
    (0.0, 0.0),
    (-0.22, 0.0),
    (0.22, 0.0),
    (0.0, -0.22),
    (0.0, 0.22),
    (-0.22, -0.22),
    (-0.22, 0.22),
    (0.22, -0.22),
    (0.22, 0.22),
)
DEFAULT_BATCH_SIZES = tuple(2**exponent for exponent in range(14))


@dataclass(frozen=True)
class BackgroundScenario:
    """One progressive immutable-background workload."""

    name: str
    obstacle_count: int

    @property
    def centers(self) -> tuple[tuple[float, float], ...]:
        return BACKGROUND_CENTERS[: self.obstacle_count]

    @property
    def nominal_area_fraction(self) -> float:
        table_area = (TABLE_BOUNDS[1] - TABLE_BOUNDS[0]) * (TABLE_BOUNDS[3] - TABLE_BOUNDS[2])
        return self.obstacle_count * BACKGROUND_SIZE_M**2 / table_area


SCENARIOS = (
    BackgroundScenario("no-background", 0),
    BackgroundScenario("central-background", 1),
    BackgroundScenario("cross-background", 5),
    BackgroundScenario("grid-background", 9),
)


class FixedAabbBackground:
    """Minimal passive AABB collision object in world coordinates."""

    def __init__(self, name: str, center_xy: tuple[float, float]) -> None:
        from isaaclab_arena.relations.collision_mode import CollisionMode
        from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
        from isaaclab_arena.utils.pose import Pose

        half_size = BACKGROUND_SIZE_M / 2
        self.name = name
        self.collision_mode = CollisionMode.BBOX
        self.repair_collision_mesh_non_watertight = False
        self._pose = Pose.identity()
        self._bbox = AxisAlignedBoundingBox(
            min_point=(center_xy[0] - half_size, center_xy[1] - half_size, 0.0),
            max_point=(center_xy[0] + half_size, center_xy[1] + half_size, OBJECT_HEIGHT_M),
        )

    @property
    def is_anchor(self) -> bool:
        return False

    def get_initial_pose(self):
        return self._pose

    def get_bounding_box(self):
        return self._bbox

    def get_world_bounding_box(self):
        return self._bbox

    def get_collision_mesh(self):
        return None


def _backgrounds(scenario: BackgroundScenario) -> list[FixedAabbBackground]:
    return [FixedAabbBackground(f"background-{index}", center) for index, center in enumerate(scenario.centers)]


def _aabb_axis_displacement(
    first,
    second,
    first_dims,
    second_dims,
    collision_margin: float,
    move_both: bool,
) -> None:
    dx = first.x - second.x
    dy = first.y - second.y
    penetration_x = (first_dims[0] + second_dims[0]) / 2 + 2 * collision_margin - abs(dx)
    penetration_y = (first_dims[1] + second_dims[1]) / 2 + 2 * collision_margin - abs(dy)
    extra_buffer = 0.01
    if penetration_x <= penetration_y:
        direction = 1.0 if dx >= 0 else -1.0
        displacement = penetration_x + extra_buffer
        first.x += direction * (displacement / 2 if move_both else displacement)
        if move_both:
            second.x -= direction * displacement / 2
    else:
        direction = 1.0 if dy >= 0 else -1.0
        displacement = penetration_y + extra_buffer
        first.y += direction * (displacement / 2 if move_both else displacement)
        if move_both:
            second.y -= direction * displacement / 2


def _measure_arena(
    scenario: BackgroundScenario,
    num_objects: int,
    batch_size: int,
    calibration_layouts: int,
    iterations: int,
    repeat: int,
) -> dict:
    import torch

    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

    class CollisionOnlyRelationSolver(RelationSolver):
        def _compute_total_loss(self, state, debug: bool = False):
            total_loss = self._compute_no_overlap_loss(state, debug)
            self._last_loss_per_env = total_loss.detach().clone()
            return total_loss.mean()

    objects, positions = _build_arena_workload(num_objects, batch_size)
    table = objects[0]
    for env_index, env_positions in enumerate(positions):
        layout = _clustered_xy(num_objects, env_index % calibration_layouts)
        for obj in objects:
            if obj is not table:
                env_positions[obj] = (*layout[obj.name], OBJECT_HEIGHT_M / 2)
    solver = CollisionOnlyRelationSolver(
        RelationSolverParams(
            max_iters=iterations,
            convergence_threshold=0.0,
            clearance_m=0.0,
            verbose=False,
            save_position_history=False,
        )
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    solver.solve(objects, positions, collision_objects=_backgrounds(scenario))
    elapsed_ms = solver.last_optimization_elapsed_ms
    iterations_run = len(solver.last_loss_history)
    assert iterations_run == iterations
    return {
        "algorithm": "arena",
        "scenario": scenario.name,
        "background_objects": scenario.obstacle_count,
        "num_objects": num_objects,
        "batch_size": batch_size,
        "repeat": repeat,
        "iterations": iterations_run,
        "optimization_elapsed_ms": elapsed_ms,
        "layouts_per_second": batch_size * 1e3 / elapsed_ms,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else None,
        "scored_directed_pairs": solver.last_no_overlap_pair_count,
    }


def _load_robolab(robolab_root: Path):
    sys.path.insert(0, str(robolab_root.resolve()))
    from robolab.scene_gen.llm_scene_gen.predicates import ObjectState
    from robolab.scene_gen.llm_scene_gen.spatial_solver import SpatialSolver

    from isaaclab_arena_examples.relations.collision_representation_ablation import _aabb_overlap

    class BackgroundAabbSpatialSolver(SpatialSolver):
        def _check_collisions(self, object_states, object_dims):
            collisions = []
            names = list(object_states)
            fixed = self._background_names
            for index, first_name in enumerate(names):
                first = object_states[first_name]
                for second_name in names[index + 1 :]:
                    if first_name in fixed and second_name in fixed:
                        continue
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

        def _axis_displacement(self, first, second, first_dims, second_dims, move_both: bool):
            _aabb_axis_displacement(
                first,
                second,
                first_dims,
                second_dims,
                self.collision_margin,
                move_both,
            )

        def _resolve_collision(self, first, second, first_dims, second_dims):
            self._axis_displacement(first, second, first_dims, second_dims, move_both=True)

        def _move_away_from_fixed(self, movable, fixed, movable_dims, fixed_dims):
            self._axis_displacement(movable, fixed, movable_dims, fixed_dims, move_both=False)

    return ObjectState, BackgroundAabbSpatialSolver


def _make_robolab_job(scenario, num_objects, variant, robolab_api):
    ObjectState, SpatialSolver = robolab_api
    states = {
        name: ObjectState(name=name, x=xy[0], y=xy[1], yaw=0.0, is_placed=True)
        for name, xy in _clustered_xy(num_objects, variant).items()
    }
    background_names = set()
    for index, center in enumerate(scenario.centers):
        name = f"background-{index}"
        states[name] = ObjectState(name=name, x=center[0], y=center[1], yaw=0.0, is_placed=True)
        background_names.add(name)
    solver = SpatialSolver(table_bounds=TABLE_BOUNDS, collision_margin=0.0)
    solver._background_names = background_names
    return solver, states, background_names


def _run_robolab_fixed_iterations(
    solver,
    states,
    dimensions,
    background_names: set[str],
    iterations: int,
    seed: int,
) -> None:
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
                for name, state in states.items():
                    if name not in background_names:
                        state.x += random.uniform(-0.05, 0.05)
                        state.y += random.uniform(-0.05, 0.05)
                no_progress_count = 0
        else:
            no_progress_count = 0
        previous_collision_count = len(collisions)
        for first_name, second_name in collisions[:3]:
            if first_name in background_names:
                solver._move_away_from_fixed(
                    states[second_name],
                    states[first_name],
                    dimensions[second_name],
                    dimensions[first_name],
                )
            elif second_name in background_names:
                solver._move_away_from_fixed(
                    states[first_name],
                    states[second_name],
                    dimensions[first_name],
                    dimensions[second_name],
                )
            else:
                solver._resolve_collision(
                    states[first_name],
                    states[second_name],
                    dimensions[first_name],
                    dimensions[second_name],
                )


def _measure_robolab(
    scenario: BackgroundScenario,
    num_objects: int,
    batch_size: int,
    calibration_layouts: int,
    iterations: int,
    repeat: int,
    robolab_api,
) -> dict:
    dimensions = {f"object-{index}": (OBJECT_SIZE_M, OBJECT_SIZE_M, OBJECT_HEIGHT_M) for index in range(num_objects)}
    dimensions.update({
        f"background-{index}": (BACKGROUND_SIZE_M, BACKGROUND_SIZE_M, OBJECT_HEIGHT_M)
        for index in range(scenario.obstacle_count)
    })
    jobs = [
        _make_robolab_job(scenario, num_objects, variant % calibration_layouts, robolab_api)
        for variant in range(batch_size)
    ]
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        for variant, (solver, states, background_names) in enumerate(jobs):
            _run_robolab_fixed_iterations(
                solver,
                states,
                dimensions,
                background_names,
                iterations,
                seed=repeat * batch_size + variant,
            )
    elapsed_ms = (time.perf_counter() - start) * 1e3
    per_layout_ms = elapsed_ms / batch_size
    return {
        "algorithm": "robolab",
        "scenario": scenario.name,
        "background_objects": scenario.obstacle_count,
        "num_objects": num_objects,
        "batch_size": batch_size,
        "calibration_layouts": calibration_layouts,
        "repeat": repeat,
        "iterations": iterations,
        "optimization_elapsed_ms": elapsed_ms,
        "per_layout_optimization_ms": per_layout_ms,
        "layouts_per_second": 1e3 / per_layout_ms,
        "batch_scaling_model": "direct-serial-measurement",
    }


def _cpu_model() -> str:
    """Return the host CPU model when Linux exposes it."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor()


def _runtime_metadata(torch) -> dict:
    """Collect hardware and runtime identity used by both benchmark methods."""
    cuda_available = torch.cuda.is_available()
    cuda_device = torch.cuda.current_device() if cuda_available else None
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device": cuda_device,
        "gpu_name": torch.cuda.get_device_name(cuda_device) if cuda_available else None,
    }


def generate(args: argparse.Namespace) -> int:
    import torch

    benchmark_root = Path(__file__).resolve().parents[2]
    measurements = []
    stopped = {}
    scenarios = tuple(scenario for scenario in SCENARIOS if scenario.name in args.scenario_names)
    if args.algorithm == "arena":
        for scenario in scenarios:
            for batch_size in args.batch_sizes:
                try:
                    _measure_arena(
                        scenario,
                        args.num_objects,
                        batch_size,
                        args.calibration_layouts,
                        args.iterations,
                        repeat=-1,
                    )
                    batch_measurements = [
                        _measure_arena(
                            scenario,
                            args.num_objects,
                            batch_size,
                            args.calibration_layouts,
                            args.iterations,
                            repeat,
                        )
                        for repeat in range(args.repetitions)
                    ]
                    measurements.extend(batch_measurements)
                    print(f"Arena {scenario.name:18} batch={batch_size:5}: complete")
                except RuntimeError as error:
                    if "out of memory" not in str(error).lower():
                        raise
                    stopped[scenario.name] = {"batch_size": batch_size, "reason": "cuda-out-of-memory"}
                    break
    else:
        robolab_api = _load_robolab(args.robolab_root)
        for scenario in scenarios:
            _measure_robolab(
                scenario,
                args.num_objects,
                args.calibration_layouts,
                args.calibration_layouts,
                args.iterations,
                repeat=-1,
                robolab_api=robolab_api,
            )
            for batch_size in args.batch_sizes:
                measurements.extend(
                    _measure_robolab(
                        scenario,
                        args.num_objects,
                        batch_size,
                        args.calibration_layouts,
                        args.iterations,
                        repeat,
                        robolab_api,
                    )
                    for repeat in range(args.repetitions)
                )
                print(f"RoboLab {scenario.name:18} batch={batch_size:5}: complete")
    source_root = args.robolab_root if args.algorithm == "robolab" else benchmark_root
    payload = {
        "schema_version": 2,
        "algorithm": args.algorithm,
        "source_revision": _source_revision(source_root),
        "benchmark_revision": collect_source_revision(benchmark_root),
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "num_objects": args.num_objects,
        "batch_sizes": args.batch_sizes,
        "iterations": args.iterations,
        "repetitions": args.repetitions,
        "calibration_layouts": args.calibration_layouts,
        "movable_size_m": OBJECT_SIZE_M,
        "background_size_m": BACKGROUND_SIZE_M,
        "background_centers": BACKGROUND_CENTERS,
        "collision_representation": "matched-aabb",
        "objective": "pairwise-collision-only-including-passive-background",
        "initialization": "deterministic-overlapping-cluster",
        "variant_population": f"variants 0..{args.calibration_layouts - 1}, cycled for larger Arena batches",
        "timing_scope": "optimization-loop-only",
        "early_stopping": False,
        "method_label": "Arena" if args.algorithm == "arena" else "RoboLab-derived matched-AABB serial loop",
        "runtime": _runtime_metadata(torch),
        "run_timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
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
        "scenarios",
        "num_objects",
        "batch_sizes",
        "iterations",
        "repetitions",
        "calibration_layouts",
        "movable_size_m",
        "background_size_m",
        "background_centers",
        "collision_representation",
        "objective",
        "initialization",
        "variant_population",
        "timing_scope",
        "early_stopping",
        "runtime",
    )
    for field in compatibility:
        if keyed["arena"][field] != keyed["robolab"][field]:
            raise ValueError(f"incompatible {field}")
    rows = []
    crossovers = []
    scenarios = tuple(BackgroundScenario(**scenario) for scenario in keyed["arena"]["scenarios"])
    for scenario in scenarios:
        scenario_rows = []
        for batch_size in keyed["arena"]["batch_sizes"]:
            arena_measurements = [
                measurement
                for measurement in keyed["arena"]["measurements"]
                if measurement["scenario"] == scenario.name and measurement["batch_size"] == batch_size
            ]
            if len(arena_measurements) != keyed["arena"]["repetitions"]:
                continue
            robolab_measurements = [
                measurement
                for measurement in keyed["robolab"]["measurements"]
                if measurement["scenario"] == scenario.name and measurement["batch_size"] == batch_size
            ]
            if len(robolab_measurements) != keyed["robolab"]["repetitions"]:
                continue
            arena_rates = [
                batch_size * 1e3 / measurement["optimization_elapsed_ms"] for measurement in arena_measurements
            ]
            robolab_rates = [measurement["layouts_per_second"] for measurement in robolab_measurements]
            speedups = [
                arena_rate / robolab_repeat_rate
                for arena_rate, robolab_repeat_rate in zip(arena_rates, robolab_rates, strict=True)
            ]
            row = {
                "scenario": scenario.name,
                "background_objects": scenario.obstacle_count,
                "nominal_background_area_fraction": scenario.nominal_area_fraction,
                "num_objects": keyed["arena"]["num_objects"],
                "batch_size": batch_size,
                "arena_optimization_ms": statistics.median(
                    measurement["optimization_elapsed_ms"] for measurement in arena_measurements
                ),
                "arena_layouts_per_second": statistics.median(arena_rates),
                "robolab_layouts_per_second": statistics.median(robolab_rates),
                "robolab_optimization_ms": statistics.median(
                    measurement["optimization_elapsed_ms"] for measurement in robolab_measurements
                ),
                "arena_speedup": statistics.median(speedups),
                "arena_speedup_q25": float(np.percentile(speedups, 25)),
                "arena_speedup_iqr": float(np.percentile(speedups, 75) - np.percentile(speedups, 25)),
                "arena_peak_gpu_memory_mb": statistics.median(
                    measurement["peak_gpu_memory_mb"] for measurement in arena_measurements
                ),
            }
            rows.append(row)
            scenario_rows.append(row)
        crossover = next((row for row in scenario_rows if row["arena_speedup_q25"] > 1.0), None)
        largest = scenario_rows[-1]
        crossovers.append({
            "scenario": scenario.name,
            "background_objects": scenario.obstacle_count,
            "nominal_background_area_fraction": scenario.nominal_area_fraction,
            "crossover_batch_size": crossover["batch_size"] if crossover else None,
            "speedup_at_crossover": crossover["arena_speedup"] if crossover else None,
            "largest_measured_batch": largest["batch_size"],
            "speedup_at_largest_batch": largest["arena_speedup"],
        })
    output = {
        "schema_version": 2,
        "comparison": "fixed-iteration Arena versus RoboLab-derived matched-AABB serial loop",
        "iterations": keyed["arena"]["iterations"],
        "rows": rows,
        "crossovers": crossovers,
        "inputs": [str(path) for path in args.inputs],
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for crossover in crossovers:
        print(
            f"{crossover['scenario']:18} backgrounds={crossover['background_objects']:2} "
            f"crossover={str(crossover['crossover_batch_size']):>5} "
            f"largest-speedup={crossover['speedup_at_largest_batch']:6.2f}x"
        )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--algorithm", choices=("arena", "robolab"), required=True)
    generate_parser.add_argument("--robolab-root", type=Path)
    generate_parser.add_argument("--num-objects", type=int, default=10)
    generate_parser.add_argument("--batch-sizes", type=_parse_ints, default=DEFAULT_BATCH_SIZES)
    generate_parser.add_argument("--iterations", type=int, default=600)
    generate_parser.add_argument("--repetitions", type=int, default=3)
    generate_parser.add_argument("--calibration-layouts", type=int, default=16)
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
        if min(args.num_objects, args.iterations, args.repetitions, args.calibration_layouts) <= 0:
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
