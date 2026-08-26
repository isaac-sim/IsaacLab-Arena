# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure native solver coverage of exact collision-free space."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numpy as np
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from scipy.spatial import cKDTree
from typing import TYPE_CHECKING, Literal

from isaaclab_arena.relations.benchmark.provenance import collect_source_revision

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset

Method = Literal["arena", "robolab", "reference"]
Shape = Literal["box", "disk"]
TableBounds = tuple[float, float, float, float]

TABLE_BOUNDS: TableBounds = (-0.5, 0.5, -0.5, 0.5)
MOVABLE_SIZE_M = 0.06
OBSTACLE_CENTERS = ((-0.27, 0.25), (0.22, 0.22), (-0.02, 0.0), (-0.25, -0.25), (0.26, -0.24))
OBSTACLE_SIZES_M = (0.08, 0.16, 0.24, 0.30)
COVERAGE_RADII_M = (0.01, 0.02, 0.03)
SHAPES: tuple[Shape, ...] = ("box", "disk")
OBSTACLE_NAMES = tuple(f"obstacle-{index}" for index in range(len(OBSTACLE_CENTERS)))
COMPATIBILITY_FIELDS = (
    "schema_version",
    "benchmark_revision",
    "table_bounds",
    "movable_size_m",
    "obstacle_centers",
    "obstacle_sizes_m",
    "shapes",
    "target",
    "repetitions",
    "warmup",
    "seed",
    "max_attempts_per_layout",
    "max_iterations",
    "timing_scope",
)


@dataclass(frozen=True)
class Scenario:
    """One physical obstacle shape and scale."""

    shape: Shape
    obstacle_size_m: float

    @property
    def name(self) -> str:
        return f"{self.shape}-{self.obstacle_size_m:.2f}"


@dataclass(frozen=True)
class CoverageMeasurement:
    """Coverage statistics for one method and physical scenario."""

    method: Method
    shape: Shape
    obstacle_size_m: float
    free_space_fraction: float
    sample_count_per_repetition: int
    metric_samples: int
    metric_probes: int
    repetitions: int
    coverage_1cm: float
    coverage_1cm_iqr: float
    coverage_2cm: float
    coverage_2cm_iqr: float
    coverage_3cm: float
    coverage_3cm_iqr: float
    median_probe_distance_m: float
    median_probe_distance_m_iqr: float
    energy_distance_to_reference: float
    energy_distance_to_reference_iqr: float


def scenarios() -> tuple[Scenario, ...]:
    """Return the balanced box- and disk-obstacle scale sweep."""
    return tuple(Scenario(shape, size) for shape in SHAPES for size in OBSTACLE_SIZES_M)


def _source_revision(repository_root: Path) -> str | None:
    revision = collect_source_revision(repository_root)
    if revision is not None:
        return revision
    source_files = (
        repository_root / "robolab/scene_gen/llm_scene_gen/predicates.py",
        repository_root / "robolab/scene_gen/llm_scene_gen/spatial_solver.py",
    )
    if not all(path.is_file() for path in source_files):
        return None
    digest = hashlib.sha256()
    for path in source_files:
        digest.update(path.relative_to(repository_root).as_posix().encode())
        digest.update(path.read_bytes())
    return f"tree.{digest.hexdigest()[:12]}"


def _position_valid(xy: tuple[float, float], scenario: Scenario) -> bool:
    xmin, xmax, ymin, ymax = TABLE_BOUNDS
    radius = MOVABLE_SIZE_M / 2
    x, y = xy
    if x - radius < xmin or x + radius > xmax or y - radius < ymin or y + radius > ymax:
        return False
    separation = (scenario.obstacle_size_m + MOVABLE_SIZE_M) / 2
    for obstacle_x, obstacle_y in OBSTACLE_CENTERS:
        if scenario.shape == "disk":
            collision_free = math.hypot(x - obstacle_x, y - obstacle_y) >= separation
        else:
            collision_free = abs(x - obstacle_x) >= separation or abs(y - obstacle_y) >= separation
        if not collision_free:
            return False
    return True


def _sample_uniform(seed: int) -> tuple[float, float]:
    generator = random.Random(seed)
    radius = MOVABLE_SIZE_M / 2
    xmin, xmax, ymin, ymax = TABLE_BOUNDS
    return (
        generator.uniform(xmin + radius, xmax - radius),
        generator.uniform(ymin + radius, ymax - radius),
    )


def _build_arena_assets(scenario: Scenario) -> list[PlaceableAsset]:
    from isaaclab_arena.relations.benchmark.synthetic_benchmark import BenchmarkAsset
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    xmin, xmax, ymin, ymax = TABLE_BOUNDS
    table = BenchmarkAsset(
        "table",
        AxisAlignedBoundingBox(min_point=(xmin, ymin, -0.1), max_point=(xmax, ymax, 0.0)),
    )
    table.add_relation(IsAnchor())
    table.set_initial_pose(Pose.identity())
    assets: list[PlaceableAsset] = [table]
    half_obstacle = scenario.obstacle_size_m / 2
    for name, (x, y) in zip(OBSTACLE_NAMES, OBSTACLE_CENTERS, strict=True):
        obstacle = BenchmarkAsset(
            name,
            AxisAlignedBoundingBox(
                min_point=(-half_obstacle, -half_obstacle, -0.05),
                max_point=(half_obstacle, half_obstacle, 0.05),
            ),
        )
        obstacle.add_relation(IsAnchor())
        obstacle.set_initial_pose(Pose(position_xyz=(x, y, 0.05), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        assets.append(obstacle)
    half_movable = MOVABLE_SIZE_M / 2
    movable = BenchmarkAsset(
        "movable",
        AxisAlignedBoundingBox(
            min_point=(-half_movable, -half_movable, -0.05),
            max_point=(half_movable, half_movable, 0.05),
        ),
    )
    movable.add_relation(On(table, clearance_m=0.0, edge_margin_m=0.0))
    assets.append(movable)
    return assets


def _sample_arena(scenario: Scenario, seed: int, count: int, max_iterations: int) -> list[tuple[float, float]]:
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

    assets = _build_arena_assets(scenario)
    movable = assets[-1]
    placer = ObjectPlacer(
        ObjectPlacerParams(
            solver_params=RelationSolverParams(
                max_iters=max_iterations,
                clearance_m=0.0,
                verbose=False,
                save_position_history=False,
            ),
            placement_seed=seed,
            max_placement_attempts=1,
            random_yaw_init=False,
            apply_positions_to_objects=False,
            verbose=False,
        )
    )
    return [
        (float(result.positions[movable][0]), float(result.positions[movable][1]))
        for result in placer.place(assets, num_envs=count)
        if result.success
    ]


def _load_robolab(robolab_root: Path):
    sys.path.insert(0, str(robolab_root.resolve()))
    from robolab.scene_gen.llm_scene_gen.predicates import ObjectState, PlaceOnBasePredicate
    from robolab.scene_gen.llm_scene_gen.spatial_solver import SpatialSolver

    return ObjectState, PlaceOnBasePredicate, SpatialSolver


def _sample_robolab(
    scenario: Scenario,
    seed: int,
    max_iterations: int,
    robolab_api,
) -> tuple[float, float] | None:
    ObjectState, PlaceOnBasePredicate, SpatialSolver = robolab_api
    random.seed(seed)
    np.random.seed(seed % (2**32))
    states = {
        name: ObjectState(name=name, x=x, y=y, yaw=0.0, is_placed=True)
        for name, (x, y) in zip(OBSTACLE_NAMES, OBSTACLE_CENTERS, strict=True)
    }
    states["movable"] = ObjectState(
        name="movable",
        predicates=[PlaceOnBasePredicate("movable", yaw=0.0)],
    )
    dimensions = {name: (scenario.obstacle_size_m, scenario.obstacle_size_m, 0.1) for name in OBSTACLE_NAMES}
    dimensions["movable"] = (MOVABLE_SIZE_M, MOVABLE_SIZE_M, 0.1)
    solver = SpatialSolver(table_bounds=TABLE_BOUNDS, collision_margin=0.0)
    success, _ = solver.solve(
        states,
        dimensions,
        max_iterations=max_iterations,
        fixed_objects=list(OBSTACLE_NAMES),
        allow_relaxation=False,
    )
    state = states["movable"]
    if not success or state.x is None or state.y is None:
        return None
    return float(state.x), float(state.y)


def _generate_sample(
    method: Method,
    scenario: Scenario,
    seed: int,
    target: int,
    max_attempts_per_layout: int,
    max_iterations: int,
    robolab_api=None,
) -> dict:
    positions: dict[tuple[float, float], None] = {}
    attempts = accepted = 0
    maximum_attempts = target * max_attempts_per_layout
    start = time.perf_counter()
    while attempts < maximum_attempts and len(positions) < target:
        if method == "arena":
            batch_size = min(target - len(positions), maximum_attempts - attempts)
            candidates = _sample_arena(scenario, seed + attempts, batch_size, max_iterations)
            attempts += batch_size
        elif method == "robolab":
            candidate = _sample_robolab(scenario, seed + attempts, max_iterations, robolab_api)
            candidates = [] if candidate is None else [candidate]
            attempts += 1
        else:
            candidates = [_sample_uniform(seed + attempts)]
            attempts += 1
        for position in candidates:
            if _position_valid(position, scenario):
                accepted += 1
                positions.setdefault((round(position[0], 7), round(position[1], 7)), None)
    return {
        "seed": seed,
        "target": target,
        "target_reached": len(positions) == target,
        "elapsed_ms": (time.perf_counter() - start) * 1e3,
        "attempts": attempts,
        "accepted": accepted,
        "positions": [list(position) for position in positions],
    }


def generate(args: argparse.Namespace) -> int:
    """Generate exact-valid positions across the shape/scale sweep."""
    robolab_api = _load_robolab(args.robolab_root) if args.method == "robolab" else None
    stride = args.target * args.max_attempts_per_layout
    samples = []
    for scenario_index, scenario in enumerate(scenarios()):
        scenario_seed = args.seed + scenario_index * stride * (args.repetitions + 1)
        for warmup_index in range(args.warmup):
            _generate_sample(
                args.method,
                scenario,
                scenario_seed - (warmup_index + 1) * stride,
                min(args.target, 32),
                args.max_attempts_per_layout,
                args.max_iterations,
                robolab_api,
            )
        for repeat in range(args.repetitions):
            samples.append({
                "scenario": asdict(scenario),
                **_generate_sample(
                    args.method,
                    scenario,
                    scenario_seed + repeat * stride,
                    args.target,
                    args.max_attempts_per_layout,
                    args.max_iterations,
                    robolab_api,
                ),
            })
    benchmark_root = Path(__file__).resolve().parents[2]
    method_root = args.robolab_root if args.method == "robolab" else benchmark_root
    payload = {
        "schema_version": 1,
        "method": args.method,
        "source_revision": _source_revision(method_root),
        "benchmark_revision": _source_revision(benchmark_root),
        "table_bounds": TABLE_BOUNDS,
        "movable_size_m": MOVABLE_SIZE_M,
        "obstacle_centers": OBSTACLE_CENTERS,
        "obstacle_sizes_m": OBSTACLE_SIZES_M,
        "shapes": list(SHAPES),
        "target": args.target,
        "repetitions": args.repetitions,
        "warmup": args.warmup,
        "seed": args.seed,
        "max_attempts_per_layout": args.max_attempts_per_layout,
        "max_iterations": args.max_iterations,
        "host": platform.node(),
        "processor": platform.processor(),
        "timing_scope": "complete-exact-k-generation",
        "solver_config": {
            "collision_representation": {
                "arena": "axis-aligned bounding boxes for both physical shape families",
                "robolab": "max-XY-radius circles for both physical shape families",
                "reference": "exact physical shape declared by each scenario",
            }[args.method],
            "execution": "native-batch" if args.method == "arena" else "serial",
        },
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "samples": samples,
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{args.method}: completed {sum(sample['target_reached'] for sample in samples)}/{len(samples)} targets")
    return 0


def _free_space_fraction(scenario: Scenario, resolution: int = 512) -> float:
    radius = MOVABLE_SIZE_M / 2
    xmin, xmax, ymin, ymax = TABLE_BOUNDS
    xs = np.linspace(xmin + radius, xmax - radius, resolution)
    ys = np.linspace(ymin + radius, ymax - radius, resolution)
    valid = sum(_position_valid((float(x), float(y)), scenario) for x in xs for y in ys)
    return valid / (resolution * resolution)


def _probe_metrics(
    positions: np.ndarray,
    reference: np.ndarray,
    seed: int,
    count: int,
    self_reference: bool = False,
) -> tuple[float, float, float, float, float]:
    assert count <= len(positions) and count <= len(reference)
    generator = np.random.default_rng(seed % (2**32))
    positions = positions[generator.permutation(len(positions))[:count]]
    reference = reference[generator.permutation(len(reference))[:count]]
    split = count // 2
    probes = reference[:split]
    if self_reference:
        samples = reference[split : 2 * split]
    else:
        samples = positions[:split]
    distances = cKDTree(samples).query(probes, k=1)[0]
    coverage_1cm, coverage_2cm, coverage_3cm = (float(np.mean(distances <= radius)) for radius in COVERAGE_RADII_M)
    pair_count = min(20_000, split * 10)
    left = generator.integers(0, split, size=pair_count)
    right = generator.integers(0, split, size=pair_count)
    cross = np.linalg.norm(samples[left] - probes[right], axis=1).mean()
    within_samples = np.linalg.norm(samples[left] - samples[right], axis=1).mean()
    within_probes = np.linalg.norm(probes[left] - probes[right], axis=1).mean()
    energy = math.sqrt(max(0.0, 2 * cross - within_samples - within_probes))
    return coverage_1cm, coverage_2cm, coverage_3cm, float(np.median(distances)), energy


def _median_iqr(values: list[float]) -> tuple[float, float]:
    return float(np.median(values)), float(np.percentile(values, 75) - np.percentile(values, 25))


def analyze(args: argparse.Namespace) -> int:
    """Compare native solver coverage against exact physical references."""
    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    runs = {run["method"]: run for run in loaded}
    if set(runs) != {"arena", "robolab", "reference"}:
        raise ValueError("analysis requires Arena, RoboLab, and reference runs")
    for method, run in runs.items():
        for field in COMPATIBILITY_FIELDS:
            if run.get(field) != runs["reference"].get(field):
                raise ValueError(f"{method} has incompatible {field}")
    measurements = []
    for scenario_index, scenario in enumerate(scenarios()):
        by_method = {
            method: [sample for sample in run["samples"] if sample["scenario"] == asdict(scenario)]
            for method, run in runs.items()
        }
        count = min(len(sample["positions"]) for samples in by_method.values() for sample in samples)
        for method, samples in by_method.items():
            metrics = []
            for repeat, (sample, reference_sample) in enumerate(zip(samples, by_method["reference"], strict=True)):
                metrics.append(
                    _probe_metrics(
                        np.asarray(sample["positions"], dtype=np.float64),
                        np.asarray(reference_sample["positions"], dtype=np.float64),
                        seed=scenario_index * 1_000_000 + repeat,
                        count=count,
                        self_reference=method == "reference",
                    )
                )
            (
                coverage_1cm,
                coverage_2cm,
                coverage_3cm,
                median_probe_distance,
                energy_distance,
            ) = (_median_iqr(list(values)) for values in zip(*metrics, strict=True))
            measurements.append(
                CoverageMeasurement(
                    method=method,
                    shape=scenario.shape,
                    obstacle_size_m=scenario.obstacle_size_m,
                    free_space_fraction=_free_space_fraction(scenario),
                    sample_count_per_repetition=count,
                    metric_samples=count // 2,
                    metric_probes=count // 2,
                    repetitions=len(metrics),
                    coverage_1cm=coverage_1cm[0],
                    coverage_1cm_iqr=coverage_1cm[1],
                    coverage_2cm=coverage_2cm[0],
                    coverage_2cm_iqr=coverage_2cm[1],
                    coverage_3cm=coverage_3cm[0],
                    coverage_3cm_iqr=coverage_3cm[1],
                    median_probe_distance_m=median_probe_distance[0],
                    median_probe_distance_m_iqr=median_probe_distance[1],
                    energy_distance_to_reference=energy_distance[0],
                    energy_distance_to_reference_iqr=energy_distance[1],
                )
            )
    payload = {
        "schema_version": 1,
        "coverage_definition": "fraction of exact-valid reference probes within radius of an equal-count solver sample",
        "coverage_radii_m": COVERAGE_RADII_M,
        "metric_samples_and_probes_per_repetition": count // 2,
        "inputs": [
            {
                "path": str(path),
                "method": run["method"],
                "source_revision": run["source_revision"],
                "benchmark_revision": run["benchmark_revision"],
                "solver_config": run["solver_config"],
            }
            for path, run in zip(args.inputs, loaded, strict=True)
        ],
        "measurements": [asdict(measurement) for measurement in measurements],
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for measurement in measurements:
        if measurement.method != "reference":
            print(
                f"{measurement.method:8} {measurement.shape:4} size={measurement.obstacle_size_m:.2f} "
                f"free={measurement.free_space_fraction:.3f} coverage@2cm={measurement.coverage_2cm:.3f} "
                f"energy={measurement.energy_distance_to_reference:.5f}"
            )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--method", choices=("arena", "robolab", "reference"), required=True)
    generate_parser.add_argument("--robolab-root", type=Path)
    generate_parser.add_argument("--target", type=int, default=5000)
    generate_parser.add_argument("--repetitions", type=int, default=3)
    generate_parser.add_argument("--warmup", type=int, default=1)
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument("--max-attempts-per-layout", type=int, default=1000)
    generate_parser.add_argument("--max-iterations", type=int, default=600)
    generate_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("inputs", type=Path, nargs=3)
    analyze_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        if args.method == "robolab" and args.robolab_root is None:
            parser.error("--robolab-root is required for RoboLab")
        if (
            args.target <= 1
            or args.repetitions <= 0
            or args.warmup < 0
            or args.max_attempts_per_layout <= 0
            or args.max_iterations <= 0
        ):
            parser.error("generation counts must be positive, target > 1, and warmup non-negative")
    return args


def main() -> int:
    args = _parse_args()
    return generate(args) if args.command == "generate" else analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
