# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run a controlled factorial benchmark of background-aware layout generation."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import numpy as np
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from isaaclab_arena.relations.benchmark.provenance import collect_source_revision
from isaaclab_arena_examples.relations.background_batch_scaling_benchmark import BACKGROUND_SIZE_M, _source_revision
from isaaclab_arena_examples.relations.background_practical_generation_benchmark import (
    _initial_layout,
    _layout_key,
    _make_counted_robolab_api,
    _runtime_metadata,
    _summarize_diversity,
)
from isaaclab_arena_examples.relations.fixed_iteration_batch_scaling_benchmark import (
    OBJECT_HEIGHT_M,
    OBJECT_SIZE_M,
    TABLE_BOUNDS,
    _parse_ints,
)

Algorithm = Literal["arena", "robolab"]
Topology = Literal["scattered", "corridor"]
Layout = dict[str, tuple[float, float]]
DEFAULT_OBJECT_COUNTS = (5, 10, 20)
SQUARE_EXCLUSION_FRACTION = (BACKGROUND_SIZE_M + OBJECT_SIZE_M) ** 2 / (
    TABLE_BOUNDS[1] - TABLE_BOUNDS[0] - OBJECT_SIZE_M
) ** 2
DEFAULT_EXCLUDED_FRACTIONS = (0.0, 4 * SQUARE_EXCLUSION_FRACTION, 8 * SQUARE_EXCLUSION_FRACTION)
DEFAULT_TOPOLOGIES: tuple[Topology, ...] = ("scattered", "corridor")


@dataclass(frozen=True)
class FactorialObstacle:
    """One fixed axis-aligned background obstacle."""

    x: float
    y: float
    width: float
    depth: float


@dataclass(frozen=True)
class FactorialScene:
    """One deterministic fixed-obstacle scene in the factorial design."""

    topology: Topology
    target_excluded_fraction: float
    scene_index: int
    obstacles: tuple[FactorialObstacle, ...]
    measured_excluded_fraction: float
    center_corridor_width_m: float | None

    @property
    def name(self) -> str:
        percentage = round(self.target_excluded_fraction * 100)
        return f"{self.topology}-excluded-{percentage:02d}-scene-{self.scene_index:02d}"

    @property
    def obstacle_count(self) -> int:
        return len(self.obstacles)

    @property
    def centers(self) -> tuple[tuple[float, float], ...]:
        return tuple((obstacle.x, obstacle.y) for obstacle in self.obstacles)


def _parse_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(","))
    assert values and all(
        0.0 <= item <= 0.45 for item in values
    ), f"expected comma-separated fractions in [0, 0.45], got {value}"
    return values


def _parse_topologies(value: str) -> tuple[Topology, ...]:
    values = tuple(value.split(","))
    assert values and set(values) <= set(
        DEFAULT_TOPOLOGIES
    ), f"expected comma-separated values from {DEFAULT_TOPOLOGIES}, got {value}"
    return values  # type: ignore[return-value]


def _obstacle_count(target_excluded_fraction: float) -> int:
    center_domain_area = (TABLE_BOUNDS[1] - TABLE_BOUNDS[0] - OBJECT_SIZE_M) * (
        TABLE_BOUNDS[3] - TABLE_BOUNDS[2] - OBJECT_SIZE_M
    )
    exclusion_area = (BACKGROUND_SIZE_M + OBJECT_SIZE_M) ** 2
    return round(target_excluded_fraction * center_domain_area / exclusion_area)


def _union_area(rectangles: list[tuple[float, float, float, float]]) -> float:
    if not rectangles:
        return 0.0
    x_edges = sorted({edge for rectangle in rectangles for edge in rectangle[:2]})
    area = 0.0
    for left, right in zip(x_edges, x_edges[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (bottom, top)
            for rect_left, rect_right, bottom, top in rectangles
            if rect_left < right and rect_right > left
        )
        covered_y = 0.0
        if intervals:
            current_bottom, current_top = intervals[0]
            for bottom, top in intervals[1:]:
                if bottom > current_top:
                    covered_y += current_top - current_bottom
                    current_bottom, current_top = bottom, top
                else:
                    current_top = max(current_top, top)
            covered_y += current_top - current_bottom
        area += (right - left) * covered_y
    return area


def _measured_excluded_fraction(obstacles: tuple[FactorialObstacle, ...]) -> float:
    center_margin = OBJECT_SIZE_M / 2
    domain = (
        TABLE_BOUNDS[0] + center_margin,
        TABLE_BOUNDS[1] - center_margin,
        TABLE_BOUNDS[2] + center_margin,
        TABLE_BOUNDS[3] - center_margin,
    )
    rectangles = [
        (
            max(domain[0], obstacle.x - (obstacle.width + OBJECT_SIZE_M) / 2),
            min(domain[1], obstacle.x + (obstacle.width + OBJECT_SIZE_M) / 2),
            max(domain[2], obstacle.y - (obstacle.depth + OBJECT_SIZE_M) / 2),
            min(domain[3], obstacle.y + (obstacle.depth + OBJECT_SIZE_M) / 2),
        )
        for obstacle in obstacles
    ]
    domain_area = (domain[1] - domain[0]) * (domain[3] - domain[2])
    return _union_area(rectangles) / domain_area


def _scattered_obstacles(obstacle_count: int, scene_index: int) -> tuple[FactorialObstacle, ...]:
    if obstacle_count == 0:
        return ()
    generator = random.Random(31_337 + scene_index)
    obstacle_limit = 0.36
    minimum_separation = BACKGROUND_SIZE_M + OBJECT_SIZE_M + 0.005
    centers = []
    attempts = 0
    while len(centers) < obstacle_count and attempts < 20_000:
        attempts += 1
        candidate = (
            generator.uniform(-obstacle_limit, obstacle_limit),
            generator.uniform(-obstacle_limit, obstacle_limit),
        )
        if all(
            abs(candidate[0] - existing[0]) >= minimum_separation
            or abs(candidate[1] - existing[1]) >= minimum_separation
            for existing in centers
        ):
            centers.append(candidate)
    assert len(centers) == obstacle_count, "could not generate separated scattered obstacles"
    return tuple(FactorialObstacle(x, y, BACKGROUND_SIZE_M, BACKGROUND_SIZE_M) for x, y in centers)


def _corridor_obstacles(
    obstacle_count: int,
    target_excluded_fraction: float,
    scene_index: int,
) -> tuple[tuple[FactorialObstacle, ...], float | None]:
    if obstacle_count == 0:
        return (), None
    assert obstacle_count in (4, 8), f"corridor topology supports 4 or 8 obstacles, got {obstacle_count}"
    generator = random.Random(73_000 + scene_index)
    center_passage = generator.uniform(0.08, 0.12)
    center_domain_width = TABLE_BOUNDS[1] - TABLE_BOUNDS[0] - OBJECT_SIZE_M
    obstacle_width = target_excluded_fraction * center_domain_width / 2 - OBJECT_SIZE_M
    assert obstacle_width > 0.0
    column_x = (center_passage + obstacle_width + OBJECT_SIZE_M) / 2
    segments_per_column = obstacle_count // 2
    segment_depth = (TABLE_BOUNDS[3] - TABLE_BOUNDS[2]) / segments_per_column
    obstacles = tuple(
        FactorialObstacle(
            x=x,
            y=TABLE_BOUNDS[2] + (row + 0.5) * segment_depth,
            width=obstacle_width,
            depth=segment_depth,
        )
        for row in range(segments_per_column)
        for x in (-column_x, column_x)
    )
    return obstacles, center_passage


def make_scene(
    topology: Topology,
    target_excluded_fraction: float,
    scene_index: int,
) -> FactorialScene:
    """Create one deterministic scene for a factorial condition."""
    obstacle_count = _obstacle_count(target_excluded_fraction)
    if topology == "scattered":
        obstacles = _scattered_obstacles(obstacle_count, scene_index)
        passage = None
    else:
        obstacles, passage = _corridor_obstacles(obstacle_count, target_excluded_fraction, scene_index)
    return FactorialScene(
        topology=topology,
        target_excluded_fraction=target_excluded_fraction,
        scene_index=scene_index,
        obstacles=obstacles,
        measured_excluded_fraction=_measured_excluded_fraction(obstacles),
        center_corridor_width_m=passage,
    )


def build_scenes(
    excluded_fractions: tuple[float, ...],
    topologies: tuple[Topology, ...],
    scene_instances: int,
) -> tuple[FactorialScene, ...]:
    """Build all deterministic scenes, deduplicating the empty topology condition."""
    scenes = []
    for excluded_fraction in excluded_fractions:
        is_empty = _obstacle_count(excluded_fraction) == 0
        condition_topologies: tuple[Topology, ...] = ("scattered",) if is_empty else topologies
        condition_instances = 1 if is_empty else scene_instances
        for topology in condition_topologies:
            scenes.extend(
                make_scene(topology, excluded_fraction, scene_index) for scene_index in range(condition_instances)
            )
    return tuple(scenes)


def _overlap(
    first_xy: tuple[float, float],
    first_size: tuple[float, float],
    second_xy: tuple[float, float],
    second_size: tuple[float, float],
) -> bool:
    return (
        abs(first_xy[0] - second_xy[0]) < (first_size[0] + second_size[0]) / 2
        and abs(first_xy[1] - second_xy[1]) < (first_size[1] + second_size[1]) / 2
    )


def _valid_factorial_layout(
    layout: Layout,
    num_objects: int,
    scene: FactorialScene,
) -> bool:
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
        if any(
            _overlap(
                xy,
                (OBJECT_SIZE_M, OBJECT_SIZE_M),
                (obstacle.x, obstacle.y),
                (obstacle.width, obstacle.depth),
            )
            for obstacle in scene.obstacles
        ):
            return False
    positions = list(layout.values())
    return not any(
        _overlap(
            first,
            (OBJECT_SIZE_M, OBJECT_SIZE_M),
            second,
            (OBJECT_SIZE_M, OBJECT_SIZE_M),
        )
        for index, first in enumerate(positions)
        for second in positions[index + 1 :]
    )


class _FixedFactorialObstacle:
    def __init__(self, name: str, obstacle: FactorialObstacle) -> None:
        from isaaclab_arena.relations.collision_mode import CollisionMode
        from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
        from isaaclab_arena.utils.pose import Pose

        self.name = name
        self.collision_mode = CollisionMode.BBOX
        self.repair_collision_mesh_non_watertight = False
        self._pose = Pose.identity()
        self._bbox = AxisAlignedBoundingBox(
            min_point=(
                obstacle.x - obstacle.width / 2,
                obstacle.y - obstacle.depth / 2,
                0.0,
            ),
            max_point=(
                obstacle.x + obstacle.width / 2,
                obstacle.y + obstacle.depth / 2,
                OBJECT_HEIGHT_M,
            ),
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


def _sample_arena_factorial_batch(
    scene: FactorialScene,
    num_objects: int,
    seeds: range,
    max_iterations: int,
) -> tuple[list[Layout], int, int, float | None]:
    import torch

    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena_examples.relations.fixed_iteration_batch_scaling_benchmark import _build_arena_workload

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
    collision_objects = [
        _FixedFactorialObstacle(f"background-{index}", obstacle) for index, obstacle in enumerate(scene.obstacles)
    ]
    solved_positions = solver.solve(
        objects,
        positions,
        collision_objects=collision_objects,
    )
    movable_objects = objects[1:]
    layouts = [
        {obj.name: (float(env_positions[obj][0]), float(env_positions[obj][1])) for obj in movable_objects}
        for env_positions in solved_positions
    ]
    assert solver.last_loss_per_env is not None
    native_successes = int((solver.last_loss_per_env < solver.params.convergence_threshold).sum().item())
    peak_memory_mb = float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else None
    return layouts, len(solver.last_loss_history), native_successes, peak_memory_mb


def _sample_robolab_controlled(
    scene: FactorialScene,
    num_objects: int,
    seed: int,
    max_iterations: int,
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
    for index, obstacle in enumerate(scene.obstacles):
        name = f"background-{index}"
        states[name] = ObjectState(
            name=name,
            x=obstacle.x,
            y=obstacle.y,
            yaw=0.0,
            is_placed=True,
        )
        background_names.append(name)
    dimensions = {f"object-{index}": (OBJECT_SIZE_M, OBJECT_SIZE_M, OBJECT_HEIGHT_M) for index in range(num_objects)}
    dimensions.update({
        f"background-{index}": (
            obstacle.width,
            obstacle.depth,
            OBJECT_HEIGHT_M,
        )
        for index, obstacle in enumerate(scene.obstacles)
    })
    solver = SpatialSolver(table_bounds=TABLE_BOUNDS, collision_margin=0.0)
    solver._background_names = set(background_names)
    with contextlib.redirect_stdout(io.StringIO()):
        native_success = solver._optimize_placement(
            states,
            dimensions,
            max_iterations=max_iterations,
            fixed_objects=background_names,
        )
    for index, obstacle in enumerate(scene.obstacles):
        state = states[f"background-{index}"]
        assert (state.x, state.y) == (
            obstacle.x,
            obstacle.y,
        ), "RoboLab moved a fixed background object"
    layout = {
        name: (float(states[name].x), float(states[name].y))
        for name in initial_layout
        if states[name].x is not None and states[name].y is not None
    }
    return layout, solver.collision_checks, native_success


def _generate_repetition(
    algorithm: Algorithm,
    scene: FactorialScene,
    num_objects: int,
    target_layouts: int,
    max_attempts_per_layout: int,
    max_iterations: int,
    seed: int,
    repeat: int,
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
            candidates, work_units, batch_native_successes, peak_memory_mb = _sample_arena_factorial_batch(
                scene, num_objects, seeds, max_iterations
            )
            attempted += batch_size
            native_successes += batch_native_successes
            solver_calls += 1
            work_unit_layouts += work_units * batch_size
        else:
            layout, work_units, native_success = _sample_robolab_controlled(
                scene,
                num_objects,
                seed + attempted,
                max_iterations,
                robolab_api,
            )
            attempted += 1
            candidates = [layout]
            native_successes += int(native_success)
            solver_calls += 1
            work_unit_layouts += work_units
        for layout in candidates:
            if _valid_factorial_layout(layout, num_objects, scene):
                accepted += 1
                unique.setdefault(_layout_key(layout), layout)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1e3
    layouts = list(unique.values())
    measurement = {
        "seed": seed,
        "repeat": repeat,
        "scene": scene.name,
        "scene_index": scene.scene_index,
        "topology": scene.topology,
        "target_excluded_fraction": scene.target_excluded_fraction,
        "measured_excluded_fraction": scene.measured_excluded_fraction,
        "center_corridor_width_m": scene.center_corridor_width_m,
        "background_objects": scene.obstacle_count,
        "num_objects": num_objects,
        "target_layouts": target_layouts,
        "target_reached": len(layouts) == target_layouts,
        "elapsed_ms": elapsed_ms,
        "attempted_layouts": attempted,
        "native_success_rate": native_successes / attempted,
        "shared_valid_rate": accepted / attempted,
        "unique_layouts": len(layouts),
        "solver_calls": solver_calls,
        "unique_valid_layouts_per_second": len(layouts) * 1e3 / elapsed_ms,
        "peak_gpu_memory_mb": peak_memory_mb if algorithm == "arena" else None,
        "diversity": _summarize_diversity(layouts, num_objects),
    }
    work_unit_name = (
        "arena_optimizer_steps_per_attempt" if algorithm == "arena" else "robolab_collision_checks_per_attempt"
    )
    measurement[work_unit_name] = work_unit_layouts / attempted
    return measurement


def generate(args: argparse.Namespace) -> int:
    benchmark_root = Path(__file__).resolve().parents[2]
    scenes = build_scenes(args.excluded_fractions, args.topologies, args.scene_instances)
    robolab_api = _make_counted_robolab_api(args.robolab_root) if args.algorithm == "robolab" else None
    maximum_target = max(args.target_layout_counts)
    seed_stride = maximum_target * args.max_attempts_per_layout
    for num_objects in args.object_counts:
        warmed_counts = set()
        for scene in scenes:
            if scene.obstacle_count in warmed_counts:
                continue
            _generate_repetition(
                args.algorithm,
                scene,
                num_objects,
                min(16, maximum_target),
                args.max_attempts_per_layout,
                args.max_iterations,
                args.seed - seed_stride,
                repeat=-1,
                robolab_api=robolab_api,
            )
            warmed_counts.add(scene.obstacle_count)
    measurements = []
    for num_objects in args.object_counts:
        for scene in scenes:
            for target_layouts in args.target_layout_counts:
                for repeat in range(args.repetitions):
                    measurement = _generate_repetition(
                        args.algorithm,
                        scene,
                        num_objects,
                        target_layouts,
                        args.max_attempts_per_layout,
                        args.max_iterations,
                        args.seed + repeat * seed_stride,
                        repeat,
                        robolab_api,
                    )
                    measurements.append(measurement)
                    print(
                        f"{args.algorithm:7} n={num_objects:2} {scene.name:34} K={target_layouts:4} "
                        f"valid/s={measurement['unique_valid_layouts_per_second']:8.2f} "
                        f"accepted={measurement['shared_valid_rate']:.3f}"
                    )
    source_root = args.robolab_root if args.algorithm == "robolab" else benchmark_root
    payload = {
        "schema_version": 1,
        "algorithm": args.algorithm,
        "source_revision": _source_revision(source_root),
        "benchmark_revision": collect_source_revision(benchmark_root),
        "runtime": _runtime_metadata(),
        "object_counts": args.object_counts,
        "excluded_fractions": args.excluded_fractions,
        "topologies": args.topologies,
        "scene_instances": args.scene_instances,
        "scenes": [asdict(scene) | {"name": scene.name} for scene in scenes],
        "target_layout_counts": args.target_layout_counts,
        "repetitions": args.repetitions,
        "max_attempts_per_layout": args.max_attempts_per_layout,
        "max_iterations": args.max_iterations,
        "seed": args.seed,
        "seed_stride": seed_stride,
        "table_bounds": TABLE_BOUNDS,
        "movable_size_m": OBJECT_SIZE_M,
        "scattered_background_size_m": BACKGROUND_SIZE_M,
        "collision_representation": "matched-aabb",
        "collision_margin_m": 0.0,
        "constraints": "table-containment-and-pairwise-non-overlap",
        "initialization": "shared-seeded-uniform-center-positions",
        "early_stopping": True,
        "robolab_adaptive_policy": False,
        "robolab_relaxation": False,
        "corridor_width_distribution_m": "deterministic draws from Uniform(0.08, 0.12)",
        "timing_scope": "complete-bounded-exact-k-generation",
        "measurements": measurements,
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


def _bootstrap_scene_median(
    values_by_scene: dict[int, list[float]],
    seed: int,
    bootstrap_samples: int,
) -> tuple[float, float, float]:
    scene_indices = sorted(values_by_scene)
    scene_medians = [statistics.median(values_by_scene[index]) for index in scene_indices]
    estimate = statistics.median(scene_medians)
    generator = np.random.default_rng(seed)
    bootstrap_estimates = []
    for _ in range(bootstrap_samples):
        sampled_scenes = generator.choice(scene_indices, size=len(scene_indices), replace=True)
        sampled_scene_medians = []
        for scene_index in sampled_scenes:
            values = values_by_scene[int(scene_index)]
            sampled_values = generator.choice(values, size=len(values), replace=True)
            sampled_scene_medians.append(float(np.median(sampled_values)))
        bootstrap_estimates.append(float(np.median(sampled_scene_medians)))
    return (
        estimate,
        float(np.percentile(bootstrap_estimates, 2.5)),
        float(np.percentile(bootstrap_estimates, 97.5)),
    )


def _paired_speedups_by_scene(
    condition: dict[str, list[dict]],
) -> tuple[bool, dict[int, list[float]]]:
    paired = {
        algorithm: {(measurement["scene_index"], measurement["repeat"]): measurement for measurement in measurements}
        for algorithm, measurements in condition.items()
    }
    if paired["arena"].keys() != paired["robolab"].keys():
        raise ValueError("paired scene/repetition keys differ")
    all_complete = all(
        measurement["target_reached"] for measurements in condition.values() for measurement in measurements
    )
    speedups_by_scene: dict[int, list[float]] = {}
    if all_complete:
        for key, arena_measurement in paired["arena"].items():
            robolab_measurement = paired["robolab"][key]
            speedups_by_scene.setdefault(key[0], []).append(
                arena_measurement["unique_valid_layouts_per_second"]
                / robolab_measurement["unique_valid_layouts_per_second"]
            )
    return all_complete, speedups_by_scene


def analyze(args: argparse.Namespace) -> int:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    keyed = {run["algorithm"]: run for run in runs}
    if set(keyed) != {"arena", "robolab"}:
        raise ValueError("analysis requires one Arena and one RoboLab run")
    compatibility = (
        "schema_version",
        "benchmark_revision",
        "runtime",
        "object_counts",
        "excluded_fractions",
        "topologies",
        "scene_instances",
        "scenes",
        "target_layout_counts",
        "repetitions",
        "max_attempts_per_layout",
        "max_iterations",
        "seed",
        "seed_stride",
        "table_bounds",
        "movable_size_m",
        "scattered_background_size_m",
        "collision_representation",
        "collision_margin_m",
        "constraints",
        "initialization",
        "early_stopping",
        "robolab_adaptive_policy",
        "robolab_relaxation",
        "corridor_width_distribution_m",
        "timing_scope",
    )
    for field in compatibility:
        if keyed["arena"][field] != keyed["robolab"][field]:
            raise ValueError(f"incompatible {field}")
    rows = []
    for num_objects in keyed["arena"]["object_counts"]:
        for excluded_fraction in keyed["arena"]["excluded_fractions"]:
            condition_topologies = (
                ("scattered",) if _obstacle_count(excluded_fraction) == 0 else keyed["arena"]["topologies"]
            )
            for topology in condition_topologies:
                for target_layouts in keyed["arena"]["target_layout_counts"]:
                    condition = {
                        algorithm: [
                            measurement
                            for measurement in run["measurements"]
                            if measurement["num_objects"] == num_objects
                            and measurement["target_excluded_fraction"] == excluded_fraction
                            and measurement["topology"] == topology
                            and measurement["target_layouts"] == target_layouts
                        ]
                        for algorithm, run in keyed.items()
                    }
                    condition_scene_count = len({measurement["scene_index"] for measurement in condition["arena"]})
                    expected = condition_scene_count * keyed["arena"]["repetitions"]
                    if any(len(measurements) != expected for measurements in condition.values()):
                        raise ValueError(
                            f"incomplete condition n={num_objects}, excluded={excluded_fraction}, "
                            f"topology={topology}, K={target_layouts}"
                        )
                    all_complete, speedups_by_scene = _paired_speedups_by_scene(condition)
                    row = {
                        "num_objects": num_objects,
                        "target_excluded_fraction": excluded_fraction,
                        "measured_excluded_fraction": statistics.median(
                            measurement["measured_excluded_fraction"] for measurement in condition["arena"]
                        ),
                        "topology": topology,
                        "target_layouts": target_layouts,
                        "paired_exact_k_complete": all_complete,
                    }
                    for algorithm, measurements in condition.items():
                        for metric in (
                            "unique_valid_layouts_per_second",
                            "native_success_rate",
                            "shared_valid_rate",
                            "peak_gpu_memory_mb",
                        ):
                            values = [
                                measurement[metric] for measurement in measurements if measurement[metric] is not None
                            ]
                            if values:
                                row[f"{algorithm}_{metric}_median"] = statistics.median(values)
                        for metric in (
                            "unique_1cm_rate",
                            "mean_marginal_grid_coverage",
                            "mean_marginal_normalized_entropy",
                            "mean_intra_layout_pair_distance_m",
                        ):
                            row[f"{algorithm}_{metric}_median"] = statistics.median(
                                measurement["diversity"][metric] for measurement in measurements
                            )
                    if all_complete:
                        estimate, lower, upper = _bootstrap_scene_median(
                            speedups_by_scene,
                            seed=args.bootstrap_seed,
                            bootstrap_samples=args.bootstrap_samples,
                        )
                        row.update({
                            "arena_speedup_scene_median": estimate,
                            "arena_speedup_bootstrap_ci95_low": lower,
                            "arena_speedup_bootstrap_ci95_high": upper,
                        })
                    else:
                        row.update({
                            "arena_speedup_scene_median": None,
                            "arena_speedup_bootstrap_ci95_low": None,
                            "arena_speedup_bootstrap_ci95_high": None,
                        })
                    rows.append(row)
    output = {
        "schema_version": 1,
        "comparison": "controlled factorial background-aware exact-K generation",
        "bootstrap": {
            "unit": "scene instance with within-scene repetition resampling",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "rows": rows,
        "inputs": [str(path) for path in args.inputs],
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in rows:
        speedup = row["arena_speedup_scene_median"]
        speedup_text = f"{speedup:6.2f}x" if speedup is not None else "censored"
        print(
            f"n={row['num_objects']:2} excluded={row['target_excluded_fraction']:.0%} "
            f"{row['topology']:9} K={row['target_layouts']:4}: {speedup_text}"
        )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--algorithm", choices=("arena", "robolab"), required=True)
    generate_parser.add_argument("--robolab-root", type=Path)
    generate_parser.add_argument("--object-counts", type=_parse_ints, default=DEFAULT_OBJECT_COUNTS)
    generate_parser.add_argument(
        "--excluded-fractions",
        type=_parse_floats,
        default=DEFAULT_EXCLUDED_FRACTIONS,
    )
    generate_parser.add_argument(
        "--topologies",
        type=_parse_topologies,
        default=DEFAULT_TOPOLOGIES,
    )
    generate_parser.add_argument("--scene-instances", type=int, default=5)
    generate_parser.add_argument("--target-layout-counts", type=_parse_ints, default=(128, 512, 2048))
    generate_parser.add_argument("--repetitions", type=int, default=5)
    generate_parser.add_argument("--max-attempts-per-layout", type=int, default=4)
    generate_parser.add_argument("--max-iterations", type=int, default=600)
    generate_parser.add_argument("--seed", type=int, default=20_000)
    generate_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("inputs", nargs=2, type=Path)
    analyze_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    analyze_parser.add_argument("--bootstrap-seed", type=int, default=93_117)
    analyze_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        if args.algorithm == "robolab" and args.robolab_root is None:
            parser.error("--robolab-root is required for RoboLab")
        if (
            min(
                args.scene_instances,
                args.repetitions,
                args.max_attempts_per_layout,
                args.max_iterations,
            )
            <= 0
        ):
            parser.error("generation counts must be positive")
        if any(
            not any(
                math.isclose(fraction, supported, rel_tol=0.0, abs_tol=1e-9) for supported in DEFAULT_EXCLUDED_FRACTIONS
            )
            for fraction in args.excluded_fractions
        ):
            parser.error(f"excluded fractions must use the supported realized levels {DEFAULT_EXCLUDED_FRACTIONS}")
    elif min(args.bootstrap_samples, args.bootstrap_seed) <= 0:
        parser.error("bootstrap configuration must be positive")
    return args


def main() -> int:
    args = _parse_args()
    return generate(args) if args.command == "generate" else analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
