# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Arena and RoboLab solver APIs without placement orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import numpy as np
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

InitMode = Literal["random", "clustered", "overlap"]
Method = Literal["arena", "robolab"]
RelationSide = Literal["positive_x", "negative_x", "positive_y", "negative_y"]
TableBounds = tuple[float, float, float, float]
OBJECT_HEIGHT_M = 0.1


@dataclass(frozen=True)
class RelationEdge:
    """One shared directional relation expressed as an edge-to-edge gap."""

    child: str
    parent: str
    side: RelationSide
    distance_m: float = 0.03
    tolerance_m: float = 0.01


@dataclass(frozen=True)
class Scenario:
    """One progressively more difficult controlled tabletop workload."""

    stage: str
    name: str
    num_objects: int
    object_size_m: float
    init_mode: InitMode
    relations: tuple[RelationEdge, ...] = ()


@dataclass(frozen=True)
class Measurement:
    """One direct-solver measurement."""

    method: Method
    scenario: str
    stage: str
    num_objects: int
    object_size_m: float
    init_mode: InitMode
    batch_size: int
    repeat: int
    elapsed_ms: float
    layouts_per_second: float
    shared_valid_layouts_per_second: float
    native_success_rate: float
    shared_geometry_valid_rate: float
    shared_relation_valid_rate: float
    shared_valid_rate: float
    iterations: int | None


def progressive_scenarios() -> tuple[Scenario, ...]:
    """Return the fixed simple-to-difficult experiment ladder."""
    scenarios = [Scenario("simple", "one-object", 1, 0.12, "random")]
    scenarios.extend(
        Scenario("object-scaling", f"objects-{count}", count, 0.12, "random") for count in (2, 4, 6, 8, 10)
    )
    scenarios.extend(
        Scenario("density-scaling", f"density-{size:.2f}", 10, size, "random") for size in (0.08, 0.16, 0.20, 0.24)
    )
    scenarios.extend(
        Scenario("initialization", f"initialization-{mode}", 10, 0.12, mode) for mode in ("clustered", "overlap")
    )
    return tuple(scenarios)


def relational_scenarios() -> tuple[Scenario, ...]:
    """Return shared directional-relation workloads supported by both solvers."""
    pair = (RelationEdge("object-1", "object-0", "positive_x"),)
    chain = tuple(RelationEdge(f"object-{index}", f"object-{index - 1}", "positive_x") for index in range(1, 5))
    star = (
        RelationEdge("object-1", "object-0", "positive_x"),
        RelationEdge("object-2", "object-0", "negative_x"),
        RelationEdge("object-3", "object-0", "positive_y"),
        RelationEdge("object-4", "object-0", "negative_y"),
    )
    dual_chain = tuple(
        RelationEdge(f"object-{offset + index}", f"object-{offset + index - 1}", "positive_x")
        for offset in (0, 5)
        for index in range(1, 5)
    )
    return (
        Scenario("relations", "directional-pair", 2, 0.08, "random", pair),
        Scenario("relations", "directional-chain-5", 5, 0.08, "random", chain),
        Scenario("relations", "directional-star-5", 5, 0.08, "random", star),
        Scenario("relations", "directional-dual-chain-10", 10, 0.08, "random", dual_chain),
    )


def initial_xy(
    scenario: Scenario,
    batch_size: int,
    seed: int,
    table_bounds: TableBounds,
) -> tuple[dict[str, tuple[float, float]], ...]:
    """Generate deterministic initial XY positions shared by both methods."""
    xmin, xmax, ymin, ymax = table_bounds
    half_size = scenario.object_size_m / 2.0
    if scenario.init_mode == "random":
        x_bounds = (xmin + half_size, xmax - half_size)
        y_bounds = (ymin + half_size, ymax - half_size)
    elif scenario.init_mode == "clustered":
        cluster_half_width = min(xmax - xmin, ymax - ymin) * 0.1
        x_bounds = y_bounds = (-cluster_half_width, cluster_half_width)
    else:
        x_bounds = y_bounds = (0.0, 0.0)

    layouts = []
    for env_index in range(batch_size):
        generator = random.Random(seed + env_index)
        layout = {
            f"object-{index}": (
                generator.uniform(*x_bounds),
                generator.uniform(*y_bounds),
            )
            for index in range(scenario.num_objects)
        }
        for relation in scenario.relations:
            child_x, child_y = layout[relation.child]
            parent_x, parent_y = layout[relation.parent]
            horizontal = relation.side in ("positive_x", "negative_x")
            layout[relation.child] = (
                child_x if horizontal else parent_x,
                parent_y if horizontal else child_y,
            )
        layouts.append(layout)
    return tuple(layouts)


def _valid_geometry(
    poses: dict[str, tuple[float, float]],
    scenario: Scenario,
    table_bounds: TableBounds,
) -> bool:
    xmin, xmax, ymin, ymax = table_bounds
    half_size = scenario.object_size_m / 2.0
    if len(poses) != scenario.num_objects:
        return False
    if any(
        x - half_size < xmin or x + half_size > xmax or y - half_size < ymin or y + half_size > ymax
        for x, y in poses.values()
    ):
        return False
    items = list(poses.items())
    for index, (_, (x1, y1)) in enumerate(items):
        for _, (x2, y2) in items[index + 1 :]:
            if abs(x1 - x2) < scenario.object_size_m and abs(y1 - y2) < scenario.object_size_m:
                return False
    return True


def _relations_valid(poses: dict[str, tuple[float, float]], scenario: Scenario) -> bool:
    """Check shared directional relations using exact square footprints."""
    for relation in scenario.relations:
        if relation.child not in poses or relation.parent not in poses:
            return False
        child_x, child_y = poses[relation.child]
        parent_x, parent_y = poses[relation.parent]
        center_delta = scenario.object_size_m + relation.distance_m
        axis_delta = {
            "positive_x": child_x - parent_x,
            "negative_x": parent_x - child_x,
            "positive_y": child_y - parent_y,
            "negative_y": parent_y - child_y,
        }[relation.side]
        if abs(axis_delta - center_delta) > relation.tolerance_m:
            return False
    return True


def _valid_layout(
    poses: dict[str, tuple[float, float]],
    scenario: Scenario,
    table_bounds: TableBounds,
) -> bool:
    return _valid_geometry(poses, scenario, table_bounds) and _relations_valid(poses, scenario)


def _shared_validity_rates(
    layouts: list[dict[str, tuple[float, float]]],
    scenario: Scenario,
    table_bounds: TableBounds,
    batch_size: int,
) -> tuple[float, float, float]:
    """Return geometry, relation, and combined validity rates."""
    geometry_valid_count = 0
    relation_valid_count = 0
    shared_valid_count = 0
    for layout in layouts:
        geometry_valid_count += _valid_geometry(layout, scenario, table_bounds)
        relation_valid_count += _relations_valid(layout, scenario)
        shared_valid_count += _valid_layout(layout, scenario, table_bounds)

    return (
        geometry_valid_count / batch_size,
        relation_valid_count / batch_size,
        shared_valid_count / batch_size,
    )


def _git_output(repository_root: Path, *args: str) -> str:
    """Return stdout from one Git command."""
    return subprocess.run(
        ["git", "-C", str(repository_root), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout


def _source_revision(path: Path) -> str | None:
    try:
        commit = _git_output(path, "rev-parse", "HEAD").strip()
        if not _git_output(path, "status", "--porcelain"):
            return commit
        digest = hashlib.sha256(_git_output(path, "diff", "--binary", "HEAD").encode())
        untracked = _git_output(path, "ls-files", "--others", "--exclude-standard").splitlines()
        for relative_path in sorted(untracked):
            digest.update(relative_path.encode())
            digest.update((path / relative_path).read_bytes())
        return f"{commit}+dirty.{digest.hexdigest()[:12]}"
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _seed_robolab(seed: int) -> None:
    """Seed global generators used internally by RoboLab's solver."""
    random.seed(seed)
    np.random.seed(seed % (2**32))


def _measure_arena(
    scenario: Scenario,
    batch_size: int,
    repeat: int,
    seed: int,
    max_iterations: int,
    table_bounds: TableBounds,
) -> Measurement:
    from isaaclab_arena.relations.benchmark.layout_generation import build_arena_controlled_scene
    from isaaclab_arena.relations.benchmark.timing import time_call
    from isaaclab_arena.relations.bounding_box_helpers import assign_variants_for_envs, build_per_env_bounding_boxes
    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import NextTo, Side

    sizes = {
        f"object-{index}": (scenario.object_size_m, scenario.object_size_m) for index in range(scenario.num_objects)
    }
    objects = build_arena_controlled_scene(table_bounds, sizes)
    by_name = {obj.name: obj for obj in objects}
    arena_sides = {
        "positive_x": Side.POSITIVE_X,
        "negative_x": Side.NEGATIVE_X,
        "positive_y": Side.POSITIVE_Y,
        "negative_y": Side.NEGATIVE_Y,
    }
    for relation in scenario.relations:
        by_name[relation.child].add_relation(
            NextTo(
                by_name[relation.parent],
                side=arena_sides[relation.side],
                distance_m=relation.distance_m,
                tolerance_m=relation.tolerance_m,
            )
        )
    assign_variants_for_envs(objects, batch_size, placement_seed=seed)
    env_bboxes = build_per_env_bounding_boxes(objects, batch_size).get_bounding_boxes_for_solver_candidates(1)
    xy_layouts = initial_xy(scenario, batch_size, seed, table_bounds)
    table = objects[0]
    positions = [
        {obj: (0.0, 0.0, 0.0) if obj is table else (*xy_layout[obj.name], OBJECT_HEIGHT_M / 2.0) for obj in objects}
        for xy_layout in xy_layouts
    ]
    solver = RelationSolver(
        RelationSolverParams(
            max_iters=max_iterations,
            clearance_m=0.0,
            verbose=False,
            save_position_history=False,
        )
    )
    elapsed_ms, solved = time_call(lambda: solver.solve(objects, positions, env_bboxes=env_bboxes), time.perf_counter)
    solved_xy = [{obj.name: tuple(result[obj][:2]) for obj in objects[1:]} for result in solved]
    assert solver.last_loss_per_env is not None
    native_success = float((solver.last_loss_per_env <= solver.params.convergence_threshold).sum().item() / batch_size)
    geometry_valid, relation_valid, shared_valid = _shared_validity_rates(solved_xy, scenario, table_bounds, batch_size)
    return Measurement(
        method="arena",
        scenario=scenario.name,
        stage=scenario.stage,
        num_objects=scenario.num_objects,
        object_size_m=scenario.object_size_m,
        init_mode=scenario.init_mode,
        batch_size=batch_size,
        repeat=repeat,
        elapsed_ms=elapsed_ms,
        layouts_per_second=batch_size * 1e3 / elapsed_ms,
        shared_valid_layouts_per_second=batch_size * shared_valid * 1e3 / elapsed_ms,
        native_success_rate=native_success,
        shared_geometry_valid_rate=geometry_valid,
        shared_relation_valid_rate=relation_valid,
        shared_valid_rate=shared_valid,
        iterations=len(solver.last_loss_history),
    )


def _load_robolab(robolab_root: Path):
    sys.path.insert(0, str(robolab_root.resolve()))
    from robolab.scene_gen.llm_scene_gen.predicates import (
        ObjectState,
        PlaceOnBasePredicate,
        PredicateType,
        RelativePositionPredicate,
    )
    from robolab.scene_gen.llm_scene_gen.spatial_solver import SpatialSolver

    return ObjectState, PlaceOnBasePredicate, PredicateType, RelativePositionPredicate, SpatialSolver


def _measure_robolab(
    scenario: Scenario,
    batch_size: int,
    repeat: int,
    seed: int,
    max_iterations: int,
    table_bounds: TableBounds,
    robolab_api,
) -> Measurement:
    ObjectState, PlaceOnBasePredicate, PredicateType, RelativePositionPredicate, SpatialSolver = robolab_api
    layouts = initial_xy(scenario, batch_size, seed, table_bounds)
    dimensions = {
        f"object-{index}": (scenario.object_size_m, scenario.object_size_m, OBJECT_HEIGHT_M)
        for index in range(scenario.num_objects)
    }
    robolab_directions = {
        "positive_x": PredicateType.FRONT_OF,
        "negative_x": PredicateType.BACK_OF,
        "positive_y": PredicateType.LEFT_OF,
        "negative_y": PredicateType.RIGHT_OF,
    }
    jobs = []
    for layout in layouts:
        states = {
            name: ObjectState(
                name=name,
                predicates=[PlaceOnBasePredicate(name, x=xy[0], y=xy[1], yaw=0.0)],
            )
            for name, xy in layout.items()
        }
        for relation in scenario.relations:
            states[relation.child].predicates.append(
                RelativePositionPredicate(
                    relation.child,
                    relation.parent,
                    robolab_directions[relation.side],
                    distance=scenario.object_size_m + relation.distance_m,
                )
            )
        jobs.append((SpatialSolver(table_bounds=table_bounds, collision_margin=0.0), states))
    native_successes = []
    start = time.perf_counter()
    for env_index, (solver, states) in enumerate(jobs):
        _seed_robolab(seed + env_index)
        success, _ = solver.solve(
            states,
            dimensions,
            max_iterations=max_iterations,
            allow_relaxation=False,
        )
        native_successes.append(success)
    elapsed_ms = (time.perf_counter() - start) * 1e3
    solved_xy = [
        {
            name: (float(state.x), float(state.y))
            for name, state in states.items()
            if state.x is not None and state.y is not None
        }
        for _, states in jobs
    ]
    geometry_valid, relation_valid, shared_valid = _shared_validity_rates(solved_xy, scenario, table_bounds, batch_size)
    return Measurement(
        method="robolab",
        scenario=scenario.name,
        stage=scenario.stage,
        num_objects=scenario.num_objects,
        object_size_m=scenario.object_size_m,
        init_mode=scenario.init_mode,
        batch_size=batch_size,
        repeat=repeat,
        elapsed_ms=elapsed_ms,
        layouts_per_second=batch_size * 1e3 / elapsed_ms,
        shared_valid_layouts_per_second=batch_size * shared_valid * 1e3 / elapsed_ms,
        native_success_rate=sum(native_successes) / batch_size,
        shared_geometry_valid_rate=geometry_valid,
        shared_relation_valid_rate=relation_valid,
        shared_valid_rate=shared_valid,
        iterations=None,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("arena", "robolab"), required=True)
    parser.add_argument("--suite", choices=("progressive", "relational"), default="progressive")
    parser.add_argument("--robolab-root", type=Path)
    parser.add_argument("--batch-size", type=int, action="append", default=[])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=600)
    parser.add_argument("--table-bounds", type=float, nargs=4, default=(-0.5, 0.5, -0.5, 0.5))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat <= 0 or args.warmup < 0 or args.max_iterations <= 0:
        parser.error("--repeat and --max-iterations must be positive; --warmup must be non-negative")
    if args.method == "robolab" and args.robolab_root is None:
        parser.error("--robolab-root is required for RoboLab")
    if any(size <= 0 for size in args.batch_size):
        parser.error("--batch-size must be positive")
    return args


def _print_summary(measurements: list[Measurement]) -> None:
    print("| Stage | Scenario | K | Layouts/s | Shared-valid/s | Geometry | Relations | Shared valid | Iterations |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    keys = {(item.stage, item.scenario, item.batch_size) for item in measurements}
    for stage, scenario, batch_size in sorted(keys):
        rows = [
            item
            for item in measurements
            if (item.stage, item.scenario, item.batch_size) == (stage, scenario, batch_size)
        ]
        iterations = [item.iterations for item in rows if item.iterations is not None]
        print(
            f"| {stage} | {scenario} | {batch_size} | "
            f"{statistics.median(item.layouts_per_second for item in rows):.3f} | "
            f"{statistics.median(item.shared_valid_layouts_per_second for item in rows):.3f} | "
            f"{statistics.median(item.shared_geometry_valid_rate for item in rows):.3f} | "
            f"{statistics.median(item.shared_relation_valid_rate for item in rows):.3f} | "
            f"{statistics.median(item.shared_valid_rate for item in rows):.3f} | "
            f"{statistics.median(iterations) if iterations else 'N/A'} |"
        )


def main() -> int:
    args = _parse_args()
    method: Method = args.method
    table_bounds = tuple(args.table_bounds)
    batch_sizes = tuple(args.batch_size or (1, 32, 128, 256))
    scenarios = relational_scenarios() if args.suite == "relational" else progressive_scenarios()
    robolab_api = _load_robolab(args.robolab_root) if method == "robolab" else None
    measurements = []
    for scenario_index, scenario in enumerate(scenarios):
        for batch_size in batch_sizes:
            for repeat in range(-args.warmup, args.repeat):
                sample_seed = args.seed + scenario_index * 1_000_000 + batch_size * 1_000 + repeat * 10_000
                if method == "arena":
                    measurement = _measure_arena(
                        scenario,
                        batch_size,
                        repeat,
                        sample_seed,
                        args.max_iterations,
                        table_bounds,
                    )
                else:
                    measurement = _measure_robolab(
                        scenario,
                        batch_size,
                        repeat,
                        sample_seed,
                        args.max_iterations,
                        table_bounds,
                        robolab_api,
                    )
                if repeat >= 0:
                    measurements.append(measurement)
    benchmark_root = Path(__file__).resolve().parents[2]
    source_root = args.robolab_root if method == "robolab" else benchmark_root
    output = {
        "schema_version": 1,
        "method": method,
        "suite": args.suite,
        "source_commit": _source_revision(source_root),
        "benchmark_revision": _source_revision(benchmark_root),
        "host": platform.node(),
        "processor": platform.processor(),
        "table_bounds": table_bounds,
        "batch_sizes": batch_sizes,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "seed": args.seed,
        "max_iterations": args.max_iterations,
        "collision_model": "aabb" if method == "arena" else "max-xy-radius-circle",
        "timing_scope": "RelationSolver.solve" if method == "arena" else "SpatialSolver.solve",
        "relation_semantics": "directional edge-to-edge gap with shared external tolerance",
        "measurements": [asdict(measurement) for measurement in measurements],
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _print_summary(measurements)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
