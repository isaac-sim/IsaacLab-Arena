# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare Arena and RoboLab while holding collision representation fixed."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import numpy as np
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from isaaclab_arena_examples.relations.collision_space_coverage_benchmark import (
    COVERAGE_RADII_M,
    MOVABLE_SIZE_M,
    OBSTACLE_CENTERS,
    OBSTACLE_SIZES_M,
    TABLE_BOUNDS,
    Scenario,
    _build_arena_assets,
    _free_space_fraction,
    _position_valid,
    _probe_metrics,
    _sample_uniform,
    _source_revision,
)

Algorithm = Literal["arena", "robolab", "reference"]
Representation = Literal["aabb", "circle"]
REPRESENTATIONS: tuple[Representation, ...] = ("aabb", "circle")
REFERENCE_SEED_OFFSET = 1_000_000_000_000
"""Disjoint seed stream used only for independent exact-valid probes."""
EXPECTED_RUNS = {
    ("arena", "aabb"),
    ("arena", "circle"),
    ("robolab", "aabb"),
    ("robolab", "circle"),
    ("reference", "both"),
}
COMPATIBILITY_FIELDS = (
    "schema_version",
    "benchmark_revision",
    "target",
    "repetitions",
    "seed",
    "max_attempts_per_layout",
    "max_iterations",
    "table_bounds",
    "movable_size_m",
    "obstacle_centers",
    "obstacle_sizes_m",
    "initialization",
    "reference_seed_offset",
)


@dataclass(frozen=True)
class MatchedCoverage:
    """Coverage for one algorithm under one shared collision representation."""

    algorithm: Algorithm
    representation: Representation
    obstacle_size_m: float
    free_space_fraction: float
    samples: int
    probes: int
    repetitions: int
    coverage_1cm: float
    coverage_1cm_iqr: float
    coverage_2cm: float
    coverage_2cm_iqr: float
    coverage_3cm: float
    coverage_3cm_iqr: float
    median_probe_distance_m: float
    energy_distance_to_reference: float


def _shape(representation: Representation) -> Literal["box", "disk"]:
    return "box" if representation == "aabb" else "disk"


def _initial_xy(seed: int) -> tuple[float, float]:
    return _sample_uniform(seed)


def _aabb_overlap(
    first_xy: tuple[float, float],
    second_xy: tuple[float, float],
    first_dims: tuple[float, float, float],
    second_dims: tuple[float, float, float],
    margin: float,
) -> bool:
    """Return whether two axis-aligned footprints overlap."""
    return (
        abs(first_xy[0] - second_xy[0]) < (first_dims[0] + second_dims[0]) / 2 + 2 * margin
        and abs(first_xy[1] - second_xy[1]) < (first_dims[1] + second_dims[1]) / 2 + 2 * margin
    )


def _make_arena_solver(representation: Representation, max_iterations: int):
    import torch

    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import On

    params = RelationSolverParams(
        max_iters=max_iterations,
        clearance_m=0.0,
        verbose=False,
        save_position_history=False,
    )
    if representation == "aabb":
        return RelationSolver(params)

    class CircleRelationSolver(RelationSolver):
        """RelationSolver using a radial no-overlap loss for this ablation."""

        def _compute_no_overlap_loss(self, state, debug: bool = False):
            def footprint_radius(bbox):
                width = bbox.max_point[..., 0] - bbox.min_point[..., 0]
                depth = bbox.max_point[..., 1] - bbox.min_point[..., 1]
                return torch.maximum(width, depth) / 2

            del debug
            loss = torch.zeros(state.batch_size, device=state.device, dtype=torch.float32)
            optimizable = state.optimizable_objects
            fixed = [*state.anchor_objects, *state.collision_objects]
            on_pairs = {
                (id(obj), id(relation.parent))
                for obj in [*optimizable, *state.anchor_objects]
                for relation in obj.get_relations()
                if isinstance(relation, On)
            }
            pair_count = 0
            for subject in optimizable:
                subject_position = state.get_position(subject)
                subject_bbox = state.get_bbox(subject)
                subject_radius = footprint_radius(subject_bbox)
                for obstacle in fixed:
                    if (id(subject), id(obstacle)) in on_pairs:
                        continue
                    obstacle_bbox = state.get_fixed_obstacle_world_bbox(obstacle)
                    obstacle_center = (obstacle_bbox.min_point + obstacle_bbox.max_point) / 2
                    obstacle_radius = footprint_radius(obstacle_bbox)
                    distance = torch.linalg.vector_norm(subject_position[..., :2] - obstacle_center[..., :2], dim=-1)
                    penetration = torch.relu(subject_radius + obstacle_radius + self.params.clearance_m - distance)
                    loss = loss + self._no_collision_strategy.slope * penetration.square()
                    pair_count += 1
            self._last_aabb_no_overlap_pair_count = pair_count
            self._last_mesh_no_overlap_pair_count = 0
            return loss

    return CircleRelationSolver(params)


def _sample_arena(
    representation: Representation,
    obstacle_size_m: float,
    seeds: list[int],
    max_iterations: int,
) -> list[tuple[float, float]]:
    scenario = Scenario(_shape(representation), obstacle_size_m)
    assets = _build_arena_assets(scenario)
    movable = assets[-1]
    initial_positions = []
    for seed in seeds:
        movable_xy = _initial_xy(seed)
        positions = {assets[0]: (0.0, 0.0, 0.0), movable: (*movable_xy, 0.05)}
        positions.update(
            {obstacle: (*center, 0.05) for obstacle, center in zip(assets[1:-1], OBSTACLE_CENTERS, strict=True)}
        )
        initial_positions.append(positions)
    solved = _make_arena_solver(representation, max_iterations).solve(assets, initial_positions)
    return [(float(result[movable][0]), float(result[movable][1])) for result in solved]


def _load_robolab(robolab_root: Path, representation: Representation):
    sys.path.insert(0, str(robolab_root.resolve()))
    from robolab.scene_gen.llm_scene_gen.predicates import ObjectState, PlaceOnBasePredicate
    from robolab.scene_gen.llm_scene_gen.spatial_solver import SpatialSolver

    if representation == "circle":
        return ObjectState, PlaceOnBasePredicate, SpatialSolver

    class AabbSpatialSolver(SpatialSolver):
        """RoboLab pushing algorithm with an AABB collision oracle."""

        def _check_collisions(self, object_states, object_dims):
            collisions = []
            names = list(object_states)
            fixed_objects = getattr(self, "_ablation_fixed_objects", set())
            for index, first_name in enumerate(names):
                first = object_states[first_name]
                if first.x is None or first.y is None:
                    continue
                for second_name in names[index + 1 :]:
                    if first_name in fixed_objects and second_name in fixed_objects:
                        continue
                    second = object_states[second_name]
                    if second.x is None or second.y is None:
                        continue
                    if _aabb_overlap(
                        (first.x, first.y),
                        (second.x, second.y),
                        object_dims[first_name],
                        object_dims[second_name],
                        self.collision_margin,
                    ):
                        collisions.append((first_name, second_name))
            return collisions

        def _move_away_from_fixed(self, movable_state, fixed_state, movable_dims, fixed_dims):
            dx = movable_state.x - fixed_state.x
            dy = movable_state.y - fixed_state.y
            separation_x = (movable_dims[0] + fixed_dims[0]) / 2 + self.collision_margin
            separation_y = (movable_dims[1] + fixed_dims[1]) / 2 + self.collision_margin
            penetration_x = separation_x - abs(dx)
            penetration_y = separation_y - abs(dy)
            extra_buffer = 0.05
            if penetration_x <= penetration_y:
                direction = 1.0 if dx >= 0 else -1.0
                movable_state.x = fixed_state.x + direction * (separation_x + extra_buffer)
            else:
                direction = 1.0 if dy >= 0 else -1.0
                movable_state.y = fixed_state.y + direction * (separation_y + extra_buffer)
            half_width = movable_dims[0] / 2 + self.collision_margin
            half_depth = movable_dims[1] / 2 + self.collision_margin
            movable_state.x = np.clip(movable_state.x, self.min_x + half_width, self.max_x - half_width)
            movable_state.y = np.clip(movable_state.y, self.min_y + half_depth, self.max_y - half_depth)

    return ObjectState, PlaceOnBasePredicate, AabbSpatialSolver


def _sample_robolab(
    representation: Representation,
    obstacle_size_m: float,
    seeds: list[int],
    max_iterations: int,
    robolab_api,
) -> list[tuple[float, float]]:
    ObjectState, PlaceOnBasePredicate, SpatialSolver = robolab_api
    dimensions = {
        f"obstacle-{index}": (obstacle_size_m, obstacle_size_m, 0.1) for index in range(len(OBSTACLE_CENTERS))
    }
    dimensions["movable"] = (MOVABLE_SIZE_M, MOVABLE_SIZE_M, 0.1)
    fixed_names = list(dimensions)[:-1]
    results = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        x, y = _initial_xy(seed)
        states = {
            name: ObjectState(name=name, x=center[0], y=center[1], yaw=0.0, is_placed=True)
            for name, center in zip(fixed_names, OBSTACLE_CENTERS, strict=True)
        }
        states["movable"] = ObjectState(
            name="movable",
            predicates=[PlaceOnBasePredicate("movable", x=x, y=y, yaw=0.0)],
        )
        solver = SpatialSolver(table_bounds=TABLE_BOUNDS, collision_margin=0.0)
        setattr(solver, "_ablation_fixed_objects", set(fixed_names))
        with contextlib.redirect_stdout(io.StringIO()):
            success, _ = solver.solve(
                states,
                dimensions,
                max_iterations=max_iterations,
                fixed_objects=fixed_names,
                allow_relaxation=False,
            )
        movable = states["movable"]
        if success and movable.x is not None and movable.y is not None:
            results.append((float(movable.x), float(movable.y)))
    return results


def _sample_reference(representation: Representation, obstacle_size_m: float, seeds: list[int]):
    scenario = Scenario(_shape(representation), obstacle_size_m)
    positions = (_initial_xy(seed) for seed in seeds)
    return [position for position in positions if _position_valid(position, scenario)]


def _generate_sample(
    algorithm: Algorithm,
    representation: Representation,
    obstacle_size_m: float,
    seed: int,
    target: int,
    max_attempts_per_layout: int,
    max_iterations: int,
    robolab_api=None,
) -> dict:
    scenario = Scenario(_shape(representation), obstacle_size_m)

    positions: dict[tuple[float, float], None] = {}
    attempts = 0
    maximum_attempts = target * max_attempts_per_layout
    start = time.perf_counter()
    while attempts < maximum_attempts and len(positions) < target:
        batch_size = min(target - len(positions), maximum_attempts - attempts)
        seeds = list(range(seed + attempts, seed + attempts + batch_size))
        if algorithm == "arena":
            candidates = _sample_arena(representation, obstacle_size_m, seeds, max_iterations)
        elif algorithm == "robolab":
            candidates = _sample_robolab(representation, obstacle_size_m, seeds, max_iterations, robolab_api)
        else:
            candidates = _sample_reference(representation, obstacle_size_m, seeds)
        attempts += batch_size
        for position in candidates:
            if _position_valid(position, scenario):
                positions.setdefault((round(position[0], 7), round(position[1], 7)), None)
    return {
        "representation": representation,
        "obstacle_size_m": obstacle_size_m,
        "seed": seed,
        "target": target,
        "target_reached": len(positions) == target,
        "attempts": attempts,
        "elapsed_ms": (time.perf_counter() - start) * 1e3,
        "positions": [list(position) for position in positions],
    }


def generate(args: argparse.Namespace) -> int:
    representations = REPRESENTATIONS if args.algorithm == "reference" else (args.representation,)
    robolab_apis = (
        {representation: _load_robolab(args.robolab_root, representation) for representation in representations}
        if args.algorithm == "robolab"
        else {}
    )
    stride = args.target * args.max_attempts_per_layout
    samples = []
    for representation_index, representation in enumerate(representations):
        for size_index, obstacle_size_m in enumerate(OBSTACLE_SIZES_M):
            scenario_index = representation_index * len(OBSTACLE_SIZES_M) + size_index
            scenario_seed = args.seed + scenario_index * stride * (args.repetitions + 1)
            if args.algorithm == "reference":
                scenario_seed += REFERENCE_SEED_OFFSET
            for repeat in range(args.repetitions):
                samples.append(
                    _generate_sample(
                        args.algorithm,
                        representation,
                        obstacle_size_m,
                        scenario_seed + repeat * stride,
                        args.target,
                        args.max_attempts_per_layout,
                        args.max_iterations,
                        robolab_apis.get(representation),
                    )
                )
    benchmark_root = Path(__file__).resolve().parents[2]
    source_root = args.robolab_root if args.algorithm == "robolab" else benchmark_root
    payload = {
        "schema_version": 1,
        "algorithm": args.algorithm,
        "representation": args.representation if args.algorithm != "reference" else "both",
        "source_revision": _source_revision(source_root),
        "benchmark_revision": _source_revision(benchmark_root),
        "target": args.target,
        "repetitions": args.repetitions,
        "seed": args.seed,
        "max_attempts_per_layout": args.max_attempts_per_layout,
        "max_iterations": args.max_iterations,
        "table_bounds": TABLE_BOUNDS,
        "movable_size_m": MOVABLE_SIZE_M,
        "obstacle_centers": OBSTACLE_CENTERS,
        "obstacle_sizes_m": OBSTACLE_SIZES_M,
        "initialization": "identical deterministic uniform XY by candidate seed",
        "reference_seed_offset": REFERENCE_SEED_OFFSET,
        "samples": samples,
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{args.algorithm}-{payload['representation']}: {sum(s['target_reached'] for s in samples)}/{len(samples)}")
    return 0


def _median_iqr(values) -> tuple[float, float]:
    return float(np.median(values)), float(np.percentile(values, 75) - np.percentile(values, 25))


def analyze(args: argparse.Namespace) -> int:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    keyed = {(run["algorithm"], run["representation"]): run for run in runs}
    if set(keyed) != EXPECTED_RUNS:
        raise ValueError(f"expected factorial inputs {EXPECTED_RUNS}, got {set(keyed)}")
    reference = keyed[("reference", "both")]
    for run in runs:
        for field in COMPATIBILITY_FIELDS:
            if run[field] != reference[field]:
                raise ValueError(f"{run['algorithm']}-{run['representation']} has incompatible {field}")

    measurements = []
    metrics_by_cell = {}
    for representation in REPRESENTATIONS:
        reference_samples = [sample for sample in reference["samples"] if sample["representation"] == representation]
        for obstacle_size_m in OBSTACLE_SIZES_M:
            matched_reference = [sample for sample in reference_samples if sample["obstacle_size_m"] == obstacle_size_m]
            for algorithm in ("arena", "robolab", "reference"):
                if algorithm == "reference":
                    samples = matched_reference
                else:
                    samples = [
                        sample
                        for sample in keyed[(algorithm, representation)]["samples"]
                        if sample["obstacle_size_m"] == obstacle_size_m
                    ]
                count = min(
                    len(sample["positions"]) for collection in (samples, matched_reference) for sample in collection
                )
                metrics = [
                    _probe_metrics(
                        np.asarray(sample["positions"]),
                        np.asarray(reference_sample["positions"]),
                        seed=repeat + int(obstacle_size_m * 1000) + (0 if representation == "aabb" else 10_000),
                        count=count,
                        self_reference=algorithm == "reference",
                    )
                    for repeat, (sample, reference_sample) in enumerate(zip(samples, matched_reference, strict=True))
                ]
                metrics_by_cell[(algorithm, representation, obstacle_size_m)] = metrics
                stats = [_median_iqr(values) for values in zip(*metrics, strict=True)]
                measurements.append(
                    MatchedCoverage(
                        algorithm=algorithm,
                        representation=representation,
                        obstacle_size_m=obstacle_size_m,
                        free_space_fraction=_free_space_fraction(Scenario(_shape(representation), obstacle_size_m)),
                        samples=count // 2,
                        probes=count // 2,
                        repetitions=len(metrics),
                        coverage_1cm=stats[0][0],
                        coverage_1cm_iqr=stats[0][1],
                        coverage_2cm=stats[1][0],
                        coverage_2cm_iqr=stats[1][1],
                        coverage_3cm=stats[2][0],
                        coverage_3cm_iqr=stats[2][1],
                        median_probe_distance_m=stats[3][0],
                        energy_distance_to_reference=stats[4][0],
                    )
                )
    effects = []
    for representation in REPRESENTATIONS:
        for obstacle_size_m in OBSTACLE_SIZES_M:
            arena_metrics = metrics_by_cell[("arena", representation, obstacle_size_m)]
            robolab_metrics = metrics_by_cell[("robolab", representation, obstacle_size_m)]
            coverage_1cm = _median_iqr(
                [arena[0] - robolab[0] for arena, robolab in zip(arena_metrics, robolab_metrics, strict=True)]
            )
            coverage_2cm = _median_iqr(
                [arena[1] - robolab[1] for arena, robolab in zip(arena_metrics, robolab_metrics, strict=True)]
            )
            coverage_3cm = _median_iqr(
                [arena[2] - robolab[2] for arena, robolab in zip(arena_metrics, robolab_metrics, strict=True)]
            )
            effects.append({
                "representation": representation,
                "obstacle_size_m": obstacle_size_m,
                "arena_minus_robolab_coverage_1cm": coverage_1cm[0],
                "arena_minus_robolab_coverage_1cm_iqr": coverage_1cm[1],
                "arena_minus_robolab_coverage_2cm": coverage_2cm[0],
                "arena_minus_robolab_coverage_2cm_iqr": coverage_2cm[1],
                "arena_minus_robolab_coverage_3cm": coverage_3cm[0],
                "arena_minus_robolab_coverage_3cm_iqr": coverage_3cm[1],
            })
    output = {
        "schema_version": 1,
        "design": "2 algorithms x 2 matched collision representations",
        "coverage_radii_m": COVERAGE_RADII_M,
        "measurements": [asdict(measurement) for measurement in measurements],
        "algorithm_effects": effects,
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for effect in effects:
        print(
            f"{effect['representation']:6} size={effect['obstacle_size_m']:.2f} "
            f"Arena-RoboLab coverage@2cm={effect['arena_minus_robolab_coverage_2cm']:+.3f} "
            f"(IQR {effect['arena_minus_robolab_coverage_2cm_iqr']:.3f})"
        )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--algorithm", choices=("arena", "robolab", "reference"), required=True)
    generate_parser.add_argument("--representation", choices=REPRESENTATIONS)
    generate_parser.add_argument("--robolab-root", type=Path)
    generate_parser.add_argument("--target", type=int, default=5000)
    generate_parser.add_argument("--repetitions", type=int, default=3)
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument("--max-attempts-per-layout", type=int, default=1000)
    generate_parser.add_argument("--max-iterations", type=int, default=600)
    generate_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("inputs", nargs=5, type=Path)
    analyze_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        if args.algorithm != "reference" and args.representation is None:
            parser.error("--representation is required for solver runs")
        if args.algorithm == "robolab" and args.robolab_root is None:
            parser.error("--robolab-root is required for RoboLab")
    return args


def main() -> int:
    args = _parse_args()
    return generate(args) if args.command == "generate" else analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
