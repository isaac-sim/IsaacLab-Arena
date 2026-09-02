# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare actual solver sampling around fixed real-asset obstacles."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import numpy as np
import platform
import random
import statistics
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from isaaclab_arena.relations.benchmark.provenance import collect_source_revision
from isaaclab_arena_examples.relations.real_asset_collision_region_benchmark import (
    _feasible_centers,
    _file_sha256,
    _mesh_footprint,
    _mesh_sha256,
    _parse_floats,
    _parse_strings,
    _rotate_mesh,
    _source_revision,
)

TABLE_ASSET = "procedural_table"
TABLE_BOUNDS = (-0.4, 0.4, -0.75, 0.75)
DEFAULT_PLACEABLE_ASSETS = (
    "cracker_box",
    "tomato_soup_can",
    "banana_ycb_robolab",
    "spoon_handal_robolab",
    "sweet_potato",
)
DEFAULT_OBSTACLE_ASSETS = (
    "cracker_box",
    "sugar_box",
    "sugar_box_ycb_robolab",
    "tomato_soup_can",
    "mustard_bottle",
    "power_drill",
    "sweet_potato",
    "jug",
    "beer_bottle",
    "lemon_01_fruits_veggies_robolab",
    "orange_01_fruits_veggies_robolab",
    "banana_ycb_robolab",
    "dry_erase_marker_ycb_robolab",
    "spoon_handal_robolab",
    "salad_tongs_handal_robolab",
    "scissors_ycb_robolab",
    "spring_clamp_ycb_robolab",
)
DEFAULT_TARGET_RATIOS = (0.0, 0.08, 0.16, 0.24)
DEFAULT_YAWS_DEG = (0.0, 45.0, 90.0)
SHAPE_GROUPS = ("compact", "elongated", "irregular")
METHODS = ("arena_mesh", "arena_aabb", "robolab_circle")
METHOD_LABELS = {
    "arena_mesh": "Arena mesh",
    "arena_aabb": "Arena AABB",
    "robolab_circle": "RoboLab max-XY-radius circle",
}
METHOD_COLORS = {
    "arena_mesh": "tab:blue",
    "arena_aabb": "tab:orange",
    "robolab_circle": "tab:red",
}
ELONGATED_ASPECT_RATIO = 1.9
IRREGULAR_FILL_RATIO = 0.65


@dataclass(frozen=True)
class ShapeDescriptor:
    """Canonical mesh-footprint descriptors used for predeclared grouping."""

    asset: str
    pca_aspect_ratio: float
    aabb_fill_ratio: float
    shape_group: str


@dataclass(frozen=True)
class ObstacleInstance:
    """One fixed real object in an obstacle scene."""

    asset: str
    x: float
    y: float
    yaw_deg: float


@dataclass(frozen=True)
class SamplingMeasurement:
    """Sampling metrics for one paired method workload."""

    scene_id: str
    obstacle_shape_group: str
    target_obstacle_ratio: float
    realized_mesh_obstacle_ratio: float
    obstacle_set_index: int
    method: str
    placeable_asset: str
    placeable_yaw_deg: float
    proposal_seed: int
    proposal_count: int
    computed_proposal_count: int
    terminal_batch_overshoot: int
    native_success_count: int
    shared_mesh_valid_count: int
    unique_shared_valid_count: int
    target_samples: int
    target_reached: bool
    elapsed_ms: float
    mean_nearest_distance_cm: float
    p90_nearest_distance_cm: float
    coverage_1cm: float
    coverage_2cm: float
    coverage_3cm: float


def _grid_shape(pitch_m: float) -> tuple[int, int]:
    xmin, xmax, ymin, ymax = TABLE_BOUNDS
    return round((ymax - ymin) / pitch_m), round((xmax - xmin) / pitch_m)


def _shape_descriptor(asset: str, footprint: np.ndarray) -> ShapeDescriptor:
    """Classify one canonical real-mesh footprint using fixed criteria."""
    coordinates = np.column_stack(np.nonzero(footprint)).astype(float)
    assert len(coordinates) > 0, f"{asset} has an empty mesh footprint"
    if len(coordinates) == 1:
        aspect_ratio = 1.0
    else:
        covariance = np.atleast_2d(np.cov(coordinates, rowvar=False))
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), np.finfo(float).eps)
        aspect_ratio = math.sqrt(float(eigenvalues[-1] / eigenvalues[0]))
    rows, columns = np.nonzero(footprint)
    bounding_area = (rows.max() - rows.min() + 1) * (columns.max() - columns.min() + 1)
    fill_ratio = float(footprint.sum() / bounding_area)
    if fill_ratio <= IRREGULAR_FILL_RATIO:
        shape_group = "irregular"
    elif aspect_ratio >= ELONGATED_ASPECT_RATIO:
        shape_group = "elongated"
    else:
        shape_group = "compact"
    return ShapeDescriptor(asset, aspect_ratio, fill_ratio, shape_group)


def _footprint_slices(
    mask_shape: tuple[int, int],
    footprint_shape: tuple[int, int],
    center_xy: tuple[float, float],
    pitch_m: float,
) -> tuple[tuple[slice, slice], tuple[slice, slice]] | None:
    """Return matching mask and footprint slices for their in-table overlap."""
    xmin, _xmax, ymin, _ymax = TABLE_BOUNDS
    center_x = round((center_xy[0] - xmin) / pitch_m - 0.5)
    center_y = round((center_xy[1] - ymin) / pitch_m - 0.5)
    half_y, half_x = np.asarray(footprint_shape) // 2
    y_start, x_start = center_y - half_y, center_x - half_x
    y_stop, x_stop = y_start + footprint_shape[0], x_start + footprint_shape[1]
    mask_y_start, mask_x_start = max(0, y_start), max(0, x_start)
    mask_y_stop, mask_x_stop = min(mask_shape[0], y_stop), min(mask_shape[1], x_stop)
    if mask_y_start >= mask_y_stop or mask_x_start >= mask_x_stop:
        return None
    footprint_y_start, footprint_x_start = mask_y_start - y_start, mask_x_start - x_start
    footprint_y_stop = footprint_y_start + mask_y_stop - mask_y_start
    footprint_x_stop = footprint_x_start + mask_x_stop - mask_x_start
    return (
        (slice(mask_y_start, mask_y_stop), slice(mask_x_start, mask_x_stop)),
        (slice(footprint_y_start, footprint_y_stop), slice(footprint_x_start, footprint_x_stop)),
    )


def _stamp(mask: np.ndarray, footprint: np.ndarray, center_xy: tuple[float, float], pitch_m: float) -> bool:
    """Stamp a centered footprint, returning False without mutation if it leaves the table."""
    slices = _footprint_slices(mask.shape, footprint.shape, center_xy, pitch_m)
    if slices is None or mask[slices[0]].shape != footprint.shape:
        return False
    mask[slices[0]] |= footprint
    return True


def _stamp_clipped(mask: np.ndarray, footprint: np.ndarray, center_xy: tuple[float, float], pitch_m: float) -> None:
    """Stamp the portion of a footprint inside the table domain."""
    slices = _footprint_slices(mask.shape, footprint.shape, center_xy, pitch_m)
    if slices is not None:
        mask[slices[0]] |= footprint[slices[1]]


def _candidate_centers(seed: int) -> list[tuple[float, float]]:
    centers = [(float(x), float(y)) for y in np.linspace(-0.70, 0.70, 29) for x in np.linspace(-0.35, 0.35, 15)]
    random.Random(seed).shuffle(centers)
    return centers


def _build_obstacle_set(
    target_ratio: float,
    set_index: int,
    pitch_m: float,
    mesh_footprints: dict[tuple[str, float], np.ndarray],
    obstacle_assets: tuple[str, ...],
    obstacle_yaws_deg: tuple[float, ...],
) -> tuple[list[ObstacleInstance], float]:
    """Build a deterministic mesh-nonoverlapping scene to a projected area target."""
    assert obstacle_assets, "obstacle shape group is empty"
    occupied = np.zeros(_grid_shape(pitch_m), dtype=bool)
    instances: list[ObstacleInstance] = []
    if target_ratio == 0.0:
        return instances, 0.0
    shuffled_assets = list(obstacle_assets)
    random.Random(81_000 + set_index).shuffle(shuffled_assets)
    for index, center in enumerate(_candidate_centers(71_000 + set_index)):
        asset_name = shuffled_assets[index % len(shuffled_assets)]
        yaw_deg = obstacle_yaws_deg[
            random.Random(91_000 + 10_000 * set_index + index).randrange(len(obstacle_yaws_deg))
        ]
        addition = np.zeros_like(occupied)
        if not _stamp(addition, mesh_footprints[(asset_name, yaw_deg)], center, pitch_m):
            continue
        if np.any(occupied & addition):
            continue
        occupied |= addition
        instances.append(ObstacleInstance(asset_name, center[0], center[1], yaw_deg))
        realized_ratio = float(occupied.mean())
        if realized_ratio >= target_ratio:
            return instances, realized_ratio
    raise RuntimeError(f"could not realize obstacle ratio {target_ratio:.3f} for assets {obstacle_assets}")


def _obstacle_mask(
    instances: list[ObstacleInstance],
    mesh_footprints: dict[tuple[str, float], np.ndarray],
    pitch_m: float,
) -> np.ndarray:
    occupied = np.zeros(_grid_shape(pitch_m), dtype=bool)
    for instance in instances:
        assert _stamp(
            occupied,
            mesh_footprints[(instance.asset, instance.yaw_deg)],
            (instance.x, instance.y),
            pitch_m,
        )
    return occupied


def _scene_pairing_metadata(
    shape_group: str,
    target_ratio: float,
    set_index: int,
    instances: list[ObstacleInstance],
    base_seed: int,
) -> dict:
    """Describe the scene and seed shared unchanged by every method."""
    import hashlib

    canonical = json.dumps(
        {
            "shape_group": shape_group,
            "target_ratio": target_ratio,
            "set_index": set_index,
            "instances": [asdict(instance) for instance in instances],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "scene_id": hashlib.sha256(canonical.encode()).hexdigest()[:16],
        "proposal_seed_base": base_seed,
        "paired_methods": list(METHODS),
        "same_initial_xy_seed_stream": True,
    }


def _initial_centers(seed: int, count: int) -> list[tuple[float, float]]:
    """Generate the method-independent initial XY proposal stream."""
    generator = random.Random(seed)
    xmin, xmax, ymin, ymax = TABLE_BOUNDS
    return [(generator.uniform(xmin, xmax), generator.uniform(ymin, ymax)) for _ in range(count)]


def _mask_accepts(mask: np.ndarray, xy: tuple[float, float], pitch_m: float) -> bool:
    xmin, _xmax, ymin, _ymax = TABLE_BOUNDS
    x_index = round((xy[0] - xmin) / pitch_m - 0.5)
    y_index = round((xy[1] - ymin) / pitch_m - 0.5)
    return 0 <= y_index < mask.shape[0] and 0 <= x_index < mask.shape[1] and bool(mask[y_index, x_index])


def _coverage_fraction(
    samples_xy: list[tuple[float, float]],
    feasible_mask: np.ndarray,
    pitch_m: float,
    radius_m: float,
) -> float:
    """Measure the fraction of mesh-feasible grid probes reached by samples."""
    assert radius_m > 0.0
    probe_rows, probe_columns = np.nonzero(feasible_mask)
    if len(probe_rows) == 0 or not samples_xy:
        return 0.0
    xmin, _xmax, ymin, _ymax = TABLE_BOUNDS
    probes = np.column_stack((xmin + (probe_columns + 0.5) * pitch_m, ymin + (probe_rows + 0.5) * pitch_m))
    from scipy.spatial import cKDTree

    distances, _indices = cKDTree(np.asarray(samples_xy)).query(probes, k=1)
    return float(np.mean(distances <= radius_m))


def _nearest_distance_statistics(
    samples_xy: list[tuple[float, float]],
    feasible_mask: np.ndarray,
    pitch_m: float,
) -> tuple[float, float]:
    """Return mean and 90th-percentile distance from feasible probes to samples."""
    assert samples_xy, "nearest-distance metrics require at least one shared-valid sample"
    probe_rows, probe_columns = np.nonzero(feasible_mask)
    assert len(probe_rows) > 0, "nearest-distance metrics require a non-empty feasible region"
    xmin, _xmax, ymin, _ymax = TABLE_BOUNDS
    probes = np.column_stack((xmin + (probe_columns + 0.5) * pitch_m, ymin + (probe_rows + 0.5) * pitch_m))
    from scipy.spatial import cKDTree

    distances, _indices = cKDTree(np.asarray(samples_xy)).query(probes, k=1)
    return 100 * float(np.mean(distances)), 100 * float(np.percentile(distances, 90))


def _mesh_bounds(mesh) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray(mesh.bounds, dtype=float)
    return bounds[0], bounds[1]


def _make_benchmark_asset(name: str, mesh):
    from isaaclab_arena.relations.benchmark.synthetic_benchmark import BenchmarkAsset
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    minimum, maximum = _mesh_bounds(mesh)
    return BenchmarkAsset(
        name,
        AxisAlignedBoundingBox(min_point=tuple(minimum), max_point=tuple(maximum)),
        mesh,
    )


def _sample_arena(
    method: str,
    instances: list[ObstacleInstance],
    placeable_mesh,
    rotated_meshes: dict[tuple[str, float], object],
    initial_xy: list[tuple[float, float]],
    max_iterations: int,
    batch_size: int,
) -> tuple[list[tuple[float, float]], list[bool]]:
    from isaaclab_arena.relations.benchmark.synthetic_benchmark import BenchmarkAsset
    from isaaclab_arena.relations.collision_mode import CollisionMode
    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    assert method in ("arena_mesh", "arena_aabb")
    xmin, xmax, ymin, ymax = TABLE_BOUNDS
    table = BenchmarkAsset(
        "table",
        AxisAlignedBoundingBox(min_point=(xmin, ymin, -0.05), max_point=(xmax, ymax, 0.0)),
    )
    table.add_relation(IsAnchor())
    table.set_initial_pose(Pose.identity())
    movable = _make_benchmark_asset("movable", placeable_mesh)
    movable.add_relation(On(table, clearance_m=0.0, edge_margin_m=0.0))
    movable_min, movable_max = _mesh_bounds(placeable_mesh)
    movable_center = (movable_min + movable_max) / 2
    obstacles = []
    for index, instance in enumerate(instances):
        mesh = rotated_meshes[(instance.asset, instance.yaw_deg)]
        obstacle = _make_benchmark_asset(f"obstacle-{index}", mesh)
        minimum, maximum = _mesh_bounds(mesh)
        center = (minimum + maximum) / 2
        obstacle.set_initial_pose(
            Pose(
                position_xyz=(
                    instance.x - float(center[0]),
                    instance.y - float(center[1]),
                    -float(minimum[2]),
                ),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )
        obstacles.append(obstacle)
    outputs: list[tuple[float, float]] = []
    native_successes: list[bool] = []
    collision_mode = CollisionMode.MESH if method == "arena_mesh" else CollisionMode.BBOX
    for start in range(0, len(initial_xy), batch_size):
        centers = initial_xy[start : start + batch_size]
        positions = [
            {
                table: (0.0, 0.0, 0.0),
                movable: (
                    x - float(movable_center[0]),
                    y - float(movable_center[1]),
                    -float(movable_min[2]),
                ),
            }
            for x, y in centers
        ]
        solver = RelationSolver(
            RelationSolverParams(
                collision_mode=collision_mode,
                max_iters=max_iterations,
                clearance_m=0.0,
                verbose=False,
                save_position_history=False,
            )
        )
        solved = solver.solve([table, movable], positions, collision_objects=obstacles)
        assert solver.last_loss_per_env is not None
        native_successes.extend(
            loss < solver.params.convergence_threshold for loss in solver.last_loss_per_env.tolist()
        )
        outputs.extend(
            (
                float(result[movable][0] + movable_center[0]),
                float(result[movable][1] + movable_center[1]),
            )
            for result in solved
        )
    return outputs, native_successes


def _move_circle_away_from_fixed(solver, movable, fixed, movable_dims, fixed_dims) -> None:
    """Separate a RoboLab circle pair without its fixed-object clearance policy."""
    if movable.x is None or movable.y is None or fixed.x is None or fixed.y is None:
        return
    dx = movable.x - fixed.x
    dy = movable.y - fixed.y
    distance = math.hypot(dx, dy)
    if distance < 0.01:
        dx, dy = random.uniform(-1, 1), random.uniform(-1, 1)
        distance = math.hypot(dx, dy)
    required_separation = max(movable_dims[:2]) / 2 + max(fixed_dims[:2]) / 2 + 2 * solver.collision_margin
    movable.x = fixed.x + dx / distance * required_separation
    movable.y = fixed.y + dy / distance * required_separation
    movable_radius = max(movable_dims[:2]) / 2 + solver.collision_margin
    movable.x = np.clip(movable.x, solver.min_x + movable_radius, solver.max_x - movable_radius)
    movable.y = np.clip(movable.y, solver.min_y + movable_radius, solver.max_y - movable_radius)


def _load_robolab(robolab_root: Path):
    """Load a native circle solver adapter that cannot move fixed objects."""
    sys.path.insert(0, str(robolab_root.resolve()))
    from robolab.scene_gen.llm_scene_gen.predicates import ObjectState
    from robolab.scene_gen.llm_scene_gen.spatial_solver import SpatialSolver

    class FixedSafeSpatialSolver(SpatialSolver):
        def _check_collisions(self, object_states, object_dims):
            collisions = super()._check_collisions(object_states, object_dims)
            fixed = getattr(self, "_benchmark_fixed_objects", set())
            return [(first, second) for first, second in collisions if not (first in fixed and second in fixed)]

        def _check_table_bounds(self, object_states, object_dims):
            fixed = getattr(self, "_benchmark_fixed_objects", set())
            movable_states = {name: state for name, state in object_states.items() if name not in fixed}
            movable_dims = {name: object_dims[name] for name in movable_states}
            return super()._check_table_bounds(movable_states, movable_dims)

        def _move_away_from_fixed(self, movable, fixed, movable_dims, fixed_dims):
            fixed_names = getattr(self, "_benchmark_fixed_objects", set())
            assert movable.name not in fixed_names, "RoboLab attempted to move a fixed obstacle"
            _move_circle_away_from_fixed(self, movable, fixed, movable_dims, fixed_dims)

    return ObjectState, FixedSafeSpatialSolver


def _sample_robolab(
    instances: list[ObstacleInstance],
    placeable_asset: str,
    canonical_meshes: Mapping[str, object],
    initial_xy: list[tuple[float, float]],
    proposal_seed: int,
    max_iterations: int,
    robolab_api,
) -> tuple[list[tuple[float, float]], list[bool]]:
    ObjectState, SpatialSolver = robolab_api
    placeable_min, placeable_max = _mesh_bounds(canonical_meshes[placeable_asset])
    placeable_size = placeable_max - placeable_min
    fixed_names = [f"obstacle-{index}" for index in range(len(instances))]
    fixed_dimensions = {}
    fixed_centers = {}
    for name, instance in zip(fixed_names, instances, strict=True):
        minimum, maximum = _mesh_bounds(canonical_meshes[instance.asset])
        fixed_dimensions[name] = tuple(float(value) for value in maximum - minimum)
        fixed_centers[name] = (instance.x, instance.y)
    outputs = []
    successes = []
    for proposal_index, xy in enumerate(initial_xy):
        seed = proposal_seed + proposal_index
        random.seed(seed)
        np.random.seed(seed % (2**32))
        states = {
            name: ObjectState(name=name, x=center[0], y=center[1], yaw=0.0, is_placed=True)
            for name, center in fixed_centers.items()
        }
        states["movable"] = ObjectState(name="movable", x=xy[0], y=xy[1], yaw=0.0, is_placed=True)
        dimensions = {**fixed_dimensions, "movable": tuple(float(value) for value in placeable_size)}
        solver = SpatialSolver(table_bounds=TABLE_BOUNDS, collision_margin=0.0)
        solver._benchmark_fixed_objects = set(fixed_names)
        before = {name: (states[name].x, states[name].y) for name in fixed_names}
        with contextlib.redirect_stdout(io.StringIO()):
            success = solver._optimize_placement(
                states,
                dimensions,
                max_iterations=max_iterations,
                fixed_objects=fixed_names,
            )
        after = {name: (states[name].x, states[name].y) for name in fixed_names}
        assert after == before, f"RoboLab moved fixed obstacles on proposal {proposal_index}"
        outputs.append((float(states["movable"].x), float(states["movable"].y)))
        successes.append(bool(success))
    return outputs, successes


def _sample_method(
    method: str,
    instances: list[ObstacleInstance],
    placeable_asset: str,
    placeable_mesh,
    canonical_meshes: Mapping[str, object],
    rotated_meshes: dict[tuple[str, float], object],
    feasible_mask: np.ndarray,
    proposal_seed: int,
    target_samples: int,
    max_attempts_per_sample: int,
    max_iterations: int,
    batch_size: int,
    pitch_m: float,
    robolab_api,
) -> dict:
    assert method in METHODS, f"unsupported method: {method}"
    maximum_proposals = target_samples * max_attempts_per_sample
    initial_xy = _initial_centers(proposal_seed, maximum_proposals)
    random.seed(proposal_seed)
    np.random.seed(proposal_seed % (2**32))
    start = time.perf_counter()
    computed_proposal_count = 0
    proposal_count = 0
    native_success_count = 0
    shared_mesh_valid_count = 0
    unique_samples: dict[tuple[float, float], None] = {}
    for proposal_start in range(0, maximum_proposals, batch_size):
        proposal_batch = initial_xy[proposal_start : proposal_start + batch_size]
        if method.startswith("arena_"):
            batch_outputs, batch_successes = _sample_arena(
                method,
                instances,
                placeable_mesh,
                rotated_meshes,
                proposal_batch,
                max_iterations,
                batch_size,
            )
        else:
            batch_outputs, batch_successes = _sample_robolab(
                instances,
                placeable_asset,
                canonical_meshes,
                proposal_batch,
                proposal_seed + proposal_start,
                max_iterations,
                robolab_api,
            )
        computed_proposal_count += len(batch_outputs)
        for xy, native_success in zip(batch_outputs, batch_successes, strict=True):
            proposal_count += 1
            native_success_count += int(native_success)
            if _mask_accepts(feasible_mask, xy, pitch_m):
                shared_mesh_valid_count += 1
                unique_samples[(round(xy[0], 6), round(xy[1], 6))] = None
            if len(unique_samples) >= target_samples:
                break
        if len(unique_samples) >= target_samples:
            break
    elapsed_ms = (time.perf_counter() - start) * 1e3
    target_reached = len(unique_samples) >= target_samples
    terminal_batch_overshoot = computed_proposal_count - proposal_count
    return {
        "proposal_count": proposal_count,
        "computed_proposal_count": computed_proposal_count,
        "terminal_batch_overshoot": terminal_batch_overshoot,
        "native_success_count": native_success_count,
        "shared_mesh_valid_count": shared_mesh_valid_count,
        "unique_shared_valid_count": len(unique_samples),
        "target_reached": target_reached,
        "elapsed_ms": elapsed_ms,
        "samples_xy": list(unique_samples),
    }


def _summaries(measurements: list[SamplingMeasurement]) -> list[dict]:
    rows = []
    for shape_group in SHAPE_GROUPS:
        for target_ratio in sorted({measurement.target_obstacle_ratio for measurement in measurements}):
            for method in METHODS:
                selected = [
                    measurement
                    for measurement in measurements
                    if measurement.obstacle_shape_group == shape_group
                    and measurement.target_obstacle_ratio == target_ratio
                    and measurement.method == method
                ]
                if not selected:
                    continue
                rows.append({
                    "obstacle_shape_group": shape_group,
                    "target_obstacle_ratio": target_ratio,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "realized_mesh_obstacle_ratio_mean": statistics.fmean(
                        item.realized_mesh_obstacle_ratio for item in selected
                    ),
                    "mean_nearest_distance_cm": statistics.fmean(item.mean_nearest_distance_cm for item in selected),
                    "p90_nearest_distance_cm": statistics.fmean(item.p90_nearest_distance_cm for item in selected),
                    "native_success_rate": sum(item.native_success_count for item in selected) / sum(
                        item.proposal_count for item in selected
                    ),
                    "shared_mesh_valid_rate": sum(item.shared_mesh_valid_count for item in selected) / sum(
                        item.proposal_count for item in selected
                    ),
                    "uniqueness_rate": sum(item.unique_shared_valid_count for item in selected) / max(
                        1, sum(item.shared_mesh_valid_count for item in selected)
                    ),
                    "target_reached_fraction": statistics.fmean(item.target_reached for item in selected),
                })
    return rows


def _write_plot(summaries: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(SHAPE_GROUPS), figsize=(11, 3.2), sharey=True)
    for axis, shape_group in zip(axes, SHAPE_GROUPS, strict=True):
        for method in METHODS:
            rows = [row for row in summaries if row["obstacle_shape_group"] == shape_group and row["method"] == method]
            rows.sort(key=lambda row: row["realized_mesh_obstacle_ratio_mean"])
            axis.plot(
                [100 * row["realized_mesh_obstacle_ratio_mean"] for row in rows],
                [row["mean_nearest_distance_cm"] for row in rows],
                marker="o",
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_title(shape_group.capitalize())
        axis.set_xlabel("Projected mesh obstacle area (%)")
        axis.set_xlim(left=0)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean nearest-layout distance (cm)")
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _load_assets(asset_names: tuple[str, ...], pitch_m: float):
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache

    registry = AssetRegistry()
    mesh_cache = WarpMeshAndSphereCache(device="cpu")
    meshes = {}
    metadata = []
    failures = {}
    for asset_name in asset_names:
        print(f"Resolving collision mesh: {asset_name}", flush=True)
        try:
            asset = registry.get_asset_by_name(asset_name)()
            mesh = mesh_cache.get_collision_mesh_or_raise(asset)
            footprint = _mesh_footprint(mesh, pitch_m)
            descriptor = _shape_descriptor(asset_name, footprint)
            minimum, maximum = _mesh_bounds(mesh)
            meshes[asset_name] = mesh
            metadata.append({
                "name": asset_name,
                "usd_path": str(asset.usd_path),
                "usd_sha256": _file_sha256(str(asset.usd_path)),
                "resolved_mesh_sha256": _mesh_sha256(mesh),
                "mesh_vertices": len(mesh.vertices),
                "mesh_faces": len(mesh.faces),
                "mesh_watertight": bool(mesh.is_watertight),
                "xyz_aabb_m": [float(value) for value in maximum - minimum],
                "shape_descriptor": asdict(descriptor),
            })
        except Exception as error:
            failures[asset_name] = f"{type(error).__name__}: {error}"
    if failures:
        print(f"Required asset resolution failures: {json.dumps(failures, sort_keys=True)}", flush=True)
        raise RuntimeError(f"required assets could not be resolved after AppLauncher startup: {failures}")
    return meshes, metadata


def _run(args: argparse.Namespace) -> int:
    unique_assets = tuple(dict.fromkeys((*args.placeable_assets, *args.obstacle_assets)))
    random.seed(args.asset_resolution_seed)
    np.random.seed(args.asset_resolution_seed % (2**32))
    meshes, asset_metadata = _load_assets(unique_assets, args.pitch_m)
    print(f"Resolved {len(meshes)} required collision meshes.", flush=True)
    descriptors = {
        item["name"]: ShapeDescriptor(**item["shape_descriptor"])
        for item in asset_metadata
        if item["name"] in args.obstacle_assets
    }
    grouped_assets = {
        group: tuple(asset for asset in args.obstacle_assets if descriptors[asset].shape_group == group)
        for group in SHAPE_GROUPS
    }
    print(
        "Shape groups: " + ", ".join(f"{group}={len(assets)}" for group, assets in grouped_assets.items()),
        flush=True,
    )
    print(
        "Shape descriptors: "
        + ", ".join(
            f"{name}=({descriptor.pca_aspect_ratio:.2f},{descriptor.aabb_fill_ratio:.2f},{descriptor.shape_group})"
            for name, descriptor in descriptors.items()
        ),
        flush=True,
    )
    empty_groups = [group for group, assets in grouped_assets.items() if not assets]
    if empty_groups:
        raise RuntimeError(f"fixed criteria produced empty obstacle groups {empty_groups}; descriptors={descriptors}")

    required_yaws = tuple(dict.fromkeys((*args.placeable_yaws_deg, *args.obstacle_yaws_deg)))
    rotated_meshes = {
        (asset, yaw_deg): _rotate_mesh(mesh, yaw_deg) for asset, mesh in meshes.items() for yaw_deg in required_yaws
    }
    mesh_footprints = {key: _mesh_footprint(mesh, args.pitch_m) for key, mesh in rotated_meshes.items()}
    print(f"Rasterized {len(mesh_footprints)} fixed-yaw mesh footprints.", flush=True)
    robolab_api = _load_robolab(args.robolab_root)
    obstacle_scenes = []
    measurements = []
    for group_index, shape_group in enumerate(SHAPE_GROUPS):
        for target_index, target_ratio in enumerate(args.target_obstacle_ratios):
            for set_index in range(args.obstacle_set_count):
                scene_index = (
                    set_index if target_ratio == 0.0 else group_index * 100_000 + target_index * 1_000 + set_index
                )
                instances, realized_ratio = _build_obstacle_set(
                    target_ratio,
                    scene_index,
                    args.pitch_m,
                    mesh_footprints,
                    grouped_assets[shape_group],
                    args.obstacle_yaws_deg,
                )
                occupied = _obstacle_mask(instances, mesh_footprints, args.pitch_m)
                pairing = _scene_pairing_metadata(
                    shape_group,
                    target_ratio,
                    set_index,
                    instances,
                    args.seed + scene_index * 10_000,
                )
                obstacle_scenes.append({
                    **pairing,
                    "shape_group": shape_group,
                    "target_ratio": target_ratio,
                    "realized_mesh_ratio": realized_ratio,
                    "set_index": set_index,
                    "instances": [asdict(instance) for instance in instances],
                })
                for placeable_index, asset_name in enumerate(args.placeable_assets):
                    for yaw_index, yaw_deg in enumerate(args.placeable_yaws_deg):
                        placeable_mesh = rotated_meshes[(asset_name, yaw_deg)]
                        feasible = _feasible_centers(
                            mesh_footprints[(asset_name, yaw_deg)],
                            occupied,
                        )
                        assert np.any(
                            feasible
                        ), f"scene {pairing['scene_id']} has no mesh-feasible centers for {asset_name} at {yaw_deg}°"
                        proposal_seed = pairing["proposal_seed_base"] + placeable_index * 100 + yaw_index
                        for method in METHODS:
                            sampled = _sample_method(
                                method,
                                instances,
                                asset_name,
                                placeable_mesh,
                                meshes,
                                rotated_meshes,
                                feasible,
                                proposal_seed,
                                args.target_samples,
                                args.max_attempts_per_sample,
                                args.max_iterations,
                                args.batch_size,
                                args.pitch_m,
                                robolab_api,
                            )
                            samples_xy = sampled.pop("samples_xy")
                            assert sampled["target_reached"], (
                                f"{method} did not reach K={args.target_samples} shared-valid unique layouts "
                                f"for scene {pairing['scene_id']}, {asset_name}, yaw={yaw_deg}"
                            )
                            mean_distance_cm, p90_distance_cm = _nearest_distance_statistics(
                                samples_xy,
                                feasible,
                                args.pitch_m,
                            )
                            measurement = SamplingMeasurement(
                                scene_id=pairing["scene_id"],
                                obstacle_shape_group=shape_group,
                                target_obstacle_ratio=target_ratio,
                                realized_mesh_obstacle_ratio=realized_ratio,
                                obstacle_set_index=set_index,
                                method=method,
                                placeable_asset=asset_name,
                                placeable_yaw_deg=yaw_deg,
                                proposal_seed=proposal_seed,
                                target_samples=args.target_samples,
                                mean_nearest_distance_cm=mean_distance_cm,
                                p90_nearest_distance_cm=p90_distance_cm,
                                coverage_1cm=_coverage_fraction(samples_xy, feasible, args.pitch_m, 0.01),
                                coverage_2cm=_coverage_fraction(samples_xy, feasible, args.pitch_m, 0.02),
                                coverage_3cm=_coverage_fraction(samples_xy, feasible, args.pitch_m, 0.03),
                                **sampled,
                            )
                            measurements.append(measurement)
                            print(
                                f"{shape_group:9} actual={100 * realized_ratio:5.1f}% "
                                f"{method:15} {asset_name:38} yaw={yaw_deg:5.1f} "
                                f"valid={measurement.shared_mesh_valid_count:4}/{measurement.proposal_count:4} "
                                f"mean-nearest={measurement.mean_nearest_distance_cm:5.2f}cm"
                            )

    summaries = _summaries(measurements)
    benchmark_root = Path(__file__).resolve().parents[2]
    payload = {
        "schema_version": 2,
        "experiment": "real-asset-obstacle-ratio-solver-sampling",
        "benchmark_revision": collect_source_revision(benchmark_root),
        "arena_source_revision": _source_revision(benchmark_root),
        "robolab_source_revision": args.robolab_source_revision or _source_revision(args.robolab_root),
        "runtime": {
            "host": platform.node(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "table_asset": TABLE_ASSET,
        "table_bounds": TABLE_BOUNDS,
        "translation_only": True,
        "pitch_m": args.pitch_m,
        "coverage_radii_m": [0.01, 0.02, 0.03],
        "primary_coverage_radius_m": 0.02,
        "mesh_oracle": "rasterized projected collision meshes with exact table containment and overlap convolution",
        "methods": [
            {
                "id": "arena_mesh",
                "label": METHOD_LABELS["arena_mesh"],
                "solver": "Arena RelationSolver",
                "collision_model": "CollisionMode.MESH sphere-to-SDF",
            },
            {
                "id": "arena_aabb",
                "label": METHOD_LABELS["arena_aabb"],
                "solver": "Arena RelationSolver",
                "collision_model": "CollisionMode.BBOX projected fixed-yaw AABB",
            },
            {
                "id": "robolab_circle",
                "label": METHOD_LABELS["robolab_circle"],
                "solver": "native RoboLab SpatialSolver",
                "collision_model": "max-XY-radius circle with radius=max(width,depth)/2",
            },
        ],
        "shape_criteria": {
            "canonical_yaw_deg": 0.0,
            "elongated_pca_aspect_ratio_gte": ELONGATED_ASPECT_RATIO,
            "irregular_aabb_fill_ratio_lte": IRREGULAR_FILL_RATIO,
            "precedence": ["irregular", "elongated", "compact"],
        },
        "placeable_assets": args.placeable_assets,
        "obstacle_assets": args.obstacle_assets,
        "obstacle_assets_by_shape_group": grouped_assets,
        "placeable_yaws_deg": args.placeable_yaws_deg,
        "obstacle_yaws_deg": args.obstacle_yaws_deg,
        "target_obstacle_ratios": args.target_obstacle_ratios,
        "obstacle_set_count": args.obstacle_set_count,
        "target_samples": args.target_samples,
        "max_attempts_per_sample": args.max_attempts_per_sample,
        "solver_max_iterations": args.max_iterations,
        "arena_batch_size": args.batch_size,
        "master_seed": args.seed,
        "asset_resolution_seed": args.asset_resolution_seed,
        "initialization": "shared deterministic uniform XY stream; fixed pre-rotated meshes and yaw",
        "solver_config": {
            "arena": {
                "max_iterations": args.max_iterations,
                "batch_size": args.batch_size,
                "learning_rate": 0.01,
                "convergence_threshold": 1e-4,
                "clearance_m": 0.0,
                "mesh_spheres_per_object": 30,
            },
            "robolab": {
                "max_iterations": args.max_iterations,
                "collision_margin_m": 0.0,
                "dimensions": "canonical mesh AABB; circle radius is invariant to sampled yaw",
                "allow_relaxation": False,
                "invocation": "SpatialSolver._optimize_placement",
                "adaptive_dense_scene_policy": False,
                "fixed_fixed_pairs_ignored": True,
            },
        },
        "assets": asset_metadata,
        "obstacle_scenes": obstacle_scenes,
        "measurements": [asdict(measurement) for measurement in measurements],
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.plot is not None:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        _write_plot(summaries, args.plot)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placeable-assets", type=_parse_strings, default=DEFAULT_PLACEABLE_ASSETS)
    parser.add_argument("--obstacle-assets", type=_parse_strings, default=DEFAULT_OBSTACLE_ASSETS)
    parser.add_argument("--target-obstacle-ratios", type=_parse_floats, default=DEFAULT_TARGET_RATIOS)
    parser.add_argument("--placeable-yaws-deg", type=_parse_floats, default=DEFAULT_YAWS_DEG)
    parser.add_argument("--obstacle-yaws-deg", type=_parse_floats, default=DEFAULT_YAWS_DEG)
    parser.add_argument("--obstacle-set-count", type=int, default=1)
    parser.add_argument("--target-samples", type=int, default=16)
    parser.add_argument("--max-attempts-per-sample", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--asset-resolution-seed", type=int, default=17)
    parser.add_argument("--pitch-m", type=float, default=0.002)
    parser.add_argument("--robolab-root", type=Path, default=Path("/home/zihaox/project/RoboLab-exp1-3"))
    parser.add_argument("--robolab-source-revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    assert args.pitch_m > 0.0
    assert (
        min(
            args.obstacle_set_count,
            args.target_samples,
            args.max_attempts_per_sample,
            args.max_iterations,
            args.batch_size,
        )
        > 0
    )
    assert args.target_obstacle_ratios[0] == 0.0
    assert tuple(sorted(args.target_obstacle_ratios)) == args.target_obstacle_ratios
    assert all(0.0 <= ratio < 1.0 for ratio in args.target_obstacle_ratios)
    assert args.robolab_root.is_dir(), f"RoboLab checkout not found: {args.robolab_root}"
    return args


def main() -> int:
    args = _parse_args()
    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app
    try:
        return _run(args)
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
