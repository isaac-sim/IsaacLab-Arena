# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Synthetic relation benchmark scenes and runners."""

from __future__ import annotations

import importlib.util
import math
import time
import torch
from dataclasses import replace
from typing import TYPE_CHECKING

from isaaclab_arena.relations.benchmark.environment_benchmark import run_environment_benchmark
from isaaclab_arena.relations.benchmark.models import (
    BenchmarkMeasurement,
    BenchmarkScenario,
    BenchmarkStatus,
    BenchmarkTarget,
    Clock,
    CollisionModeName,
)
from isaaclab_arena.relations.benchmark.timing import (
    failed_measurement,
    get_device_metadata,
    get_peak_memory,
    median,
    record_free_memory_after,
    reset_peak_memory,
    throughput,
    time_call,
)
from isaaclab_arena.relations.bounding_box_helpers import assign_variants_for_envs, build_per_env_bounding_boxes
from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_asset import PlaceableAsset
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, On, get_anchor_objects
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    import trimesh


class BenchmarkAsset(PlaceableAsset):
    """Geometry-only placeable asset used by the sim-free benchmark."""

    def __init__(
        self,
        name: str,
        bounding_box: AxisAlignedBoundingBox,
        collision_mesh: trimesh.Trimesh | None = None,
    ) -> None:
        super().__init__(name=name)
        self._bounding_box = bounding_box
        self._collision_mesh = collision_mesh

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return root-relative bounds."""
        return self._bounding_box

    def get_collision_mesh(self) -> trimesh.Trimesh | None:
        """Return the collision mesh when the scenario uses mesh collision."""
        return self._collision_mesh


def mesh_collision_available() -> bool:
    """Return whether required mesh-collision modules are installed."""
    return importlib.util.find_spec("warp") is not None and importlib.util.find_spec("trimesh") is not None


def require_mesh_collision() -> None:
    """Initialize mesh-collision dependencies or raise their runtime error."""
    if not mesh_collision_available():
        raise RuntimeError("mesh collision requires Warp and trimesh")
    import warp as wp

    wp.init()


def default_scenarios() -> tuple[BenchmarkScenario, ...]:
    """Return the short default synthetic suite."""
    return (
        BenchmarkScenario(name="small", num_objects=3, num_envs=1),
        BenchmarkScenario(name="medium", num_objects=6, num_envs=8),
        BenchmarkScenario(name="large", num_objects=10, num_envs=32),
    )


def object_count_sweep(
    *,
    num_envs: int = 1,
    counts: tuple[int, ...] = (3, 5, 6, 10),
    collision_mode: CollisionModeName = "bbox",
    max_iters: int = 600,
) -> tuple[BenchmarkScenario, ...]:
    """Hold batch size fixed and vary object count."""
    return tuple(
        BenchmarkScenario(
            name=f"objs-{count}",
            num_objects=count,
            num_envs=num_envs,
            max_iters=max_iters,
            collision_mode=collision_mode,
        )
        for count in counts
    )


def env_count_sweep(
    *,
    num_objects: int = 3,
    env_counts: tuple[int, ...] = (1, 8, 32, 128),
    collision_mode: CollisionModeName = "bbox",
    max_iters: int = 600,
) -> tuple[BenchmarkScenario, ...]:
    """Hold object count fixed and vary batch size."""
    return tuple(
        BenchmarkScenario(
            name=f"envs-{count}",
            num_objects=num_objects,
            num_envs=count,
            max_iters=max_iters,
            collision_mode=collision_mode,
        )
        for count in env_counts
    )


def scenarios_for_modes(
    base_scenarios: tuple[BenchmarkScenario, ...],
    collision_modes: tuple[CollisionModeName, ...],
) -> tuple[BenchmarkScenario, ...]:
    """Expand every base scenario across requested collision modes."""
    assert collision_modes, "at least one collision mode is required"
    return tuple(
        replace(
            scenario,
            name=scenario.name if len(collision_modes) == 1 else f"{scenario.name}-{mode}",
            collision_mode=mode,
        )
        for scenario in base_scenarios
        for mode in collision_modes
    )


def build_clutter_scene(
    num_objects: int,
    collision_mode: CollisionModeName = "bbox",
) -> list[PlaceableAsset]:
    """Build an anchor table and movable boxes with On relations."""
    assert num_objects >= 2, f"need at least anchor + one box, got {num_objects}"
    if collision_mode == "mesh":
        require_mesh_collision()
        import trimesh

        table_mesh = trimesh.creation.box(extents=(1.0, 1.0, 0.1))
        box_mesh = trimesh.creation.box(extents=(0.12, 0.12, 0.12))
        table_bbox = AxisAlignedBoundingBox(min_point=(-0.5, -0.5, -0.05), max_point=(0.5, 0.5, 0.05))
        box_bbox = AxisAlignedBoundingBox(min_point=(-0.06, -0.06, -0.06), max_point=(0.06, 0.06, 0.06))
    else:
        table_mesh = None
        box_mesh = None
        table_bbox = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1))
        box_bbox = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.12, 0.12, 0.12))

    table = BenchmarkAsset("table", table_bbox, table_mesh)
    table.add_relation(IsAnchor())
    table.set_initial_pose(Pose.identity())
    boxes: list[PlaceableAsset] = []
    for index in range(num_objects - 1):
        box = BenchmarkAsset(f"box-{index}", box_bbox, box_mesh)
        box.add_relation(On(table, clearance_m=0.01))
        boxes.append(box)
    return [table, *boxes]


def make_solver_params(scenario: BenchmarkScenario) -> RelationSolverParams:
    """Build solver parameters for a scenario."""
    if scenario.collision_mode == "mesh":
        require_mesh_collision()
    return RelationSolverParams(
        max_iters=scenario.max_iters,
        convergence_threshold=scenario.convergence_threshold,
        verbose=False,
        profile=False,
        save_position_history=False,
        collision_mode=CollisionMode(scenario.collision_mode),
        num_spheres=scenario.num_spheres,
    )


def make_placer_params(scenario: BenchmarkScenario) -> ObjectPlacerParams:
    """Build object-placer parameters for a scenario."""
    return ObjectPlacerParams(
        solver_params=make_solver_params(scenario),
        placement_seed=scenario.placement_seed,
        max_placement_attempts=scenario.max_placement_attempts,
        apply_positions_to_objects=False,
        verbose=False,
    )


def _sample_child_origin(
    parent_min: float,
    parent_max: float,
    child_min: float,
    child_max: float,
    generator: torch.Generator,
) -> float:
    low = parent_min - child_min
    high = parent_max - child_max
    if low >= high:
        return float((parent_min + parent_max) / 2.0)
    return float(low + (high - low) * torch.rand(1, generator=generator).item())


def _initial_positions_for_env(
    objects: list[PlaceableAsset],
    anchor_objects: set[PlaceableAsset],
    env_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
    generator: torch.Generator,
) -> dict[PlaceableAsset, tuple[float, float, float]]:
    anchor = next(iter(anchor_objects))
    anchor_pose = anchor.get_initial_pose()
    assert isinstance(anchor_pose, Pose)
    anchor_bbox = env_bboxes[anchor].translated(anchor_pose.position_xyz)
    positions: dict[PlaceableAsset, tuple[float, float, float]] = {}
    for obj in objects:
        if obj in anchor_objects:
            positions[obj] = anchor_pose.position_xyz
            continue
        on_relation = next(relation for relation in obj.get_relations() if isinstance(relation, On))
        parent = on_relation.parent
        parent_bbox = anchor_bbox if parent in anchor_objects else env_bboxes[parent].translated(positions[parent])
        child_bbox = env_bboxes[obj]
        positions[obj] = (
            _sample_child_origin(
                float(parent_bbox.min_point[0, 0]),
                float(parent_bbox.max_point[0, 0]),
                float(child_bbox.min_point[0, 0]),
                float(child_bbox.max_point[0, 0]),
                generator,
            ),
            _sample_child_origin(
                float(parent_bbox.min_point[0, 1]),
                float(parent_bbox.max_point[0, 1]),
                float(child_bbox.min_point[0, 1]),
                float(child_bbox.max_point[0, 1]),
                generator,
            ),
            float(parent_bbox.max_point[0, 2] + on_relation.clearance_m - child_bbox.min_point[0, 2]),
        )
    return positions


def build_solve_inputs(
    objects: list[PlaceableAsset],
    num_envs: int,
    seed: int,
) -> tuple[
    list[dict[PlaceableAsset, tuple[float, float, float]]],
    dict[PlaceableAsset, AxisAlignedBoundingBox],
]:
    """Build deterministic positions and per-environment bounds."""
    anchor_objects = set(get_anchor_objects(objects))
    assert len(anchor_objects) == 1
    assign_variants_for_envs(objects, num_envs, placement_seed=seed)
    env_bboxes = build_per_env_bounding_boxes(objects, num_envs)
    per_env_bboxes = env_bboxes.get_bounding_boxes_for_all_envs()
    candidate_bboxes = env_bboxes.get_bounding_boxes_for_solver_candidates(1)
    generator = torch.Generator()
    positions = []
    for env_index in range(num_envs):
        generator.manual_seed(seed + env_index)
        positions.append(_initial_positions_for_env(objects, anchor_objects, per_env_bboxes[env_index], generator))
    return positions, candidate_bboxes


def run_solver_benchmark(
    scenario: BenchmarkScenario,
    *,
    clock: Clock = time.perf_counter,
) -> BenchmarkMeasurement:
    """Benchmark RelationSolver.solve, including state construction."""
    device = get_device_metadata()
    try:
        objects = build_clutter_scene(scenario.num_objects, scenario.collision_mode)
        solver = RelationSolver(make_solver_params(scenario))
        positions, bboxes = build_solve_inputs(objects, scenario.num_envs, scenario.placement_seed)

        def solve() -> None:
            solver.solve(objects, positions, env_bboxes=bboxes)

        for _ in range(scenario.warmup_runs):
            time_call(solve, clock)
        reset_peak_memory()
        samples: list[float] = []
        iterations: list[int] = []
        final_losses: list[float] = []
        for _ in range(scenario.timed_runs):
            elapsed_ms, _ = time_call(solve, clock)
            samples.append(elapsed_ms)
            iterations.append(len(solver.last_loss_history))
            assert solver.last_loss_per_env is not None
            final_losses.append(float(solver.last_loss_per_env.max().item()))
        peak_allocated, peak_reserved = get_peak_memory()
        solve_ms = median(samples)
        solve_step_samples = [
            sample / iteration_count for sample, iteration_count in zip(samples, iterations, strict=True)
        ]
        finite_loss = all(math.isfinite(loss) for loss in final_losses)
        final_loss = max(final_losses) if finite_loss else None
        status: BenchmarkStatus = (
            "ok" if final_loss is not None and final_loss <= scenario.final_loss_threshold else "failed"
        )
        error = None
        if not finite_loss:
            error = "final loss is not finite"
        elif status == "failed":
            assert final_loss is not None
            error = f"final loss {final_loss:.6g} exceeds threshold {scenario.final_loss_threshold:.6g}"
        return BenchmarkMeasurement.from_scenario(
            scenario,
            "solver",
            status=status,
            device=record_free_memory_after(device),
            error=error,
            solve_ms_samples=tuple(samples),
            solve_ms=solve_ms,
            solve_step_ms_samples=tuple(solve_step_samples),
            solve_step_ms=median(solve_step_samples),
            throughput_envs_per_second=throughput(scenario.num_envs, solve_ms),
            iterations=tuple(iterations),
            final_loss=final_loss,
            aabb_pair_count=solver.last_aabb_no_overlap_pair_count,
            mesh_pair_count=solver.last_mesh_no_overlap_pair_count,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
        )
    except Exception as error:
        return failed_measurement(scenario, "solver", device, error)


def run_placer_benchmark(
    scenario: BenchmarkScenario,
    *,
    clock: Clock = time.perf_counter,
) -> BenchmarkMeasurement:
    """Benchmark ObjectPlacer.place end to end."""
    device = get_device_metadata()
    try:
        objects = build_clutter_scene(scenario.num_objects, scenario.collision_mode)
        placer = ObjectPlacer(make_placer_params(scenario))

        def place():
            return placer.place(objects, num_envs=scenario.num_envs)

        for _ in range(scenario.warmup_runs):
            time_call(place, clock)
        reset_peak_memory()
        samples: list[float] = []
        iterations: list[int] = []
        final_losses: list[float] = []
        valid_rates: list[float] = []
        for _ in range(scenario.timed_runs):
            elapsed_ms, results = time_call(place, clock)
            samples.append(elapsed_ms)
            iterations.append(len(placer.last_loss_history))
            final_losses.extend(result.final_loss for result in results)
            valid_rates.append(sum(result.success for result in results) / len(results))
        peak_allocated, peak_reserved = get_peak_memory()
        place_ms = median(samples)
        valid_rate = min(valid_rates)
        finite_loss = all(math.isfinite(loss) for loss in final_losses)
        final_loss = max(final_losses) if finite_loss else None
        status: BenchmarkStatus = "ok" if finite_loss and valid_rate >= scenario.min_valid_layout_rate else "failed"
        error = None
        if not finite_loss:
            error = "final loss is not finite"
        elif status == "failed":
            error = f"valid layout rate {valid_rate:.3f} is below {scenario.min_valid_layout_rate:.3f}"
        return BenchmarkMeasurement.from_scenario(
            scenario,
            "placer",
            status=status,
            device=record_free_memory_after(device),
            error=error,
            place_ms_samples=tuple(samples),
            place_ms=place_ms,
            throughput_envs_per_second=throughput(scenario.num_envs, place_ms),
            iterations=tuple(iterations),
            final_loss=final_loss,
            valid_layout_rate=valid_rate,
            aabb_pair_count=placer.last_aabb_no_overlap_pair_count,
            mesh_pair_count=placer.last_mesh_no_overlap_pair_count,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
        )
    except Exception as error:
        return failed_measurement(scenario, "placer", device, error)


def run_benchmarks(
    scenarios: tuple[BenchmarkScenario, ...],
    *,
    targets: tuple[BenchmarkTarget, ...] = ("solver", "placer"),
    clock: Clock = time.perf_counter,
) -> list[BenchmarkMeasurement]:
    """Run each requested target for every scenario."""
    runners = {
        "solver": run_solver_benchmark,
        "placer": run_placer_benchmark,
        "environment": run_environment_benchmark,
    }
    return [runners[target](scenario, clock=clock) for scenario in scenarios for target in targets]
