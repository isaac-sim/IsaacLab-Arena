# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for relation solving, object placement, and environment construction."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import platform
import socket
import statistics
import time
import torch
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

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

BenchmarkStatus = Literal["ok", "failed"]
BenchmarkTarget = Literal["solver", "placer", "environment"]
CollisionModeName = Literal["bbox", "mesh"]
Clock = Callable[[], float]
T = TypeVar("T")

RUN_SCHEMA_VERSION = 1


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


@dataclass(frozen=True)
class BenchmarkScenario:
    """One relation-solver benchmark configuration."""

    name: str
    """Human-readable scenario name."""

    num_objects: int
    """Total synthetic objects, including one anchor."""

    num_envs: int
    """Independent layouts in the solver batch."""

    max_iters: int = 600
    """Maximum Adam iterations."""

    convergence_threshold: float = 0.0
    """Per-environment loss threshold used to stop optimization."""

    collision_mode: CollisionModeName = "bbox"
    """Collision backend."""

    num_spheres: int = 30
    """Bounding spheres per subject in mesh mode."""

    placement_seed: int = 0
    """Seed for initial layouts."""

    max_placement_attempts: int = 10
    """Candidate layouts solved per placer result."""

    warmup_runs: int = 1
    """Untimed runs before measurement."""

    timed_runs: int = 3
    """Measured runs."""

    final_loss_threshold: float = 1e-4
    """Largest acceptable final loss in every solver environment."""

    min_valid_layout_rate: float = 1.0
    """Smallest acceptable valid-layout fraction for a placer run."""

    graph_spec_path: str | None = None
    """Graph spec used by environment-target measurements."""

    include_robot: bool = True
    """Whether environment-target measurements include the embodiment."""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if self.graph_spec_path is None:
            if self.num_objects < 2:
                raise ValueError("num_objects must include an anchor and one movable object")
        elif self.num_objects < 0:
            raise ValueError("num_objects must be non-negative")
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.max_iters <= 0:
            raise ValueError("max_iters must be positive")
        if not math.isfinite(self.convergence_threshold) or self.convergence_threshold < 0.0:
            raise ValueError("convergence_threshold must be finite and non-negative")
        if self.collision_mode not in ("bbox", "mesh"):
            raise ValueError("collision_mode must be bbox or mesh")
        if self.num_spheres <= 0:
            raise ValueError("num_spheres must be positive")
        if self.max_placement_attempts <= 0:
            raise ValueError("max_placement_attempts must be positive")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")
        if self.timed_runs <= 0:
            raise ValueError("timed_runs must be positive")
        if not math.isfinite(self.final_loss_threshold) or self.final_loss_threshold < 0.0:
            raise ValueError("final_loss_threshold must be finite and non-negative")
        if not math.isfinite(self.min_valid_layout_rate) or not 0.0 <= self.min_valid_layout_rate <= 1.0:
            raise ValueError("min_valid_layout_rate must be in [0, 1]")

    def scenario_id(self, target: BenchmarkTarget) -> str:
        """Return a stable identifier for one target measurement."""
        parts = [self.name, target, self.collision_mode, f"envs-{self.num_envs}"]
        if target == "environment":
            spec_name = Path(self.graph_spec_path).stem if self.graph_spec_path is not None else "missing-spec"
            parts.extend([spec_name, "robot" if self.include_robot else "no-robot"])
        return "__".join(parts)


@dataclass(frozen=True)
class DeviceMetadata:
    """GPU identity and memory state for one worker."""

    physical_device: str | None
    """Physical selector assigned through CUDA_VISIBLE_DEVICES."""

    name: str | None
    """CUDA device name."""

    total_memory_bytes: int | None
    """Device memory capacity."""

    free_memory_before_bytes: int | None
    """Free memory immediately before timed runs."""

    free_memory_after_bytes: int | None
    """Free memory immediately after timed runs."""

    minimum_free_memory_bytes: int | None
    """Lowest free memory observed while the measured workload was live."""

    compute_capability: str | None
    """CUDA compute capability."""


@dataclass(frozen=True)
class BenchmarkMeasurement:
    """Result for one benchmark target."""

    scenario_id: str
    """Stable case identifier."""

    scenario_name: str
    """Human-readable scenario name."""

    target: BenchmarkTarget
    """Measured operation."""

    status: BenchmarkStatus
    """Whether the operation and its correctness gate passed."""

    collision_mode: CollisionModeName
    """Collision backend."""

    num_objects: int
    """Placement object count."""

    num_envs: int
    """Number of layouts measured together."""

    device: DeviceMetadata
    """Worker device metadata."""

    max_iters: int
    """Maximum solver iterations."""

    convergence_threshold: float
    """Per-environment solver stopping threshold."""

    placement_seed: int
    """Initial-layout seed."""

    max_placement_attempts: int
    """Candidate batches available to the placer."""

    warmup_runs: int
    """Excluded warmup runs."""

    timed_runs: int
    """Recorded runs."""

    worker_id: str = "local"
    """Worker that produced the measurement."""

    throughput_envs_per_second: float | None = None
    """Measured layout throughput."""

    aggregate_throughput_envs_per_second: float | None = None
    """Sum of replicated worker throughput."""

    include_robot: bool | None = None
    """Robot inclusion for environment targets."""

    error: str | None = None
    """Failure reason."""

    solve_ms_samples: tuple[float, ...] | None = None
    """Solver latency samples."""

    solve_ms: float | None = None
    """Median solver latency."""

    solve_step_ms_samples: tuple[float, ...] | None = None
    """Solver latency divided by iterations for each timed run."""

    solve_step_ms: float | None = None
    """Median solver iteration latency."""

    place_ms_samples: tuple[float, ...] | None = None
    """ObjectPlacer latency samples."""

    place_ms: float | None = None
    """Median ObjectPlacer latency."""

    build_ms_samples: tuple[float, ...] | None = None
    """Environment construction latency samples."""

    build_ms: float | None = None
    """Median environment construction latency."""

    reset_ms_samples: tuple[float, ...] | None = None
    """First-reset latency samples."""

    reset_ms: float | None = None
    """Median first-reset latency."""

    iterations: tuple[int, ...] | None = None
    """Solver iterations for each timed run."""

    final_loss: float | None = None
    """Largest final per-environment loss."""

    valid_layout_rate: float | None = None
    """Smallest valid-layout fraction across timed runs."""

    aabb_pair_count: int | None = None
    """Directed AABB pairs in the final timed run."""

    mesh_pair_count: int | None = None
    """Directed sphere-to-mesh pairs in the final timed run."""

    peak_allocated_bytes: int | None = None
    """Peak live tensor bytes."""

    peak_reserved_bytes: int | None = None
    """Peak allocator-reserved bytes."""

    def to_dict(self) -> dict[str, object]:
        """Serialize this measurement."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> BenchmarkMeasurement:
        """Deserialize a measurement."""
        values = dict(data)
        values["device"] = DeviceMetadata(**values["device"])
        sample_fields = (
            "solve_ms_samples",
            "solve_step_ms_samples",
            "place_ms_samples",
            "build_ms_samples",
            "reset_ms_samples",
            "iterations",
        )
        for name in sample_fields:
            if values.get(name) is not None:
                values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class BenchmarkRun:
    """Complete benchmark run and its worker manifest."""

    requested_scenario_ids: tuple[str, ...]
    results: tuple[BenchmarkMeasurement, ...]
    worker_assignments: dict[str, tuple[str, ...]]
    worker_exit_codes: dict[str, int]
    worker_errors: dict[str, str] = field(default_factory=dict)
    schema_version: int = RUN_SCHEMA_VERSION
    hostname: str = field(default_factory=socket.gethostname)
    platform: str = field(default_factory=platform.platform)

    def __post_init__(self) -> None:
        assert self.schema_version == RUN_SCHEMA_VERSION
        observed = [result.scenario_id for result in self.results]
        assigned = [scenario_id for ids in self.worker_assignments.values() for scenario_id in ids]
        assert len(observed) == len(set(observed)), "benchmark results contain duplicate scenario IDs"
        assert set(observed) <= set(self.requested_scenario_ids), "benchmark results contain unrequested scenarios"
        assert assigned == list(self.requested_scenario_ids), "worker assignments do not match requested scenarios"
        assert self.worker_assignments.keys() == self.worker_exit_codes.keys(), "worker manifest keys do not match"
        assert self.worker_errors.keys() <= self.worker_exit_codes.keys(), "worker errors contain unknown workers"

    @property
    def succeeded(self) -> bool:
        """Whether every requested case passed."""
        observed = {result.scenario_id for result in self.results}
        return (
            observed == set(self.requested_scenario_ids)
            and all(result.status == "ok" for result in self.results)
            and all(code == 0 for code in self.worker_exit_codes.values())
            and not self.worker_errors
        )

    @property
    def missing_scenario_ids(self) -> tuple[str, ...]:
        """Requested cases that produced no measurement."""
        observed = {result.scenario_id for result in self.results}
        return tuple(scenario_id for scenario_id in self.requested_scenario_ids if scenario_id not in observed)

    def to_dict(self) -> dict[str, object]:
        """Serialize this run."""
        data = asdict(self)
        data["missing_scenario_ids"] = self.missing_scenario_ids
        return data


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
    num_envs: int = 8,
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
    num_objects: int = 6,
    env_counts: tuple[int, ...] = (1, 8, 32),
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


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _device_metadata() -> DeviceMetadata:
    if not torch.cuda.is_available():
        return DeviceMetadata(None, None, None, None, None, None, None)
    properties = torch.cuda.get_device_properties(0)
    free_memory, total_memory = torch.cuda.mem_get_info(0)
    return DeviceMetadata(
        physical_device=os.environ.get("ARENA_BENCHMARK_PHYSICAL_GPU") or os.environ.get("CUDA_VISIBLE_DEVICES"),
        name=properties.name,
        total_memory_bytes=total_memory,
        free_memory_before_bytes=free_memory,
        free_memory_after_bytes=None,
        minimum_free_memory_bytes=None,
        compute_capability=f"{properties.major}.{properties.minor}",
    )


def _record_free_memory_after(device: DeviceMetadata) -> DeviceMetadata:
    if not torch.cuda.is_available():
        return device
    free_memory, _ = torch.cuda.mem_get_info(0)
    return replace(device, free_memory_after_bytes=free_memory)


def _reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_memory() -> tuple[int | None, int | None]:
    if not torch.cuda.is_available():
        return None, None
    return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()


def _time_call(operation: Callable[[], T], clock: Clock) -> tuple[float, T]:
    _sync_cuda()
    start = clock()
    result = operation()
    _sync_cuda()
    return (clock() - start) * 1e3, result


def _median(samples: list[float]) -> float:
    assert samples, "timed samples must not be empty"
    return statistics.median(samples)


def _throughput(num_envs: int, elapsed_ms: float) -> float | None:
    return num_envs * 1e3 / elapsed_ms if elapsed_ms > 0.0 else None


def _error_message(error: Exception | str) -> str:
    if isinstance(error, str):
        return error
    detail = str(error)
    return f"{type(error).__name__}: {detail}" if detail else type(error).__name__


def _failed_measurement(
    scenario: BenchmarkScenario,
    target: BenchmarkTarget,
    device: DeviceMetadata,
    error: Exception | str,
) -> BenchmarkMeasurement:
    return BenchmarkMeasurement(
        scenario_id=scenario.scenario_id(target),
        scenario_name=scenario.name,
        target=target,
        status="failed",
        collision_mode=scenario.collision_mode,
        num_objects=scenario.num_objects,
        num_envs=scenario.num_envs,
        device=device,
        max_iters=scenario.max_iters,
        convergence_threshold=scenario.convergence_threshold,
        placement_seed=scenario.placement_seed,
        max_placement_attempts=scenario.max_placement_attempts,
        warmup_runs=scenario.warmup_runs,
        timed_runs=scenario.timed_runs,
        include_robot=scenario.include_robot if target == "environment" else None,
        error=_error_message(error),
    )


def run_solver_benchmark(
    scenario: BenchmarkScenario,
    *,
    clock: Clock = time.perf_counter,
) -> BenchmarkMeasurement:
    """Benchmark RelationSolver.solve, including state construction."""
    device = _device_metadata()
    try:
        objects = build_clutter_scene(scenario.num_objects, scenario.collision_mode)
        solver = RelationSolver(make_solver_params(scenario))
        positions, bboxes = build_solve_inputs(objects, scenario.num_envs, scenario.placement_seed)

        def solve() -> None:
            solver.solve(objects, positions, env_bboxes=bboxes)

        for _ in range(scenario.warmup_runs):
            _time_call(solve, clock)
        _reset_peak_memory()
        samples: list[float] = []
        iterations: list[int] = []
        final_losses: list[float] = []
        for _ in range(scenario.timed_runs):
            elapsed_ms, _ = _time_call(solve, clock)
            samples.append(elapsed_ms)
            iterations.append(len(solver.last_loss_history))
            assert solver.last_loss_per_env is not None
            final_losses.append(float(solver.last_loss_per_env.max().item()))
        peak_allocated, peak_reserved = _peak_memory()
        solve_ms = _median(samples)
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
        return BenchmarkMeasurement(
            scenario_id=scenario.scenario_id("solver"),
            scenario_name=scenario.name,
            target="solver",
            status=status,
            collision_mode=scenario.collision_mode,
            num_objects=scenario.num_objects,
            num_envs=scenario.num_envs,
            device=_record_free_memory_after(device),
            max_iters=scenario.max_iters,
            convergence_threshold=scenario.convergence_threshold,
            placement_seed=scenario.placement_seed,
            max_placement_attempts=scenario.max_placement_attempts,
            warmup_runs=scenario.warmup_runs,
            timed_runs=scenario.timed_runs,
            error=error,
            solve_ms_samples=tuple(samples),
            solve_ms=solve_ms,
            solve_step_ms_samples=tuple(solve_step_samples),
            solve_step_ms=_median(solve_step_samples),
            throughput_envs_per_second=_throughput(scenario.num_envs, solve_ms),
            iterations=tuple(iterations),
            final_loss=final_loss,
            aabb_pair_count=solver.last_aabb_no_overlap_pair_count,
            mesh_pair_count=solver.last_mesh_no_overlap_pair_count,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
        )
    except Exception as error:
        return _failed_measurement(scenario, "solver", device, error)


def run_placer_benchmark(
    scenario: BenchmarkScenario,
    *,
    clock: Clock = time.perf_counter,
) -> BenchmarkMeasurement:
    """Benchmark ObjectPlacer.place end to end."""
    device = _device_metadata()
    try:
        objects = build_clutter_scene(scenario.num_objects, scenario.collision_mode)
        placer = ObjectPlacer(make_placer_params(scenario))

        def place():
            return placer.place(objects, num_envs=scenario.num_envs)

        for _ in range(scenario.warmup_runs):
            _time_call(place, clock)
        _reset_peak_memory()
        samples: list[float] = []
        iterations: list[int] = []
        final_losses: list[float] = []
        valid_rates: list[float] = []
        for _ in range(scenario.timed_runs):
            elapsed_ms, results = _time_call(place, clock)
            samples.append(elapsed_ms)
            iterations.append(len(placer.last_loss_history))
            final_losses.extend(result.final_loss for result in results)
            valid_rates.append(sum(result.success for result in results) / len(results))
        peak_allocated, peak_reserved = _peak_memory()
        place_ms = _median(samples)
        valid_rate = min(valid_rates)
        finite_loss = all(math.isfinite(loss) for loss in final_losses)
        final_loss = max(final_losses) if finite_loss else None
        loss_ok = final_loss is not None and final_loss <= scenario.final_loss_threshold
        status: BenchmarkStatus = "ok" if loss_ok and valid_rate >= scenario.min_valid_layout_rate else "failed"
        error = None
        if not finite_loss:
            error = "final loss is not finite"
        elif not loss_ok:
            assert final_loss is not None
            error = f"final loss {final_loss:.6g} exceeds threshold {scenario.final_loss_threshold:.6g}"
        elif status == "failed":
            error = f"valid layout rate {valid_rate:.3f} is below {scenario.min_valid_layout_rate:.3f}"
        return BenchmarkMeasurement(
            scenario_id=scenario.scenario_id("placer"),
            scenario_name=scenario.name,
            target="placer",
            status=status,
            collision_mode=scenario.collision_mode,
            num_objects=scenario.num_objects,
            num_envs=scenario.num_envs,
            device=_record_free_memory_after(device),
            max_iters=scenario.max_iters,
            convergence_threshold=scenario.convergence_threshold,
            placement_seed=scenario.placement_seed,
            max_placement_attempts=scenario.max_placement_attempts,
            warmup_runs=scenario.warmup_runs,
            timed_runs=scenario.timed_runs,
            error=error,
            place_ms_samples=tuple(samples),
            place_ms=place_ms,
            throughput_envs_per_second=_throughput(scenario.num_envs, place_ms),
            iterations=tuple(iterations),
            final_loss=final_loss,
            valid_layout_rate=valid_rate,
            aabb_pair_count=placer.last_aabb_no_overlap_pair_count,
            mesh_pair_count=placer.last_mesh_no_overlap_pair_count,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
        )
    except Exception as error:
        return _failed_measurement(scenario, "placer", device, error)


def _set_environment_benchmark_params(arena_env, scenario: BenchmarkScenario) -> None:
    assets = list(arena_env.scene.assets.values())
    if arena_env.embodiment is not None:
        assets.append(arena_env.embodiment)
    for asset in assets:
        if isinstance(asset, PlaceableAsset):
            asset.collision_mode = CollisionMode(scenario.collision_mode)
    if arena_env.placer_params is None:
        arena_env.placer_params = ObjectPlacerParams()
    arena_env.placer_params.solver_params.collision_mode = CollisionMode(scenario.collision_mode)
    arena_env.placer_params.solver_params.max_iters = scenario.max_iters
    arena_env.placer_params.solver_params.convergence_threshold = scenario.convergence_threshold
    arena_env.placer_params.solver_params.num_spheres = scenario.num_spheres
    arena_env.placer_params.max_placement_attempts = scenario.max_placement_attempts
    arena_env.placer_params.allow_best_loss_fallbacks = False


def _validate_environment_mesh_assets(placement_assets: list[PlaceableAsset]) -> None:
    """Require enough environment geometry for a mesh collision pair."""
    has_collision_mesh = any(asset.get_collision_mesh() is not None for asset in placement_assets)
    if len(placement_assets) < 2 or not has_collision_mesh:
        raise RuntimeError("mesh environment benchmark did not load a usable mesh collision pair")


def _build_and_reset_environment(
    scenario: BenchmarkScenario, clock: Clock
) -> tuple[float, float, int, bool, int | None]:
    """Build and reset one graph-spec environment after SimulationApp startup."""
    import gymnasium as gym

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    assert scenario.graph_spec_path is not None
    arena_env = ArenaEnvGraphSpec.from_yaml(scenario.graph_spec_path).to_arena_env()
    if not scenario.include_robot:
        arena_env.embodiment = None
    includes_robot = arena_env.embodiment is not None
    placement_assets = arena_env.scene.get_objects_with_relations()
    if arena_env.embodiment is not None and arena_env.embodiment.get_relations():
        placement_assets.append(arena_env.embodiment)
    arena_env.name = f"{arena_env.name}-solver-benchmark-{uuid.uuid4().hex}"
    _set_environment_benchmark_params(arena_env, scenario)
    builder = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(
            num_envs=scenario.num_envs,
            placement_seed=scenario.placement_seed,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        ),
    )
    env = None
    try:
        build_ms, env = _time_call(builder.make_registered, clock)
        reset_ms, _ = _time_call(env.reset, clock)
        if scenario.collision_mode == "mesh":
            _validate_environment_mesh_assets(placement_assets)
        live_free_memory = torch.cuda.mem_get_info(0)[0] if torch.cuda.is_available() else None
        return build_ms, reset_ms, len(placement_assets), includes_robot, live_free_memory
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            gym.registry.pop(arena_env.name, None)


def run_environment_benchmark(
    scenario: BenchmarkScenario,
    *,
    clock: Clock = time.perf_counter,
) -> BenchmarkMeasurement:
    """Benchmark environment construction and first reset in a running SimulationApp."""
    device = _device_metadata()
    try:
        assert scenario.graph_spec_path is not None, "environment benchmarks require graph_spec_path"
        for _ in range(scenario.warmup_runs):
            _build_and_reset_environment(scenario, clock)
        _reset_peak_memory()
        build_samples: list[float] = []
        reset_samples: list[float] = []
        object_counts: list[int] = []
        robot_presence: list[bool] = []
        live_free_memory: list[int] = []
        for _ in range(scenario.timed_runs):
            build_ms, reset_ms, object_count, includes_robot, free_memory = _build_and_reset_environment(
                scenario, clock
            )
            build_samples.append(build_ms)
            reset_samples.append(reset_ms)
            object_counts.append(object_count)
            robot_presence.append(includes_robot)
            if free_memory is not None:
                live_free_memory.append(free_memory)
        assert len(set(object_counts)) == 1
        assert len(set(robot_presence)) == 1
        peak_allocated, peak_reserved = _peak_memory()
        build_ms = _median(build_samples)
        reset_ms = _median(reset_samples)
        measured_device = _record_free_memory_after(device)
        if live_free_memory:
            measured_device = replace(measured_device, minimum_free_memory_bytes=min(live_free_memory))
        return BenchmarkMeasurement(
            scenario_id=scenario.scenario_id("environment"),
            scenario_name=scenario.name,
            target="environment",
            status="ok",
            collision_mode=scenario.collision_mode,
            num_objects=object_counts[0],
            num_envs=scenario.num_envs,
            device=measured_device,
            max_iters=scenario.max_iters,
            convergence_threshold=scenario.convergence_threshold,
            placement_seed=scenario.placement_seed,
            max_placement_attempts=scenario.max_placement_attempts,
            warmup_runs=scenario.warmup_runs,
            timed_runs=scenario.timed_runs,
            include_robot=robot_presence[0],
            build_ms_samples=tuple(build_samples),
            build_ms=build_ms,
            reset_ms_samples=tuple(reset_samples),
            reset_ms=reset_ms,
            throughput_envs_per_second=_throughput(scenario.num_envs, build_ms + reset_ms),
            valid_layout_rate=1.0,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
        )
    except Exception as error:
        return _failed_measurement(scenario, "environment", device, error)


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


def requested_scenario_ids(
    scenarios: tuple[BenchmarkScenario, ...],
    targets: tuple[BenchmarkTarget, ...],
) -> tuple[str, ...]:
    """Return every expected result ID in execution order."""
    ids = tuple(scenario.scenario_id(target) for scenario in scenarios for target in targets)
    if len(ids) != len(set(ids)):
        raise ValueError("requested benchmark scenario IDs must be unique")
    return ids


def build_run(
    scenarios: tuple[BenchmarkScenario, ...],
    targets: tuple[BenchmarkTarget, ...],
    results: list[BenchmarkMeasurement],
    worker_assignments: dict[str, tuple[str, ...]] | None = None,
    worker_exit_codes: dict[str, int] | None = None,
    worker_errors: dict[str, str] | None = None,
) -> BenchmarkRun:
    """Build the canonical run envelope."""
    expected = requested_scenario_ids(scenarios, targets)
    exit_codes = worker_exit_codes or {"local": 0}
    if worker_assignments is None:
        worker_id = next(iter(exit_codes)) if len(exit_codes) == 1 else "local"
        worker_assignments = {worker_id: expected}
    return BenchmarkRun(expected, tuple(results), worker_assignments, exit_codes, worker_errors or {})


def build_distributed_run(
    results: list[BenchmarkMeasurement],
    worker_assignments: dict[str, tuple[str, ...]],
    worker_exit_codes: dict[str, int],
    worker_errors: dict[str, str] | None = None,
) -> BenchmarkRun:
    """Build a run whose requested IDs are already worker-qualified."""
    expected = tuple(scenario_id for ids in worker_assignments.values() for scenario_id in ids)
    return BenchmarkRun(expected, tuple(results), worker_assignments, worker_exit_codes, worker_errors or {})


def search_capacity(
    probe: Callable[[int], bool],
    *,
    start_num_envs: int = 1,
    max_num_envs: int = 4096,
) -> int | None:
    """Find the largest viable batch by exponential growth and binary search."""
    assert 0 < start_num_envs <= max_num_envs
    if not probe(start_num_envs):
        return None
    if start_num_envs == max_num_envs:
        return start_num_envs
    low = start_num_envs
    high = min(start_num_envs * 2, max_num_envs)
    while probe(high):
        low = high
        if high == max_num_envs:
            return high
        high = min(high * 2, max_num_envs)
    while low + 1 < high:
        middle = (low + high) // 2
        if probe(middle):
            low = middle
        else:
            high = middle
    return low


def format_results_table(results: list[BenchmarkMeasurement]) -> str:
    """Render a compact text report."""
    header = (
        f"{'scenario':<28} {'worker':<10} {'target':<11} {'status':<7} {'mode':<5} {'envs':>5} "
        f"{'median_ms':>10} {'step_ms':>9} {'iters':>7} {'env/s':>10} {'agg env/s':>10} "
        f"{'loss':>10} {'valid':>7}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        median_ms = {
            "solver": result.solve_ms,
            "placer": result.place_ms,
            "environment": result.build_ms,
        }[result.target]
        lines.append(
            f"{result.scenario_name:<28} {result.worker_id:<10} {result.target:<11} {result.status:<7} "
            f"{result.collision_mode:<5} {result.num_envs:>5} "
            f"{_format_number(median_ms):>10} {_format_number(result.solve_step_ms):>9} "
            f"{_format_iterations(result.iterations):>7} "
            f"{_format_number(result.throughput_envs_per_second):>10} "
            f"{_format_number(result.aggregate_throughput_envs_per_second):>10} "
            f"{_format_number(result.final_loss):>10} {_format_number(result.valid_layout_rate):>7}"
        )
        if result.error:
            lines.append(f"  error: {result.error}")
    return "\n".join(lines)


def _format_number(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _format_iterations(iterations: tuple[int, ...] | None) -> str:
    return "-" if not iterations else f"{statistics.median(iterations):.0f}"


def write_results_json(path: str | Path, run: BenchmarkRun) -> None:
    """Write the canonical benchmark envelope."""
    Path(path).write_text(json.dumps(run.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_results_csv(path: str | Path, run: BenchmarkRun) -> None:
    """Write flat result rows after run completeness validation."""
    rows = [result.to_dict() for result in run.results]
    fieldnames = list(rows[0]) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["device"] = json.dumps(row["device"], sort_keys=True)
            writer.writerow(row)
