# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Data models and shared type aliases for relation benchmarks."""

from __future__ import annotations

import math
import platform
import socket
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

BenchmarkStatus = Literal["ok", "failed"]
BenchmarkTarget = Literal["solver", "placer", "environment"]
CollisionModeName = Literal["bbox", "mesh"]
Clock = Callable[[], float]

RUN_SCHEMA_VERSION = 1


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
    """Throughput derived from the target's median latency."""

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
    """Largest final loss across timed runs and environments."""

    valid_layout_rate: float | None = None
    """Smallest placer valid-layout fraction across timed runs; None for other targets."""

    aabb_pair_count: int | None = None
    """Directed AABB pairs in the final timed run."""

    mesh_pair_count: int | None = None
    """Cached directed mesh object pairs in the final timed run."""

    peak_allocated_bytes: int | None = None
    """Peak live tensor bytes."""

    peak_reserved_bytes: int | None = None
    """Peak allocator-reserved bytes."""

    @classmethod
    def from_scenario(
        cls,
        scenario: BenchmarkScenario,
        target: BenchmarkTarget,
        *,
        status: BenchmarkStatus,
        device: DeviceMetadata,
        num_objects: int | None = None,
        include_robot: bool | None = None,
        **measurement_values,
    ) -> BenchmarkMeasurement:
        """Build a measurement with scenario metadata."""
        if target == "environment" and include_robot is None:
            include_robot = scenario.include_robot
        return cls(
            scenario_id=scenario.scenario_id(target),
            scenario_name=scenario.name,
            target=target,
            status=status,
            collision_mode=scenario.collision_mode,
            num_objects=scenario.num_objects if num_objects is None else num_objects,
            num_envs=scenario.num_envs,
            device=device,
            max_iters=scenario.max_iters,
            convergence_threshold=scenario.convergence_threshold,
            placement_seed=scenario.placement_seed,
            max_placement_attempts=scenario.max_placement_attempts,
            warmup_runs=scenario.warmup_runs,
            timed_runs=scenario.timed_runs,
            include_robot=include_robot,
            **measurement_values,
        )

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
