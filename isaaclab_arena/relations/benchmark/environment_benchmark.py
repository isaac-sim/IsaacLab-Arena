# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Environment benchmark construction and first-reset runner."""

from __future__ import annotations

import time
import torch
import uuid
from dataclasses import dataclass, replace

from isaaclab_arena.relations.benchmark.models import BenchmarkMeasurement, BenchmarkScenario, Clock
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
from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_asset import PlaceableAsset


@dataclass(frozen=True)
class _EnvironmentSample:
    build_ms: float
    reset_ms: float
    num_objects: int
    include_robot: bool
    free_memory_bytes: int | None


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
    # Reject best-loss fallbacks so a successful reset implies valid placement.
    arena_env.placer_params.allow_best_loss_fallbacks = False


def _validate_environment_mesh_assets(placement_assets: list[PlaceableAsset]) -> None:
    """Require enough environment geometry for a mesh collision pair."""
    has_collision_mesh = any(asset.get_collision_mesh() is not None for asset in placement_assets)
    if len(placement_assets) < 2 or not has_collision_mesh:
        raise RuntimeError("mesh environment benchmark did not load a usable mesh collision pair")


def _build_and_reset_environment(scenario: BenchmarkScenario, clock: Clock) -> _EnvironmentSample:
    """Build and reset one graph-spec environment after SimulationApp startup."""
    import gymnasium as gym

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    assert scenario.graph_spec_path is not None
    arena_env = ArenaEnvGraphSpec.from_yaml(scenario.graph_spec_path).to_arena_env()
    if not scenario.include_robot:
        arena_env.embodiment = None
    include_robot = arena_env.embodiment is not None
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
        build_ms, env = time_call(builder.make_registered, clock)
        reset_ms, _ = time_call(env.reset, clock)
        if scenario.collision_mode == "mesh":
            _validate_environment_mesh_assets(placement_assets)
        live_free_memory = torch.cuda.mem_get_info(0)[0] if torch.cuda.is_available() else None
        return _EnvironmentSample(
            build_ms=build_ms,
            reset_ms=reset_ms,
            num_objects=len(placement_assets),
            include_robot=include_robot,
            free_memory_bytes=live_free_memory,
        )
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
    device = get_device_metadata()
    try:
        assert scenario.graph_spec_path is not None, "environment benchmarks require graph_spec_path"
        for _ in range(scenario.warmup_runs):
            _build_and_reset_environment(scenario, clock)
        reset_peak_memory()
        samples = [_build_and_reset_environment(scenario, clock) for _ in range(scenario.timed_runs)]
        first_sample = samples[0]
        assert all(sample.num_objects == first_sample.num_objects for sample in samples)
        assert all(sample.include_robot == first_sample.include_robot for sample in samples)
        build_samples = [sample.build_ms for sample in samples]
        reset_samples = [sample.reset_ms for sample in samples]
        bring_up_samples = [sample.build_ms + sample.reset_ms for sample in samples]
        minimum_free_memory = min(
            (sample.free_memory_bytes for sample in samples if sample.free_memory_bytes is not None),
            default=None,
        )
        peak_allocated, peak_reserved = get_peak_memory()
        build_ms = median(build_samples)
        reset_ms = median(reset_samples)
        bring_up_ms = median(bring_up_samples)
        measured_device = replace(
            record_free_memory_after(device),
            minimum_free_memory_bytes=minimum_free_memory,
        )
        return BenchmarkMeasurement.from_scenario(
            scenario,
            "environment",
            status="ok",
            device=measured_device,
            num_objects=first_sample.num_objects,
            include_robot=first_sample.include_robot,
            build_ms_samples=tuple(build_samples),
            build_ms=build_ms,
            reset_ms_samples=tuple(reset_samples),
            reset_ms=reset_ms,
            bring_up_ms_samples=tuple(bring_up_samples),
            bring_up_ms=bring_up_ms,
            throughput_envs_per_second=throughput(scenario.num_envs, bring_up_ms),
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
        )
    except Exception as error:
        return failed_measurement(scenario, "environment", device, error)
