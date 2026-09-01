# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Placement diagnostics for existing environment graph specifications."""

from __future__ import annotations

import math
import time

from isaaclab_arena.relations.benchmark.models import BenchmarkMeasurement, BenchmarkScenario, BenchmarkStatus, Clock
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
from isaaclab_arena.relations.object_placer import ObjectPlacer


def _load_scene_workload(scenario: BenchmarkScenario):
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.relations.benchmark.environment_benchmark import _set_environment_benchmark_params
    from isaaclab_arena.relations.passive_collision_objects import get_passive_collision_objects
    from isaaclab_arena.relations.relations import get_anchor_objects

    assert scenario.graph_spec_path is not None
    arena_env = ArenaEnvGraphSpec.from_yaml(scenario.graph_spec_path).to_arena_env()
    if not scenario.include_robot:
        arena_env.embodiment = None
    objects = arena_env.scene.get_objects_with_relations()
    if arena_env.embodiment is not None and arena_env.embodiment.get_relations():
        objects.append(arena_env.embodiment)
    _set_environment_benchmark_params(arena_env, scenario)
    assert arena_env.placer_params is not None
    arena_env.placer_params.apply_positions_to_objects = False
    arena_env.placer_params.placement_seed = scenario.placement_seed
    scene_assets = list(arena_env.scene.assets.values())
    if scenario.background_treatment == "none":
        collision_objects = []
    else:
        include_background = scenario.background_treatment in ("mesh", "scene-default") and (
            scenario.collision_mode == "mesh"
        )
        background_mesh_exclusions = [
            asset for asset in get_anchor_objects(objects) if isinstance(asset, ObjectReference)
        ]
        collision_objects = get_passive_collision_objects(
            scene_assets,
            include_background=include_background,
            background_mesh_exclusions=background_mesh_exclusions,
        )
    relation_count = sum(len(obj.get_spatial_relations()) for obj in objects)
    return objects, collision_objects, arena_env.placer_params, relation_count


def run_scene_placement_diagnostic(
    scenario: BenchmarkScenario,
    *,
    clock: Clock = time.perf_counter,
) -> BenchmarkMeasurement:
    """Measure placement optimization for an existing graph-spec scene."""
    device = get_device_metadata()
    try:
        objects, collision_objects, placer_params, relation_count = _load_scene_workload(scenario)
        placer = ObjectPlacer(placer_params)

        def place_with_seed(seed: int):
            placer.params.placement_seed = seed
            return placer.place(objects, num_envs=scenario.num_envs, collision_objects=collision_objects)

        for warmup_index in range(scenario.warmup_runs):
            time_call(
                lambda: place_with_seed(scenario.placement_seed + scenario.timed_runs + warmup_index),
                clock,
            )
        reset_peak_memory()
        place_samples: list[float] = []
        iteration_rates: list[float] = []
        iterations: list[int] = []
        final_losses: list[float] = []
        valid_layout_rates: list[float] = []
        for run_index in range(scenario.timed_runs):
            elapsed_ms, results = time_call(lambda: place_with_seed(scenario.placement_seed + run_index), clock)
            place_samples.append(elapsed_ms)
            iteration_count = len(placer.last_loss_history)
            optimization_ms = placer.last_optimization_elapsed_ms
            assert iteration_count > 0 and optimization_ms > 0.0
            iterations.append(iteration_count)
            iteration_rates.append(iteration_count * 1e3 / optimization_ms)
            final_losses.extend(result.final_loss for result in results)
            valid_layout_rates.append(sum(result.success for result in results) / len(results))
        peak_allocated, peak_reserved = get_peak_memory()
        place_ms = median(place_samples)
        valid_layout_rate = sum(valid_layout_rates) / len(valid_layout_rates)
        losses_are_finite = all(math.isfinite(loss) for loss in final_losses)
        final_loss = max(final_losses) if losses_are_finite else None
        status: BenchmarkStatus = (
            "ok" if losses_are_finite and valid_layout_rate >= scenario.min_valid_layout_rate else "failed"
        )
        error = None
        if not losses_are_finite:
            error = "final loss is not finite"
        elif status == "failed":
            error = f"valid layout rate {valid_layout_rate:.3f} is below {scenario.min_valid_layout_rate:.3f}"
        return BenchmarkMeasurement.from_scenario(
            scenario,
            "placer",
            status=status,
            device=record_free_memory_after(device),
            num_objects=len(objects),
            error=error,
            place_ms_samples=tuple(place_samples),
            place_ms=place_ms,
            solver_iterations_per_second_samples=tuple(iteration_rates),
            solver_iterations_per_second=median(iteration_rates),
            throughput_envs_per_second=throughput(scenario.num_envs, place_ms),
            iterations=tuple(iterations),
            final_loss=final_loss,
            valid_layout_rate=valid_layout_rate,
            aabb_pair_count=placer.last_aabb_no_overlap_pair_count,
            mesh_pair_count=placer.last_mesh_no_overlap_pair_count,
            relation_count=relation_count,
            background_object_count=len(collision_objects),
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
        )
    except Exception as error:
        return failed_measurement(scenario, "placer", device, error)
