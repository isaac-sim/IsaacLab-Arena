# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CLI for relation-solver wall-clock benchmarks (sim-free dummy scenes)."""

from __future__ import annotations

import argparse
import sys

from isaaclab_arena.relations.relation_solver_benchmark import (
    BenchmarkScenario,
    CollisionModeName,
    default_scenarios,
    env_count_sweep,
    format_results_table,
    mesh_collision_available,
    object_count_sweep,
    run_benchmarks,
    scenarios_for_modes,
    write_results_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RelationSolver and ObjectPlacer wall time.")
    parser.add_argument(
        "--suite",
        choices=("presets", "objects", "envs"),
        default="presets",
        help="presets: bundled small/medium/large; objects: sweep object count; envs: sweep batch size.",
    )
    parser.add_argument(
        "--collision-mode",
        choices=("bbox", "mesh"),
        default="bbox",
        help="Collision backend. mesh requires the collision_mode module and Warp.",
    )
    parser.add_argument(
        "--compare-modes",
        action="store_true",
        help="Run bbox and mesh for each scenario (skips mesh when unavailable).",
    )
    parser.add_argument("--max-iters", type=int, default=600, help="Adam iteration cap per solve.")
    parser.add_argument("--num-spheres", type=int, default=30, help="Spheres per object in mesh mode.")
    parser.add_argument("--seed", type=int, default=0, help="Placement RNG seed.")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed solves before measurement.")
    parser.add_argument("--repeat", type=int, default=3, help="Timed solves; report median wall time.")
    parser.add_argument("--solver-only", action="store_true", help="Skip ObjectPlacer.place() timing.")
    parser.add_argument("--json", dest="json_path", metavar="PATH", help="Write results JSON to PATH.")
    return parser.parse_args()


def _base_scenarios(args: argparse.Namespace) -> tuple[BenchmarkScenario, ...]:
    if args.suite == "objects":
        return object_count_sweep(max_iters=args.max_iters)
    if args.suite == "envs":
        return env_count_sweep(max_iters=args.max_iters)
    return default_scenarios()


def _collision_modes(args: argparse.Namespace) -> tuple[CollisionModeName, ...]:
    if args.compare_modes:
        modes: list[CollisionModeName] = ["bbox"]
        if mesh_collision_available():
            modes.append("mesh")
        elif args.collision_mode == "mesh":
            print("mesh mode unavailable (need collision_mode module and Warp); running bbox only.", file=sys.stderr)
        return tuple(modes)
    if args.collision_mode == "mesh" and not mesh_collision_available():
        print("mesh collision_mode requires the collision_mode module and Warp.", file=sys.stderr)
        sys.exit(1)
    return (args.collision_mode,)


def _apply_run_settings(
    scenarios: tuple[BenchmarkScenario, ...],
    args: argparse.Namespace,
) -> tuple[BenchmarkScenario, ...]:
    return tuple(
        BenchmarkScenario(
            name=scenario.name,
            num_objects=scenario.num_objects,
            num_envs=scenario.num_envs,
            max_iters=args.max_iters,
            collision_mode=scenario.collision_mode,
            num_spheres=args.num_spheres,
            placement_seed=args.seed,
            warmup_runs=args.warmup,
            timed_runs=args.repeat,
        )
        for scenario in scenarios
    )


def main() -> None:
    args = _parse_args()
    base = _base_scenarios(args)
    modes = _collision_modes(args)
    scenarios = _apply_run_settings(scenarios_for_modes(base, modes), args)
    rows = run_benchmarks(scenarios, include_placer=not args.solver_only)
    print(format_results_table(rows))
    if args.json_path:
        write_results_json(args.json_path, rows)


if __name__ == "__main__":
    main()
