# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run relation-solver benchmarks locally or with one independent worker per GPU.

Each selected GPU runs a full copy of the benchmark matrix. A solver batch is
never split across GPUs; aggregate throughput is the sum of replicated workers.

Examples:
  Fixed bbox/mesh sweep:
    /isaac-sim/python.sh isaaclab_arena_examples/relations/relation_solver_benchmark.py \\
      --suite envs --targets solver,placer --compare-modes \\
      --num-envs 1 --num-envs 8 --num-envs 32

  Exactly one environment bring-up case:
    /isaac-sim/python.sh isaaclab_arena_examples/relations/relation_solver_benchmark.py \\
      --targets environment \\
      --environment-spec isaaclab_arena_environments/robolab/tasks/banana_in_bowl.yaml \\
      --collision-mode bbox --robot-mode with --num-envs 1 --warmup 0 --repeat 1

  Comprehensive synthetic and environment benchmark on one GPU:
    /isaac-sim/python.sh isaaclab_arena_examples/relations/relation_solver_benchmark.py \\
      --suite comprehensive

  Question-driven solver diagnostic page for an MR:
    /isaac-sim/python.sh isaaclab_arena_examples/relations/relation_solver_benchmark.py \\
      --suite diagnostic

  Mesh-complexity and GPU memory-capacity experiment:
    /isaac-sim/python.sh isaaclab_arena_examples/relations/relation_solver_benchmark.py \\
      --suite memory --output-dir ~/solver-benchmark-results/memory

  Stable solver-performance regression matrix for local or OSMO runs:
    /isaac-sim/python.sh isaaclab_arena_examples/relations/relation_solver_benchmark.py \\
      --suite regression --max-iters 200 --warmup 1 --repeat 5 \\
      --output-dir /results/solver-regression

  Run a candidate and compare it with a saved baseline:
    /isaac-sim/python.sh isaaclab_arena_examples/relations/relation_solver_benchmark.py \\
      --suite regression --max-iters 200 --warmup 1 --repeat 5 \\
      --baseline-json /results/baseline/benchmark.json --output-dir /results/candidate

  Replicate the full matrix on two GPUs:
    /isaac-sim/python.sh isaaclab_arena_examples/relations/relation_solver_benchmark.py \\
      --suite envs --targets solver --compare-modes --num-envs 1 --num-envs 8 --gpus 0,1

  Tune maximum batch size independently on each GPU:
    /isaac-sim/python.sh isaaclab_arena_examples/relations/relation_solver_benchmark.py \\
      --suite envs --targets solver --compare-modes --num-envs 1 --gpus 0,1 \\
      --capacity-search --capacity-max-envs 8192 --memory-headroom-gib 2

Results print to the terminal. Pass --output-dir PATH to also write JSON and CSV.
Solver status uses the final-loss threshold; placer status uses geometric layout validity.
Environment-spec generation by an LLM is not measured. Environment bring-up measures
make_registered() plus the first reset; graph-spec parsing, builder setup, and teardown
are outside the timed boundary.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import torch
from dataclasses import replace
from pathlib import Path
from typing import cast

from isaaclab_arena.relations.benchmark import (
    BenchmarkMeasurement,
    BenchmarkRun,
    BenchmarkScenario,
    BenchmarkTarget,
    CollisionModeName,
    build_distributed_run,
    build_run,
    compare_benchmark_runs,
    default_scenarios,
    env_count_sweep,
    format_diagnostic_markdown,
    format_memory_capacity_markdown,
    format_regression_markdown,
    format_results_table,
    format_scaling_summary,
    load_benchmark_run,
    object_count_sweep,
    requested_scenario_ids,
    run_benchmarks,
    run_capacity_search,
    run_multi_gpu,
    run_worker,
    scenarios_for_modes,
    validate_gpu_selectors,
    write_batch_scaling_svg,
    write_droid_kitchen_snapshot,
    write_droid_scene_snapshot,
    write_lightwheel_kitchen_snapshot,
    write_memory_capacity_csv,
    write_memory_scaling_svg,
    write_object_scaling_svg,
    write_regression_csv,
    write_regression_json,
    write_results_csv,
    write_results_json,
    write_robot_scaling_svg,
)

DEFAULT_BBOX_SPEC = "isaaclab_arena_environments/robolab/tasks/banana_in_bowl.yaml"
DEFAULT_MESH_SPEC = "isaaclab_arena_environments/kitchen_bench/replicator_kitchen_l_shape_banana_bowl.yaml"
KITCHEN_BENCH_DIR = Path("isaaclab_arena_environments/kitchen_bench")
KITCHEN_BENCH_SPEC_COUNT = 17
KITCHEN_BENCH_SEED_COUNT = 3
KITCHEN_BENCH_BBOX_SPECS = {
    "droid_open_fridge_lightwheel_kitchen.yaml",
    "droid_open_microwave_lightwheel_kitchen.yaml",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--suite",
        choices=("presets", "objects", "envs", "comprehensive", "diagnostic", "memory", "regression"),
        default="presets",
        help="Benchmark matrix; diagnostic prints a question-driven Markdown report.",
    )
    parser.add_argument(
        "--diagnostic-topic",
        choices=(
            "all",
            "batchification",
            "object-complexity",
            "background-collision",
            "robot-impact",
            "scene-difficulty",
        ),
        default="all",
        help="Run one diagnostic question instead of the complete diagnostic report.",
    )
    parser.add_argument(
        "--collision-mode",
        choices=("auto", "bbox", "mesh"),
        default="auto",
        help="Collision mode; auto uses bbox for synthetic scenes and each environment's natural mode.",
    )
    parser.add_argument("--compare-modes", action="store_true", help="Run both bbox and mesh scenarios.")
    parser.add_argument(
        "--targets",
        default=None,
        help="Comma-separated solver, placer, or environment (default: solver,placer; fixed by comprehensive).",
    )
    parser.add_argument("--environment-spec", action="append", help="Graph spec for an environment target.")
    parser.add_argument("--robot-mode", choices=("with", "without", "both"), default="both")
    parser.add_argument("--num-envs", type=int, action="append", help="Override suite environment counts.")
    parser.add_argument("--max-iters", type=int, default=600)
    parser.add_argument("--convergence-threshold", type=float, default=0.0)
    parser.add_argument("--num-spheres", type=int, default=30)
    parser.add_argument("--max-placement-attempts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--final-loss-threshold", type=float, default=1e-4)
    parser.add_argument("--min-valid-layout-rate", type=float, default=1.0)
    parser.add_argument(
        "--gpus",
        help=(
            "Comma-separated GPUs. Each runs an independent full matrix; solver batches are not split across GPUs, "
            "and aggregate throughput sums the replicated workers."
        ),
    )
    parser.add_argument(
        "--capacity-search",
        action="store_true",
        help="Find the largest viable num_envs independently on each selected GPU.",
    )
    parser.add_argument("--capacity-max-envs", type=int, default=8192, help="Upper bound for capacity search.")
    parser.add_argument(
        "--memory-headroom-gib",
        type=float,
        default=2.0,
        help="Required free GPU memory after each capacity probe.",
    )
    parser.add_argument("--output-dir", type=Path, help="Write JSON and CSV reports to this directory.")
    parser.add_argument("--baseline-json", type=Path, help="Compare a regression run with this benchmark.json.")
    parser.add_argument(
        "--maximum-regression-percent",
        type=float,
        default=10.0,
        help="Largest allowed iter/s decrease when --baseline-json is used (default: 10).",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Report detected performance regressions without returning a failing exit code.",
    )
    parser.add_argument("--worker-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--physical-gpu", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _targets(value: str) -> tuple[BenchmarkTarget, ...]:
    targets = tuple(target.strip() for target in value.split(","))
    valid = {"solver", "placer", "environment"}
    if not targets or any(target not in valid for target in targets) or len(targets) != len(set(targets)):
        raise ValueError(f"targets must be comma-separated values from {sorted(valid)}")
    return cast(tuple[BenchmarkTarget, ...], targets)


def _validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.maximum_regression_percent) or args.maximum_regression_percent < 0.0:
        raise ValueError("--maximum-regression-percent must be finite and non-negative")
    if args.suite != "regression" and (args.baseline_json is not None or args.report_only):
        raise ValueError("--baseline-json and --report-only require --suite regression")
    if args.report_only and args.baseline_json is None:
        raise ValueError("--report-only requires --baseline-json")
    if (
        args.baseline_json is not None
        and args.output_dir is not None
        and args.baseline_json.resolve() == (args.output_dir / "benchmark.json").resolve()
    ):
        raise ValueError("--baseline-json must not be overwritten by the candidate --output-dir")
    if args.suite == "regression":
        incompatible = (
            ("--gpus", args.gpus is not None),
            ("--capacity-search", args.capacity_search),
            ("--targets", args.targets is not None),
            ("--environment-spec", args.environment_spec is not None),
            ("--collision-mode", args.collision_mode != "auto"),
            ("--compare-modes", args.compare_modes),
            ("--robot-mode", args.robot_mode != "both"),
            ("--num-envs", args.num_envs is not None),
            ("--diagnostic-topic", args.diagnostic_topic != "all"),
        )
        rejected = [option for option, supplied in incompatible if supplied]
        if rejected:
            raise ValueError(f"--suite regression owns its execution matrix; cannot combine with {', '.join(rejected)}")
    if args.suite == "memory":
        incompatible = (
            ("--capacity-search", args.capacity_search),
            ("--targets", args.targets is not None),
            ("--environment-spec", args.environment_spec is not None),
            ("--collision-mode", args.collision_mode != "auto"),
            ("--compare-modes", args.compare_modes),
            ("--robot-mode", args.robot_mode != "both"),
            ("--num-envs", args.num_envs is not None),
            ("--diagnostic-topic", args.diagnostic_topic != "all"),
        )
        rejected = [option for option, supplied in incompatible if supplied]
        if rejected:
            raise ValueError(f"--suite memory owns its execution matrix; cannot combine with {', '.join(rejected)}")
        if args.gpus is not None and "," in args.gpus:
            raise ValueError("--suite memory accepts exactly one GPU")
    if args.suite in ("comprehensive", "diagnostic"):
        incompatible = (
            ("--gpus", args.gpus is not None),
            ("--capacity-search", args.capacity_search),
            ("--targets", args.targets is not None),
            ("--environment-spec", args.environment_spec is not None),
            ("--collision-mode", args.collision_mode != "auto"),
            ("--compare-modes", args.compare_modes),
            ("--robot-mode", args.robot_mode != "both"),
            ("--num-envs", args.num_envs is not None),
            ("--capacity-max-envs", args.capacity_max_envs != 8192),
            ("--memory-headroom-gib", args.memory_headroom_gib != 2.0),
            ("--diagnostic-topic", args.suite == "comprehensive" and args.diagnostic_topic != "all"),
        )
        rejected = [option for option, supplied in incompatible if supplied]
        if rejected:
            raise ValueError(
                f"--suite {args.suite} owns its execution matrix; cannot combine with {', '.join(rejected)}"
            )
    elif args.diagnostic_topic != "all":
        raise ValueError("--diagnostic-topic requires --suite diagnostic")
    if args.compare_modes and args.collision_mode != "auto":
        raise ValueError("--compare-modes cannot be combined with --collision-mode")
    if args.num_envs and (any(value <= 0 for value in args.num_envs) or len(args.num_envs) != len(set(args.num_envs))):
        raise ValueError("--num-envs values must be positive and unique")
    if args.capacity_max_envs <= 0:
        raise ValueError("--capacity-max-envs must be positive")
    if not math.isfinite(args.memory_headroom_gib) or args.memory_headroom_gib < 0.0:
        raise ValueError("--memory-headroom-gib must be finite and non-negative")


def _collision_modes(args: argparse.Namespace) -> tuple[CollisionModeName, ...]:
    if args.compare_modes:
        return ("bbox", "mesh")
    return (cast(CollisionModeName, "bbox" if args.collision_mode == "auto" else args.collision_mode),)


def _apply_scenario_options(
    scenarios: tuple[BenchmarkScenario, ...],
    args: argparse.Namespace,
) -> tuple[BenchmarkScenario, ...]:
    """Apply shared command-line options to benchmark scenarios."""
    return tuple(
        replace(
            scenario,
            max_iters=args.max_iters,
            convergence_threshold=args.convergence_threshold,
            num_spheres=args.num_spheres,
            max_placement_attempts=args.max_placement_attempts,
            placement_seed=args.seed,
            warmup_runs=args.warmup,
            timed_runs=args.repeat,
            final_loss_threshold=args.final_loss_threshold,
            min_valid_layout_rate=args.min_valid_layout_rate,
        )
        for scenario in scenarios
    )


def _base_scenarios(args: argparse.Namespace) -> tuple[BenchmarkScenario, ...]:
    if args.suite == "objects":
        scenarios = object_count_sweep()
    elif args.suite == "envs":
        scenarios = env_count_sweep(env_counts=tuple(args.num_envs) if args.num_envs else (1, 8, 32, 128))
    else:
        scenarios = default_scenarios()
    if args.num_envs and args.suite != "envs":
        scenarios = tuple(replace(scenario, num_envs=num_envs) for scenario in scenarios for num_envs in args.num_envs)
    return _apply_scenario_options(scenarios, args)


def _build_environment_scenarios(
    args: argparse.Namespace,
    spec_modes: tuple[tuple[str, tuple[CollisionModeName, ...]], ...],
    robot_modes: tuple[bool, ...],
    env_counts: tuple[int, ...],
) -> tuple[BenchmarkScenario, ...]:
    """Build environment scenarios from an explicit execution matrix."""
    scenarios = []
    for spec, modes in spec_modes:
        for mode in modes:
            for include_robot in robot_modes:
                for num_envs in env_counts:
                    scenarios.append(
                        BenchmarkScenario(
                            name=f"{Path(spec).stem}-{'robot' if include_robot else 'no-robot'}",
                            num_objects=0,
                            num_envs=num_envs,
                            collision_mode=mode,
                            graph_spec_path=spec,
                            include_robot=include_robot,
                        )
                    )
    return _apply_scenario_options(tuple(scenarios), args)


def _environment_scenarios(args: argparse.Namespace) -> tuple[BenchmarkScenario, ...]:
    spec_modes: tuple[tuple[str, tuple[CollisionModeName, ...]], ...]
    if args.environment_spec:
        if args.collision_mode == "auto" and not args.compare_modes:
            raise ValueError("custom --environment-spec requires --collision-mode or --compare-modes")
        modes = _collision_modes(args)
        spec_modes = tuple((spec, modes) for spec in args.environment_spec)
    elif args.collision_mode == "auto" and not args.compare_modes:
        spec_modes = (
            (DEFAULT_BBOX_SPEC, ("bbox",)),
            (DEFAULT_MESH_SPEC, ("mesh",)),
        )
    else:
        modes = _collision_modes(args)
        spec_modes = (
            (DEFAULT_BBOX_SPEC, modes),
            (DEFAULT_MESH_SPEC, modes),
        )
    robot_modes = (True, False) if args.robot_mode == "both" else (args.robot_mode == "with",)
    env_counts = tuple(args.num_envs or [1, 8, 32])
    return _build_environment_scenarios(args, spec_modes, robot_modes, env_counts)


def _comprehensive_scenario_groups(
    args: argparse.Namespace,
) -> tuple[tuple[BenchmarkScenario, ...], tuple[BenchmarkScenario, ...]]:
    """Build the fixed synthetic and environment scenario groups."""
    synthetic_base = object_count_sweep(num_envs=1, counts=(3, 5, 10)) + env_count_sweep(
        num_objects=3,
        env_counts=(8, 32, 128),
    )
    synthetic = scenarios_for_modes(
        _apply_scenario_options(synthetic_base, args),
        ("bbox", "mesh"),
    )
    environment = _build_environment_scenarios(
        args,
        (
            (DEFAULT_BBOX_SPEC, ("bbox", "mesh")),
            (DEFAULT_MESH_SPEC, ("bbox", "mesh")),
        ),
        (True, False),
        (1,),
    )
    return synthetic, environment


def _memory_scenarios(args: argparse.Namespace) -> tuple[BenchmarkScenario, ...]:
    """Build matched table and kitchen workloads for isolated capacity probes."""
    return tuple(
        BenchmarkScenario(
            name=f"memory-{scene}",
            num_objects=6,
            num_envs=1,
            collision_mode=mode,
            max_iters=2,
            convergence_threshold=0.0,
            num_spheres=args.num_spheres,
            placement_seed=args.seed,
            max_placement_attempts=1,
            warmup_runs=0,
            timed_runs=1,
            final_loss_threshold=1e9,
            background_treatment="mesh" if scene == "kitchen" and mode == "mesh" else "none",
            scene_label="DROID Maple table" if scene == "table" else "Lightwheel RoboCasa kitchen",
            asset_set_name=f"memory-{scene}",
        )
        for scene in ("table", "kitchen")
        for mode in ("bbox", "mesh")
    )


def _regression_scenarios(args: argparse.Namespace) -> tuple[BenchmarkScenario, ...]:
    """Build the stable solver-performance regression matrix."""
    workloads = (
        ("regression-small", 6, 1),
        ("regression-batch-heavy", 6, 1024),
        ("regression-pair-heavy", 21, 1),
        ("regression-combined", 21, 256),
    )
    scenarios = tuple(
        BenchmarkScenario(
            name=name,
            num_objects=num_objects,
            num_envs=num_envs,
            scene_label="Synthetic table",
            asset_set_name="regression-synthetic",
        )
        for name, num_objects, num_envs in workloads
    )
    configured = tuple(
        replace(scenario, final_loss_threshold=1e9) for scenario in _apply_scenario_options(scenarios, args)
    )
    return scenarios_for_modes(configured, ("bbox", "mesh"))


def _diagnostic_scenario_groups(
    args: argparse.Namespace,
) -> tuple[tuple[BenchmarkScenario, ...], tuple[BenchmarkScenario, ...]]:
    """Build synthetic and graph-placement diagnostic matrices."""
    synthetic = []
    for mode in ("bbox", "mesh"):
        synthetic.extend(
            replace(
                scenario,
                diagnostic_topic="batchification",
                scene_label="DROID homogeneous table",
                asset_set_name="droid-homogeneous",
            )
            for scenario in env_count_sweep(
                num_objects=6,
                env_counts=(1, 8, 32, 128, 256),
                collision_mode=mode,
            )
        )
        synthetic.extend(
            BenchmarkScenario(
                name=f"robot-impact-{count - 1}-objects-{'with' if include_robot else 'without'}",
                num_objects=count,
                num_envs=1,
                collision_mode=mode,
                include_robot=include_robot,
                diagnostic_topic="robot-impact",
                background_treatment="scene-default",
                scene_label="Lightwheel RoboCasa kitchen",
                asset_set_name="lightwheel-kitchen-counter",
            )
            for count in (2, 3, 4, 6, 11, 16, 21)
            for include_robot in (False, True)
        )
        synthetic.extend(
            replace(
                scenario,
                diagnostic_topic="object-complexity",
                background_treatment="scene-default",
                scene_label="Lightwheel RoboCasa kitchen",
                asset_set_name="lightwheel-kitchen-counter",
                include_robot=False,
            )
            for scenario in object_count_sweep(
                num_envs=1,
                counts=(2, 3, 4, 6, 11, 16, 21),
                collision_mode=mode,
            )
        )
        backgrounds = ("none", "aabb") if mode == "bbox" else ("none", "mesh")
        synthetic.extend(
            BenchmarkScenario(
                name=f"background-{background}",
                num_objects=3,
                num_envs=1,
                collision_mode=mode,
                diagnostic_topic="background-collision",
                background_treatment=background,
                scene_label="toy",
            )
            for background in backgrounds
        )
    kitchen_specs = tuple(sorted(KITCHEN_BENCH_DIR.glob("*.yaml")))
    spec_count = len(kitchen_specs)
    spec_count_error = f"expected {KITCHEN_BENCH_SPEC_COUNT} kitchen benchmark specs, found {spec_count}"
    assert spec_count == KITCHEN_BENCH_SPEC_COUNT, spec_count_error
    graph = [
        BenchmarkScenario(
            name=f"scene-difficulty-{spec.stem}",
            num_objects=0,
            num_envs=1,
            collision_mode="bbox" if spec.name in KITCHEN_BENCH_BBOX_SPECS else "mesh",
            graph_spec_path=str(spec),
            include_robot=True,
            diagnostic_topic="scene-difficulty",
            background_treatment="scene-default",
            scene_label=spec.name,
        )
        for spec in kitchen_specs
    ]
    if args.diagnostic_topic != "all":
        synthetic = [scenario for scenario in synthetic if scenario.diagnostic_topic == args.diagnostic_topic]
        graph = [scenario for scenario in graph if scenario.diagnostic_topic == args.diagnostic_topic]
    configured_graph = tuple(
        replace(scenario, timed_runs=KITCHEN_BENCH_SEED_COUNT)
        for scenario in _apply_scenario_options(tuple(graph), args)
    )
    return _apply_scenario_options(tuple(synthetic), args), configured_graph


def _scenarios(args: argparse.Namespace, targets: tuple[BenchmarkTarget, ...]) -> tuple[BenchmarkScenario, ...]:
    if "environment" in targets:
        if targets != ("environment",):
            raise ValueError("environment must be run separately from synthetic solver/placer targets")
        return _environment_scenarios(args)
    return scenarios_for_modes(_base_scenarios(args), _collision_modes(args))


def _write_report(output_dir: Path, run: BenchmarkRun) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_results_json(output_dir / "benchmark.json", run)
    write_results_csv(output_dir / "benchmark.csv", run)


def _finish_run(args: argparse.Namespace, rows: list[BenchmarkMeasurement], run: BenchmarkRun) -> int:
    """Print benchmark results and write requested reports."""
    if args.suite == "diagnostic":
        markdown = format_diagnostic_markdown(run)
        print(markdown)
        if args.output_dir is not None:
            _write_report(args.output_dir, run)
            if any(row.diagnostic_topic == "batchification" for row in rows):
                write_batch_scaling_svg(args.output_dir / "batch_scaling.svg", rows)
                write_droid_scene_snapshot(args.output_dir / "table_scene.png")
            if any(row.diagnostic_topic == "object-complexity" for row in rows):
                write_object_scaling_svg(args.output_dir / "object_scaling.svg", rows)
                write_lightwheel_kitchen_snapshot(args.output_dir / "kitchen_scene.png")
            if any(row.diagnostic_topic == "robot-impact" for row in rows):
                write_robot_scaling_svg(args.output_dir / "robot_scaling.svg", rows)
                write_droid_kitchen_snapshot(args.output_dir / "robot_scene.png")
            (args.output_dir / "diagnostic.md").write_text(markdown + "\n", encoding="utf-8")
            print(f"\nReports written to: {args.output_dir.resolve()}")
        return 0 if run.succeeded else 1
    print("\n=== Relation Solver Benchmark Results ===")
    if any(row.target != "environment" for row in rows):
        print("solver: one seeded RelationSolver.solve call; status uses the worst layout's final loss.")
        print("placer: end-to-end ObjectPlacer.place with candidate generation, retries, and layout validation.")
        print("bbox/mesh: collision backends. Compare modes within the same target and workload.")
    print(format_results_table(rows))
    scaling_summary = format_scaling_summary(rows)
    if scaling_summary:
        print("\n=== Scaling Summary ===")
        print(scaling_summary)
    for worker_id, error in run.worker_errors.items():
        print(f"  worker {worker_id}: {error}")
    if run.missing_scenario_ids:
        print(f"  missing results: {', '.join(run.missing_scenario_ids)}")
    if args.output_dir is not None:
        _write_report(args.output_dir, run)
        print(f"\nReports written to: {args.output_dir.resolve()}")
    else:
        print("\nReports were not written. Pass --output-dir PATH to save JSON and CSV.")
    if args.suite != "regression" or args.baseline_json is None:
        return 0 if run.succeeded else 1
    try:
        baseline = load_benchmark_run(args.baseline_json)
        comparison = compare_benchmark_runs(
            baseline,
            run,
            maximum_regression_percent=args.maximum_regression_percent,
        )
    except (AssertionError, OSError, ValueError) as error:
        print(f"Unable to compare benchmark runs: {error}", file=sys.stderr)
        return 2
    markdown = format_regression_markdown(comparison)
    print("\n=== Baseline Comparison ===")
    print(markdown)
    if args.output_dir is not None:
        (args.output_dir / "regression.md").write_text(markdown + "\n", encoding="utf-8")
        write_regression_json(args.output_dir / "regression.json", comparison)
        write_regression_csv(args.output_dir / "regression.csv", comparison)
    return 0 if run.succeeded and (comparison.passed or args.report_only) else 1


def _run_memory_suite(
    args: argparse.Namespace,
    scenarios: tuple[BenchmarkScenario, ...],
    gpu: str,
) -> int:
    """Run isolated capacity probes and write the memory diagnostic."""
    results = [
        run_capacity_search(
            scenario,
            "solver",
            gpu,
            Path(__file__).resolve(),
            max_num_envs=args.capacity_max_envs,
            memory_headroom_gib=args.memory_headroom_gib,
            require_success_status=False,
        )
        for scenario in scenarios
    ]
    markdown = format_memory_capacity_markdown(results)
    print(markdown)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "capacity.json").write_text(
            json.dumps({"results": results}, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "memory.md").write_text(markdown + "\n", encoding="utf-8")
        write_memory_capacity_csv(args.output_dir / "memory.csv", results)
        write_memory_scaling_svg(args.output_dir / "memory_scaling.svg", results)
        print(f"\nReports written to: {args.output_dir.resolve()}")
    return 0 if all(result["max_num_envs"] is not None for result in results) else 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _validate_args(args)
        targets = _targets(args.targets or "solver,placer")
        if args.worker_input is not None:
            assert args.worker_output is not None
            return run_worker(args.worker_input, args.worker_output, args.physical_gpu, targets)
        if args.suite == "memory":
            targets = ("solver",)
            scenarios = _memory_scenarios(args)
            gpus = (args.gpus or "0",)
        elif args.suite == "regression":
            targets = ("solver",)
            scenarios = _regression_scenarios(args)
            requested_scenario_ids(scenarios, targets)
            gpus = ()
        elif args.suite in ("comprehensive", "diagnostic"):
            if not torch.cuda.is_available():
                raise ValueError(f"--suite {args.suite} requires a CUDA GPU")
            if args.suite == "comprehensive":
                synthetic_scenarios, environment_scenarios = _comprehensive_scenario_groups(args)
                synthetic_targets: tuple[BenchmarkTarget, ...] = ("solver", "placer")
                environment_targets: tuple[BenchmarkTarget, ...] = ("environment",)
            else:
                synthetic_scenarios, environment_scenarios = _diagnostic_scenario_groups(args)
                synthetic_targets = ("solver",)
                environment_targets = ("placer",)
            expected = requested_scenario_ids(synthetic_scenarios, synthetic_targets)
            expected += requested_scenario_ids(environment_scenarios, environment_targets)
            if len(expected) != len(set(expected)):
                raise ValueError(f"{args.suite} benchmark scenario IDs must be unique")
            gpus = ()
        else:
            scenarios = _scenarios(args, targets)
            requested_scenario_ids(scenarios, targets)
            gpus = tuple(gpu.strip() for gpu in args.gpus.split(",")) if args.gpus else ()
        if any(not gpu for gpu in gpus) or len(gpus) != len(set(gpus)):
            raise ValueError("--gpus must contain unique, non-empty device selectors")
        validate_gpu_selectors(gpus)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    if args.suite == "memory":
        return _run_memory_suite(args, scenarios, gpus[0])

    if args.capacity_search:
        if not gpus:
            print("--capacity-search requires --gpus", file=sys.stderr)
            return 2
        capacity_results = [
            run_capacity_search(
                scenario,
                target,
                gpu,
                Path(__file__).resolve(),
                max_num_envs=args.capacity_max_envs,
                memory_headroom_gib=args.memory_headroom_gib,
            )
            for gpu in gpus
            for scenario in scenarios
            for target in targets
        ]
        capacity_report = json.dumps({"results": capacity_results}, indent=2, allow_nan=False)
        print("\n=== Capacity Search Results ===")
        print(capacity_report)
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            capacity_path = args.output_dir / "capacity.json"
            capacity_path.write_text(capacity_report + "\n", encoding="utf-8")
            print(f"\nReport written to: {capacity_path.resolve()}")
        return 0 if all(result["max_num_envs"] is not None for result in capacity_results) else 1

    if args.suite in ("comprehensive", "diagnostic"):
        from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

        with SimulationAppContext(argparse.Namespace(headless=True)):
            rows = run_benchmarks(synthetic_scenarios, targets=synthetic_targets)
            rows.extend(run_benchmarks(environment_scenarios, targets=environment_targets))
            run = build_distributed_run(rows, {"local": expected}, {"local": 0})
            return _finish_run(args, rows, run)
    elif gpus:
        rows, assignments, exit_codes, worker_errors = run_multi_gpu(
            scenarios,
            targets,
            gpus,
            Path(__file__).resolve(),
        )
        run = build_distributed_run(rows, assignments, exit_codes, worker_errors)
    elif "environment" in targets:
        from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

        with SimulationAppContext(argparse.Namespace(headless=True)):
            rows = run_benchmarks(scenarios, targets=targets)
            assignments = {"local": requested_scenario_ids(scenarios, targets)}
            run = build_run(scenarios, targets, rows, assignments, {"local": 0})
            return _finish_run(args, rows, run)
    else:
        rows = run_benchmarks(scenarios, targets=targets)
        assignments = {"local": requested_scenario_ids(scenarios, targets)}
        run = build_run(scenarios, targets, rows, assignments, {"local": 0})
    return _finish_run(args, rows, run)


if __name__ == "__main__":
    raise SystemExit(main())
