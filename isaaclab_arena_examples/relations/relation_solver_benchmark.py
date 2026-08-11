# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run relation-solver benchmarks locally or with one worker per GPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import torch
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

from isaaclab_arena.relations.relation_solver_benchmark import (
    BenchmarkMeasurement,
    BenchmarkRun,
    BenchmarkScenario,
    BenchmarkTarget,
    CollisionModeName,
    build_distributed_run,
    build_run,
    default_scenarios,
    env_count_sweep,
    format_results_table,
    object_count_sweep,
    requested_scenario_ids,
    run_benchmarks,
    scenarios_for_modes,
    search_capacity,
    write_results_csv,
    write_results_json,
)

DEFAULT_BBOX_SPEC = "isaaclab_arena_environments/robolab/tasks/banana_in_bowl.yaml"
DEFAULT_MESH_SPEC = "isaaclab_arena_environments/kitchen_bench/replicator_kitchen_l_shape_banana_bowl.yaml"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("presets", "objects", "envs"), default="presets")
    parser.add_argument(
        "--collision-mode",
        choices=("auto", "bbox", "mesh"),
        default="auto",
        help="Collision mode; auto uses bbox for synthetic scenes and each environment's natural mode.",
    )
    parser.add_argument("--compare-modes", action="store_true", help="Run both bbox and mesh scenarios.")
    parser.add_argument("--targets", default="solver,placer", help="Comma-separated solver, placer, or environment.")
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
    parser.add_argument("--gpus", help="Comma-separated GPUs; each runs the full matrix for aggregate throughput.")
    parser.add_argument("--capacity-search", action="store_true")
    parser.add_argument("--capacity-max-envs", type=int, default=4096)
    parser.add_argument("--memory-headroom-gib", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, help="Write JSON and CSV reports to this directory.")
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
    if args.compare_modes and args.collision_mode != "auto":
        raise ValueError("--compare-modes cannot be combined with --collision-mode")
    if args.num_envs and (any(value <= 0 for value in args.num_envs) or len(args.num_envs) != len(set(args.num_envs))):
        raise ValueError("--num-envs values must be positive and unique")
    if args.max_iters <= 0:
        raise ValueError("--max-iters must be positive")
    if not math.isfinite(args.convergence_threshold) or args.convergence_threshold < 0.0:
        raise ValueError("--convergence-threshold must be finite and non-negative")
    if args.num_spheres <= 0:
        raise ValueError("--num-spheres must be positive")
    if args.max_placement_attempts <= 0:
        raise ValueError("--max-placement-attempts must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if not math.isfinite(args.final_loss_threshold) or args.final_loss_threshold < 0.0:
        raise ValueError("--final-loss-threshold must be finite and non-negative")
    if not math.isfinite(args.min_valid_layout_rate) or not 0.0 <= args.min_valid_layout_rate <= 1.0:
        raise ValueError("--min-valid-layout-rate must be in [0, 1]")
    if args.capacity_max_envs <= 0:
        raise ValueError("--capacity-max-envs must be positive")
    if not math.isfinite(args.memory_headroom_gib) or args.memory_headroom_gib < 0.0:
        raise ValueError("--memory-headroom-gib must be finite and non-negative")


def _validate_gpu_selectors(gpus: tuple[str, ...]) -> None:
    if not gpus:
        return
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"could not query available GPUs: {error}") from error
    selectors_to_uuid = {}
    for line in result.stdout.splitlines():
        index, uuid = (value.strip() for value in line.split(",", maxsplit=1))
        selectors_to_uuid[index] = uuid
        selectors_to_uuid[uuid] = uuid
    unknown = tuple(gpu for gpu in gpus if gpu not in selectors_to_uuid)
    if unknown:
        available = ", ".join(sorted(selectors_to_uuid))
        raise ValueError(f"unknown GPU selector(s): {', '.join(unknown)}; available: {available}")
    canonical_gpus = tuple(selectors_to_uuid[gpu] for gpu in gpus)
    if len(canonical_gpus) != len(set(canonical_gpus)):
        raise ValueError("--gpus selectors must refer to different physical devices")


def _collision_modes(args: argparse.Namespace) -> tuple[CollisionModeName, ...]:
    if args.compare_modes:
        return ("bbox", "mesh")
    return (cast(CollisionModeName, "bbox" if args.collision_mode == "auto" else args.collision_mode),)


def _base_scenarios(args: argparse.Namespace) -> tuple[BenchmarkScenario, ...]:
    if args.suite == "objects":
        scenarios = object_count_sweep(max_iters=args.max_iters)
    elif args.suite == "envs":
        scenarios = env_count_sweep(
            env_counts=tuple(args.num_envs) if args.num_envs else (1, 8, 32),
            max_iters=args.max_iters,
        )
    else:
        scenarios = default_scenarios()
    if args.num_envs and args.suite != "envs":
        scenarios = tuple(replace(scenario, num_envs=num_envs) for scenario in scenarios for num_envs in args.num_envs)
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


def _environment_scenarios(args: argparse.Namespace) -> tuple[BenchmarkScenario, ...]:
    if args.environment_spec:
        if args.collision_mode == "auto" and not args.compare_modes:
            raise ValueError("custom --environment-spec requires --collision-mode or --compare-modes")
        specs: tuple[tuple[str, CollisionModeName | None], ...] = tuple((spec, None) for spec in args.environment_spec)
    else:
        specs = ((DEFAULT_BBOX_SPEC, "bbox"), (DEFAULT_MESH_SPEC, "mesh"))
    robot_modes = (True, False) if args.robot_mode == "both" else (args.robot_mode == "with",)
    env_counts = tuple(args.num_envs or [1, 8, 32])
    scenarios = []
    for spec, natural_mode in specs:
        modes = (
            (natural_mode,)
            if natural_mode is not None and args.collision_mode == "auto" and not args.compare_modes
            else _collision_modes(args)
        )
        for mode in modes:
            for include_robot in robot_modes:
                for num_envs in env_counts:
                    scenarios.append(
                        BenchmarkScenario(
                            name=f"{Path(spec).stem}-{'robot' if include_robot else 'no-robot'}",
                            num_objects=0,
                            num_envs=num_envs,
                            max_iters=args.max_iters,
                            convergence_threshold=args.convergence_threshold,
                            collision_mode=mode,
                            num_spheres=args.num_spheres,
                            max_placement_attempts=args.max_placement_attempts,
                            placement_seed=args.seed,
                            warmup_runs=args.warmup,
                            timed_runs=args.repeat,
                            final_loss_threshold=args.final_loss_threshold,
                            min_valid_layout_rate=args.min_valid_layout_rate,
                            graph_spec_path=spec,
                            include_robot=include_robot,
                        )
                    )
    return tuple(scenarios)


def _scenarios(args: argparse.Namespace, targets: tuple[BenchmarkTarget, ...]) -> tuple[BenchmarkScenario, ...]:
    if "environment" in targets:
        if targets != ("environment",):
            raise ValueError("environment must be run separately from synthetic solver/placer targets")
        return _environment_scenarios(args)
    return scenarios_for_modes(_base_scenarios(args), _collision_modes(args))


def _run_rows(
    scenarios: tuple[BenchmarkScenario, ...],
    targets: tuple[BenchmarkTarget, ...],
) -> list[BenchmarkMeasurement]:
    if "environment" not in targets:
        return run_benchmarks(scenarios, targets=targets)
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        return run_benchmarks(scenarios, targets=targets)
    finally:
        app.close()


def _worker_main(args: argparse.Namespace, targets: tuple[BenchmarkTarget, ...]) -> int:
    assert args.worker_input is not None and args.worker_output is not None
    if args.physical_gpu is not None and not torch.cuda.is_available():
        raise RuntimeError(f"GPU {args.physical_gpu} is not available inside the benchmark worker")
    payload = json.loads(args.worker_input.read_text(encoding="utf-8"))
    scenarios = tuple(BenchmarkScenario(**scenario) for scenario in payload["scenarios"])
    os.environ["ARENA_BENCHMARK_PHYSICAL_GPU"] = args.physical_gpu or ""
    rows = _run_rows(scenarios, targets)
    args.worker_output.write_text(
        json.dumps([row.to_dict() for row in rows], indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


def _launch_worker(
    scenarios: tuple[BenchmarkScenario, ...],
    targets: tuple[BenchmarkTarget, ...],
    physical_gpu: str,
    directory: Path,
    worker_id: str,
) -> tuple[subprocess.Popen, Path]:
    input_path = directory / f"{worker_id}-input.json"
    output_path = directory / f"{worker_id}-output.json"
    input_path.write_text(
        json.dumps({"scenarios": [asdict(scenario) for scenario in scenarios]}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = physical_gpu
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--targets",
        ",".join(targets),
        "--worker-input",
        str(input_path),
        "--worker-output",
        str(output_path),
        "--physical-gpu",
        physical_gpu,
    ]
    return subprocess.Popen(command, env=env), output_path


def _worker_process_error(exit_code: int, output_path: Path) -> str | None:
    if exit_code != 0:
        return f"worker exited with code {exit_code}"
    if not output_path.is_file():
        return "worker exited successfully without writing its result file"
    return None


def _read_worker_results(output_path: Path) -> list[BenchmarkMeasurement]:
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    return [BenchmarkMeasurement.from_dict(row) for row in payload]


def _run_multi_gpu(
    scenarios: tuple[BenchmarkScenario, ...],
    targets: tuple[BenchmarkTarget, ...],
    gpus: tuple[str, ...],
) -> tuple[list[BenchmarkMeasurement], dict[str, tuple[str, ...]], dict[str, int], dict[str, str]]:
    worker_ids = tuple(f"gpu-{gpu}" for gpu in gpus)
    base_scenario_ids = requested_scenario_ids(scenarios, targets)
    assignments = {
        worker_id: tuple(f"{scenario_id}__{worker_id}" for scenario_id in base_scenario_ids) for worker_id in worker_ids
    }
    worker_rows: list[tuple[str, BenchmarkMeasurement]] = []
    exit_codes: dict[str, int] = {}
    worker_errors: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="arena-solver-benchmark-") as temp_dir:
        directory = Path(temp_dir)
        workers = {}
        for worker_id, gpu in zip(worker_ids, gpus, strict=True):
            workers[worker_id] = _launch_worker(scenarios, targets, gpu, directory, worker_id)
        for worker_id, (process, output_path) in workers.items():
            exit_code = process.wait()
            exit_codes[worker_id] = exit_code
            if process_error := _worker_process_error(exit_code, output_path):
                worker_errors[worker_id] = process_error
                continue
            try:
                worker_results = _read_worker_results(output_path)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                worker_errors[worker_id] = f"invalid worker result: {type(error).__name__}: {error}"
                continue
            result_ids = [row.scenario_id for row in worker_results]
            if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(base_scenario_ids):
                worker_errors[worker_id] = f"worker result IDs do not match its {len(base_scenario_ids)} assigned cases"
                continue
            worker_rows.extend((worker_id, row) for row in worker_results)
    aggregate_throughput: dict[str, float] = {}
    for scenario_id in base_scenario_ids:
        replicas = [row for _, row in worker_rows if row.scenario_id == scenario_id]
        if len(replicas) == len(worker_ids) and all(
            row.status == "ok" and row.throughput_envs_per_second is not None for row in replicas
        ):
            aggregate_throughput[scenario_id] = sum(
                row.throughput_envs_per_second for row in replicas if row.throughput_envs_per_second is not None
            )
    rows = [
        replace(
            row,
            scenario_id=f"{row.scenario_id}__{worker_id}",
            worker_id=worker_id,
            aggregate_throughput_envs_per_second=aggregate_throughput.get(row.scenario_id),
        )
        for worker_id, row in worker_rows
    ]
    return rows, assignments, exit_codes, worker_errors


def _run_capacity_search(
    scenario: BenchmarkScenario,
    target: BenchmarkTarget,
    gpu: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    probes: list[dict[str, object]] = []

    def probe(num_envs: int) -> bool:
        probe_scenario = replace(scenario, num_envs=num_envs)
        with tempfile.TemporaryDirectory(prefix="arena-solver-capacity-") as temp_dir:
            process, output_path = _launch_worker(
                (probe_scenario,),
                (target,),
                gpu,
                Path(temp_dir),
                f"gpu-{gpu}-envs-{num_envs}",
            )
            exit_code = process.wait()
            if process_error := _worker_process_error(exit_code, output_path):
                probes.append({
                    "num_envs": num_envs,
                    "viable": False,
                    "exit_code": exit_code,
                    "error": process_error,
                })
                return False
            try:
                [measurement] = _read_worker_results(output_path)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                probes.append({
                    "num_envs": num_envs,
                    "viable": False,
                    "exit_code": exit_code,
                    "error": f"invalid worker result: {type(error).__name__}: {error}",
                })
                return False
            minimum_free = measurement.device.minimum_free_memory_bytes
            free_after = measurement.device.free_memory_after_bytes
            headroom = int(args.memory_headroom_gib * 1024**3)
            observed_free = minimum_free if minimum_free is not None else free_after
            memory_ok = observed_free is not None and observed_free >= headroom
            viable = measurement.status == "ok" and measurement.device.name is not None and memory_ok
            probes.append({
                "num_envs": num_envs,
                "viable": viable,
                "status": measurement.status,
                "error": measurement.error,
                "free_memory_before_bytes": measurement.device.free_memory_before_bytes,
                "free_memory_after_bytes": free_after,
                "minimum_free_memory_bytes": minimum_free,
                "peak_reserved_bytes": measurement.peak_reserved_bytes,
            })
            return viable

    maximum = search_capacity(probe, max_num_envs=args.capacity_max_envs)
    return {
        "gpu": gpu,
        "scenario": scenario.name,
        "target": target,
        "collision_mode": scenario.collision_mode,
        "max_num_envs": maximum,
        "probes": probes,
    }


def _write_report(output_dir: Path, run: BenchmarkRun) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_results_json(output_dir / "benchmark.json", run)
    write_results_csv(output_dir / "benchmark.csv", run)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _validate_args(args)
        targets = _targets(args.targets)
        if args.worker_input is not None:
            return _worker_main(args, targets)
        scenarios = _scenarios(args, targets)
        requested_scenario_ids(scenarios, targets)
        gpus = tuple(gpu.strip() for gpu in args.gpus.split(",")) if args.gpus else ()
        if any(not gpu for gpu in gpus) or len(gpus) != len(set(gpus)):
            raise ValueError("--gpus must contain unique, non-empty device selectors")
        _validate_gpu_selectors(gpus)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    if args.capacity_search:
        if not gpus:
            print("--capacity-search requires --gpus", file=sys.stderr)
            return 2
        capacity_results = [
            _run_capacity_search(scenario, target, gpu, args)
            for gpu in gpus
            for scenario in scenarios
            for target in targets
        ]
        capacity_report = json.dumps({"schema_version": 1, "results": capacity_results}, indent=2, allow_nan=False)
        print("\n=== Capacity Search Results ===")
        print(capacity_report)
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            capacity_path = args.output_dir / "capacity.json"
            capacity_path.write_text(capacity_report + "\n", encoding="utf-8")
            print(f"\nReport written to: {capacity_path.resolve()}")
        return 0 if all(result["max_num_envs"] is not None for result in capacity_results) else 1

    if gpus:
        rows, assignments, exit_codes, worker_errors = _run_multi_gpu(scenarios, targets, gpus)
        run = build_distributed_run(rows, assignments, exit_codes, worker_errors)
    else:
        rows = _run_rows(scenarios, targets)
        assignments = {"local": requested_scenario_ids(scenarios, targets)}
        exit_codes = {"local": 0}
        run = build_run(scenarios, targets, rows, assignments, exit_codes)
    print("\n=== Relation Solver Benchmark Results ===")
    print(format_results_table(rows))
    for worker_id, error in run.worker_errors.items():
        print(f"  worker {worker_id}: {error}")
    if run.missing_scenario_ids:
        print(f"  missing results: {', '.join(run.missing_scenario_ids)}")
    if args.output_dir is not None:
        _write_report(args.output_dir, run)
        print(f"\nReports written to: {args.output_dir.resolve()}")
    else:
        print("\nReports were not written. Pass --output-dir PATH to save JSON and CSV.")
    return 0 if run.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
