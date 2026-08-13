# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU benchmark orchestration and capacity probing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, replace
from pathlib import Path

from isaaclab_arena.relations.benchmark.models import BenchmarkMeasurement, BenchmarkScenario, BenchmarkTarget
from isaaclab_arena.relations.benchmark.reporting import requested_scenario_ids
from isaaclab_arena.relations.benchmark.synthetic_benchmark import run_benchmarks


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


def validate_gpu_selectors(gpus: tuple[str, ...]) -> None:
    """Require selectors to identify distinct available physical GPUs."""
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


def run_worker(
    input_path: Path,
    output_path: Path,
    physical_gpu: str | None,
    targets: tuple[BenchmarkTarget, ...],
) -> int:
    """Run a hidden benchmark worker and write its measurements."""
    import torch

    if physical_gpu is not None and not torch.cuda.is_available():
        raise RuntimeError(f"GPU {physical_gpu} is not available inside the benchmark worker")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    scenarios = tuple(BenchmarkScenario(**scenario) for scenario in payload["scenarios"])
    os.environ["ARENA_BENCHMARK_PHYSICAL_GPU"] = physical_gpu or ""

    def run_and_write() -> None:
        rows = run_benchmarks(scenarios, targets=targets)
        output_path.write_text(
            json.dumps([row.to_dict() for row in rows], indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    if "environment" in targets:
        import argparse

        from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

        with SimulationAppContext(argparse.Namespace(headless=True)):
            run_and_write()
    else:
        run_and_write()
    return 0


def _launch_worker(
    scenarios: tuple[BenchmarkScenario, ...],
    targets: tuple[BenchmarkTarget, ...],
    physical_gpu: str,
    directory: Path,
    worker_id: str,
    script_path: Path,
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
        str(script_path),
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


def _terminate_running_workers(processes: Iterable[subprocess.Popen]) -> None:
    """Terminate and reap workers that have not exited."""
    processes = tuple(processes)
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            process.wait()


def run_multi_gpu(
    scenarios: tuple[BenchmarkScenario, ...],
    targets: tuple[BenchmarkTarget, ...],
    gpus: tuple[str, ...],
    script_path: Path,
) -> tuple[list[BenchmarkMeasurement], dict[str, tuple[str, ...]], dict[str, int], dict[str, str]]:
    """Replicate a benchmark matrix across independent GPU workers."""
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
        try:
            for worker_id, gpu in zip(worker_ids, gpus, strict=True):
                workers[worker_id] = _launch_worker(scenarios, targets, gpu, directory, worker_id, script_path)
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
                    worker_errors[worker_id] = (
                        f"worker result IDs do not match its {len(base_scenario_ids)} assigned cases"
                    )
                    continue
                worker_rows.extend((worker_id, row) for row in worker_results)
        finally:
            _terminate_running_workers(process for process, _ in workers.values())
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


def run_capacity_search(
    scenario: BenchmarkScenario,
    target: BenchmarkTarget,
    gpu: str,
    script_path: Path,
    *,
    max_num_envs: int,
    memory_headroom_gib: float,
) -> dict[str, object]:
    """Probe one workload's maximum viable environment batch on one GPU."""
    probes: list[dict[str, object]] = []
    memory_headroom_bytes = int(memory_headroom_gib * 1024**3)

    def probe(num_envs: int) -> bool:
        probe_scenario = replace(scenario, num_envs=num_envs)
        with tempfile.TemporaryDirectory(prefix="arena-solver-capacity-") as temp_dir:
            process, output_path = _launch_worker(
                (probe_scenario,),
                (target,),
                gpu,
                Path(temp_dir),
                f"gpu-{gpu}-envs-{num_envs}",
                script_path,
            )
            try:
                exit_code = process.wait()
            finally:
                _terminate_running_workers((process,))
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
            observed_free = minimum_free if minimum_free is not None else free_after
            memory_ok = observed_free is not None and observed_free >= memory_headroom_bytes
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

    maximum = search_capacity(probe, max_num_envs=max_num_envs)
    return {
        "gpu": gpu,
        "scenario": scenario.name,
        "target": target,
        "collision_mode": scenario.collision_mode,
        "max_num_envs": maximum,
        "probes": probes,
    }
