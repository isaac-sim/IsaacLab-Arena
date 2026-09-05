# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark time from a valid environment spec to a ready placement pool.

SimulationApp starts before the measured region. Each sample includes loading the
fixed spec, constructing Arena assets, preparing collision geometry, solving
relations, validating layouts, and filling the placement pool. It excludes
Isaac Sim startup, environment instantiation, and physics settling.

Usage::

    /isaac-sim/python.sh \
        isaaclab_arena_examples/agentic_environment_generation/benchmark_time_to_resolved_layout.py \
        --env_spec path/to/environment.yaml --num_envs 1 --num_envs 16 \
        --num_runs 3 --output_path outputs/time_to_resolved_layout.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleResult:
    """One valid-spec-to-layout-pool measurement."""

    num_envs: int
    run_index: int
    requested_layouts: int
    time_to_resolved_layout_ms: float | None
    error: str | None


def percentile(values: list[float], p: float) -> float | None:
    """Return the nearest-rank percentile, or None when values is empty."""
    assert 0 <= p <= 100, f"percentile must be in [0, 100], got {p}"
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(p / 100 * len(ordered))) - 1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env_spec", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, action="append", required=True)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--placement_seed", type=int, default=42)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Continue from samples already in --output_path.")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    assert args.env_spec.is_file(), f"environment spec not found: {args.env_spec}"
    assert all(value > 0 for value in args.num_envs), "--num_envs values must be positive"
    assert args.num_runs > 0, "--num_runs must be positive"
    assert args.warmup_runs >= 0, "--warmup_runs must be non-negative"


def _run_sample(env_spec: Path, num_envs: int, run_index: int, placement_seed: int) -> SampleResult:
    import torch

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    started_at = time.perf_counter()
    try:
        graph_spec = ArenaEnvGraphSpec.from_yaml(env_spec)
        arena_env = graph_spec.to_arena_env()
        if arena_env.placer_params is None:
            arena_env.placer_params = ObjectPlacerParams()
        arena_env.placer_params.allow_best_loss_fallbacks = False
        builder = ArenaEnvBuilder(
            arena_env,
            ArenaEnvBuilderCfg(
                num_envs=num_envs,
                placement_seed=placement_seed,
            ),
        )
        requested_layouts = num_envs * arena_env.placer_params.min_unique_layouts_per_env
        builder._solve_relations()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started_at) * 1e3
        return SampleResult(
            num_envs=num_envs,
            run_index=run_index,
            requested_layouts=requested_layouts,
            time_to_resolved_layout_ms=elapsed_ms,
            error=None,
        )
    except Exception as exc:
        return SampleResult(
            num_envs=num_envs,
            run_index=run_index,
            requested_layouts=0,
            time_to_resolved_layout_ms=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _summarize(results: list[SampleResult]) -> dict[str, object]:
    latencies = [
        result.time_to_resolved_layout_ms for result in results if result.time_to_resolved_layout_ms is not None
    ]
    return {
        "requested_samples": len(results),
        "successful_samples": len(latencies),
        "failed_samples": len(results) - len(latencies),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
    }


def _output_payload(
    args: argparse.Namespace,
    results_by_num_envs: dict[str, object],
) -> dict[str, object]:
    return {
        "type": "timing_benchmark",
        "name": "time_to_resolved_layout_pool",
        "definition": "valid environment spec to ready pool of strictly valid layouts",
        "unit": "ms",
        "simulation_app_startup_included": False,
        "environment_instantiation_included": False,
        "physics_settling_included": False,
        "env_spec": str(args.env_spec),
        "env_spec_sha256": hashlib.sha256(args.env_spec.read_bytes()).hexdigest(),
        "placement_seed": args.placement_seed,
        "warmup_runs": args.warmup_runs,
        "requested_num_runs": args.num_runs,
        "results_by_num_envs": results_by_num_envs,
    }


def _write_checkpoint(args: argparse.Namespace, results_by_num_envs: dict[str, object]) -> None:
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output = _output_payload(args, results_by_num_envs)
    args.output_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_checkpoint(args: argparse.Namespace) -> dict[str, object]:
    if not args.resume or not args.output_path.is_file():
        return {}
    output = json.loads(args.output_path.read_text(encoding="utf-8"))
    assert output["env_spec"] == str(args.env_spec), "checkpoint environment spec does not match"
    expected_spec_sha256 = hashlib.sha256(args.env_spec.read_bytes()).hexdigest()
    assert output.get("env_spec_sha256") == expected_spec_sha256, "checkpoint environment spec contents do not match"
    assert output["placement_seed"] == args.placement_seed, "checkpoint placement seed does not match"
    assert output["warmup_runs"] == args.warmup_runs, "checkpoint warmup count does not match"
    assert output.get("requested_num_runs") == args.num_runs, "checkpoint requested sample count does not match"
    for measurement in output["results_by_num_envs"].values():
        assert len(measurement.get("samples", [])) <= args.num_runs, "checkpoint has more samples than requested"
    return output["results_by_num_envs"]


def main() -> int:
    """Run the valid-spec-to-layout-pool benchmark."""
    args = _parse_args()
    _validate_args(args)

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    results_by_num_envs = _load_checkpoint(args)
    failed_sample_count = 0
    with SimulationAppContext(argparse.Namespace(headless=True)):
        for num_envs in args.num_envs:
            checkpoint = results_by_num_envs.get(str(num_envs), {})
            raw_samples = checkpoint.get("samples", []) if isinstance(checkpoint, dict) else []
            samples = [SampleResult(**sample) for sample in raw_samples]
            if len(samples) >= args.num_runs:
                print(f"[benchmark] num_envs={num_envs}: already complete, skipping", flush=True)
                failed_sample_count += sum(sample.error is not None for sample in samples[: args.num_runs])
                continue

            if not samples:
                for warmup_index in range(args.warmup_runs):
                    warmup = _run_sample(args.env_spec, num_envs, -1 - warmup_index, args.placement_seed)
                    if warmup.error is not None:
                        print(f"[benchmark] num_envs={num_envs} warmup failed: {warmup.error}", flush=True)
                        break

            for run_index in range(len(samples), args.num_runs):
                samples.append(_run_sample(args.env_spec, num_envs, run_index, args.placement_seed))
                results_by_num_envs[str(num_envs)] = {
                    "summary": _summarize(samples),
                    "samples": [asdict(sample) for sample in samples],
                }
                _write_checkpoint(args, results_by_num_envs)

            summary = results_by_num_envs[str(num_envs)]["summary"]
            failed_sample_count += summary["failed_samples"]
            print(
                f"[benchmark] num_envs={num_envs}: "
                f"p50={summary['p50_ms']} ms, successes={summary['successful_samples']}/{args.num_runs}",
                flush=True,
            )

        _write_checkpoint(args, results_by_num_envs)
        print(f"[benchmark] wrote {args.output_path}", flush=True)
    return 1 if failed_sample_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
