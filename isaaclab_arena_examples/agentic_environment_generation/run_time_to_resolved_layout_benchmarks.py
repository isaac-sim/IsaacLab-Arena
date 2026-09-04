# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the standard valid-spec-to-resolved-layout benchmark matrix.

Usage::

    /isaac-sim/python.sh \
        isaaclab_arena_examples/agentic_environment_generation/run_time_to_resolved_layout_benchmarks.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_BENCHMARK_SCRIPT = Path(__file__).with_name("benchmark_time_to_resolved_layout.py")
_BENCHMARK_ROOT = Path("isaaclab_arena_environments/agentic_env_gen_benchmark")
_DEFAULT_OUTPUT_DIR = Path("output/time_to_resolved_layout")
_DEFAULT_NUM_ENVS = (1, 16, 64, 256)

_CASES = {
    "tabletop_distractors_0": "tabletop_banana_plate_distractors_0/banana_to_plate_on_maple_table.yaml",
    "tabletop_distractors_6": "tabletop_banana_plate_distractors_6/banana_to_plate_with_fruit_distractors.yaml",
    "tabletop_distractors_14": "tabletop_banana_plate_distractors_14/droid_place_banana_on_plate.yaml",
    "kitchen_distractors_0": "kitchen_banana_plate_distractors_0/droid_banana_to_plate_on_counter.yaml",
    "kitchen_distractors_6": "kitchen_banana_plate_distractors_6/banana_to_plate_on_kitchen_counter.yaml",
    "kitchen_distractors_14": "kitchen_banana_plate_distractors_14/droid_place_banana_on_plate.yaml",
    "kitchen_open_fridge_door": "kitchen_open_fridge_door/droid_open_fridge.yaml",
    "tabletop_heterogeneous_fruit_plate": (
        "tabletop_heterogeneous_fruit_plate/droid_place_varied_fruit_on_plate.yaml"
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(_CASES),
        help="Case to run. Repeat to select multiple cases; defaults to all cases.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        action="append",
        help="Environment count. Repeat for multiple values; defaults to 1, 16, 64, 256.",
    )
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--placement_seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true", help="Resume incomplete per-configuration results.")
    return parser.parse_args()


def _command(
    args: argparse.Namespace,
    env_spec: Path,
    num_envs: int,
    output_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(_BENCHMARK_SCRIPT),
        "--env_spec",
        str(env_spec),
        "--num_envs",
        str(num_envs),
        "--num_runs",
        str(args.num_runs),
        "--warmup_runs",
        str(args.warmup_runs),
        "--placement_seed",
        str(args.placement_seed),
        "--output_path",
        str(output_path),
    ]
    if args.resume:
        command.append("--resume")
    return command


def _print_summary(results: list[dict[str, Any]]) -> None:
    print("\nValid spec to resolved layout pool", flush=True)
    print(
        f"{'case':<28} {'envs':>6} {'ok':>6} {'fail':>6} {'p50 ms':>12} {'p95 ms':>12} {'p99 ms':>12}",
        flush=True,
    )
    for result in results:
        num_envs, measurement = next(iter(result["results_by_num_envs"].items()))
        summary = measurement["summary"]
        values = [summary[name] for name in ("p50_ms", "p95_ms", "p99_ms")]
        formatted = ["n/a" if value is None else f"{value:.3f}" for value in values]
        print(
            f"{result['case']:<28} {num_envs:>6} "
            f"{summary['successful_samples']:>6} {summary['failed_samples']:>6} "
            f"{formatted[0]:>12} {formatted[1]:>12} {formatted[2]:>12}",
            flush=True,
        )


def main() -> int:
    """Run selected cases and environment counts, then combine their JSON results."""
    args = _parse_args()
    cases = args.case or list(_CASES)
    num_envs_values = args.num_envs or list(_DEFAULT_NUM_ENVS)
    assert args.num_runs > 0, "--num_runs must be positive"
    assert args.warmup_runs >= 0, "--warmup_runs must be non-negative"
    assert all(value > 0 for value in num_envs_values), "--num_envs values must be positive"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    failed_configurations: list[str] = []
    total = len(cases) * len(num_envs_values)
    configuration_index = 0

    for case_name in cases:
        env_spec = _BENCHMARK_ROOT / _CASES[case_name]
        assert env_spec.is_file(), f"benchmark environment spec not found: {env_spec}"
        for num_envs in num_envs_values:
            configuration_index += 1
            configuration_name = f"{case_name}_envs_{num_envs}"
            output_path = args.output_dir / f"{configuration_name}.json"
            print(
                f"\n[batch] configuration {configuration_index}/{total}: {configuration_name}",
                flush=True,
            )
            if args.resume and output_path.is_file():
                existing_result = json.loads(output_path.read_text(encoding="utf-8"))
                existing_measurement = existing_result.get("results_by_num_envs", {}).get(str(num_envs), {})
                existing_samples = existing_measurement.get("samples", [])
                if len(existing_samples) >= args.num_runs:
                    existing_result["case"] = case_name
                    existing_result["num_envs"] = num_envs
                    existing_result["child_returncode"] = 0
                    results.append(existing_result)
                    print(f"[batch] {configuration_name}: already complete, skipping", flush=True)
                    continue
            completed = subprocess.run(
                _command(args, env_spec, num_envs, output_path),
                check=False,
            )
            if not output_path.is_file():
                failed_configurations.append(configuration_name)
                print(f"[batch] {configuration_name}: child exited {completed.returncode}", flush=True)
                continue
            result = json.loads(output_path.read_text(encoding="utf-8"))
            result["case"] = case_name
            result["num_envs"] = num_envs
            result["child_returncode"] = completed.returncode
            results.append(result)
            if completed.returncode != 0:
                failed_configurations.append(configuration_name)

    combined_path = args.output_dir / "all_results.json"
    combined = {
        "num_runs_per_configuration": args.num_runs,
        "warmup_runs": args.warmup_runs,
        "placement_seed": args.placement_seed,
        "cases": cases,
        "num_envs": num_envs_values,
        "failed_configurations": failed_configurations,
        "results": results,
    }
    combined_path.write_text(json.dumps(combined, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _print_summary(results)
    print(f"\n[batch] combined results: {combined_path}", flush=True)
    if failed_configurations:
        print(f"[batch] failed configurations: {', '.join(failed_configurations)}", flush=True)
    return 1 if failed_configurations else 0


if __name__ == "__main__":
    raise SystemExit(main())
