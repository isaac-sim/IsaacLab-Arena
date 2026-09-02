# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run every manually defined time-to-first-spec benchmark case.

Usage::

    /isaac-sim/python.sh \
        isaaclab_arena_examples/agentic_environment_generation/run_time_to_first_spec_benchmarks.py \
        --inference_endpoint internal --num_runs 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from isaaclab_arena_examples.agentic_environment_generation.benchmark_time_to_first_spec import (
    DEFAULT_CASES_PATH,
    load_benchmark_cases,
)
from isaaclab_arena_examples.agentic_environment_generation.cli_runner import add_agent_inference_cli_args

_BENCHMARK_SCRIPT = Path(__file__).with_name("benchmark_time_to_first_spec.py")
_DEFAULT_OUTPUT_DIR = Path("output/time_to_first_spec")
_DEFAULT_SPEC_OUTPUT_DIR = Path("isaaclab_arena_environments/agentic_env_gen_benchmark")


def parse_args() -> argparse.Namespace:
    """Parse batch benchmark arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_agent_inference_cli_args(parser, include_prompt=False)
    parser.add_argument(
        "--cases_file",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help=f"Manually defined benchmark cases (default: {DEFAULT_CASES_PATH}).",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Measured requests per case (default: 3).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Per-case and combined JSON output directory (default: {_DEFAULT_OUTPUT_DIR}).",
    )
    spec_output = parser.add_mutually_exclusive_group()
    spec_output.add_argument(
        "--spec_output_dir",
        type=Path,
        default=_DEFAULT_SPEC_OUTPUT_DIR,
        help=f"Generated YAML root directory (default: {_DEFAULT_SPEC_OUTPUT_DIR}).",
    )
    spec_output.add_argument(
        "--no_save_specs",
        action="store_const",
        dest="spec_output_dir",
        const=None,
        help="Do not save a generated environment YAML for each benchmark case.",
    )
    return parser.parse_args()


def _benchmark_command(args: argparse.Namespace, case_name: str, output_path: Path) -> list[str]:
    """Build the child command for one case."""
    command = [
        sys.executable,
        str(_BENCHMARK_SCRIPT),
        "--cases_file",
        str(args.cases_file),
        "--case",
        case_name,
        "--num_runs",
        str(args.num_runs),
        "--temperature",
        str(args.temperature),
        "--output_path",
        str(output_path),
    ]
    if args.inference_endpoint is not None:
        command.extend(("--inference_endpoint", args.inference_endpoint))
    if args.model is not None:
        command.extend(("--model", args.model))
    if args.spec_output_dir is not None:
        command.extend(("--spec_output_dir", str(args.spec_output_dir / case_name)))
    return command


def _print_summary(results: list[dict[str, Any]]) -> None:
    """Print one compact percentile row per completed case."""
    print("\nCombined time-to-first-spec results", flush=True)
    print(f"{'case':<45} {'ok':>7} {'fail':>7} {'p50 ms':>12} {'p95 ms':>12} {'p99 ms':>12}", flush=True)
    for result in results:
        summary = result["summary"]
        percentiles = [summary[name] for name in ("p50_ms", "p95_ms", "p99_ms")]
        formatted = ["n/a" if value is None else f"{value:.3f}" for value in percentiles]
        print(
            f"{result['case']:<45} {summary['successful_samples']:>7} {summary['failed_samples']:>7} "
            f"{formatted[0]:>12} {formatted[1]:>12} {formatted[2]:>12}",
            flush=True,
        )


def main() -> int:
    """Run every configured case and write ``all_results.json``."""
    args = parse_args()
    assert args.num_runs > 0, f"num_runs must be positive, got {args.num_runs}"
    cases, _ = load_benchmark_cases(args.cases_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    failed_cases: list[str] = []
    with tempfile.TemporaryDirectory(prefix="arena-time-to-first-spec-") as temporary_output_dir:
        for case_index, case_name in enumerate(cases, start=1):
            output_path = Path(temporary_output_dir) / f"{case_name}.json"
            print(f"\n[batch] case {case_index}/{len(cases)}: {case_name}", flush=True)
            completed = subprocess.run(_benchmark_command(args, case_name, output_path), check=False)
            if completed.returncode != 0 or not output_path.is_file():
                failed_cases.append(case_name)
                print(f"[batch] {case_name}: child exited {completed.returncode}", flush=True)
                continue
            results.append(json.loads(output_path.read_text(encoding="utf-8")))

    combined_path = args.output_dir / "all_results.json"
    combined = {
        "num_runs_per_case": args.num_runs,
        "cases_file": str(args.cases_file),
        "spec_output_dir": str(args.spec_output_dir) if args.spec_output_dir is not None else None,
        "failed_cases": failed_cases,
        "results": results,
    }
    combined_path.write_text(json.dumps(combined, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _print_summary(results)
    print(f"\n[batch] combined results: {combined_path}", flush=True)
    if failed_cases:
        print(f"[batch] failed cases: {', '.join(failed_cases)}", flush=True)
    return 1 if failed_cases else 0


if __name__ == "__main__":
    raise SystemExit(main())
