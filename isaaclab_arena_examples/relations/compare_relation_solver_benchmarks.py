# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare baseline and candidate relation-solver benchmark artifacts.

The input files can come from local runs or from two OSMO jobs. Both jobs must
use the same regression-suite command, GPU model, seeds, and solver settings.

Example:
  /isaac-sim/python.sh isaaclab_arena_examples/relations/compare_relation_solver_benchmarks.py \\
    /results/baseline/benchmark.json /results/candidate/benchmark.json \\
    --output-dir /results/comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab_arena.relations.benchmark import (
    compare_benchmark_runs,
    format_regression_markdown,
    load_benchmark_run,
    write_regression_csv,
    write_regression_json,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path, help="Baseline benchmark.json.")
    parser.add_argument("candidate", type=Path, help="Candidate benchmark.json.")
    parser.add_argument(
        "--maximum-regression-percent",
        type=float,
        default=10.0,
        help="Largest allowed decrease in solver iter/s (default: 10).",
    )
    parser.add_argument("--output-dir", type=Path, help="Write regression.md, regression.json, and regression.csv.")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Return success after producing the report even when a regression is detected.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        baseline = load_benchmark_run(args.baseline)
        candidate = load_benchmark_run(args.candidate)
        comparison = compare_benchmark_runs(
            baseline,
            candidate,
            maximum_regression_percent=args.maximum_regression_percent,
        )
    except (AssertionError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    markdown = format_regression_markdown(comparison)
    print(markdown)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "regression.md").write_text(markdown + "\n", encoding="utf-8")
        write_regression_json(args.output_dir / "regression.json", comparison)
        write_regression_csv(args.output_dir / "regression.csv", comparison)
        print(f"\nReports written to: {args.output_dir.resolve()}")
    return 0 if comparison.passed or (args.report_only and comparison.correctness_passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
