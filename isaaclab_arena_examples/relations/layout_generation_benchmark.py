# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run and analyze the CoRL controlled layout-generation benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab_arena.relations.benchmark.layout_generation import (
    DEFAULT_TABLE_XY_BOUNDS,
    compare_layout_runs,
    format_layout_markdown,
    load_layout_run,
    make_object_sizes,
    run_layout_generation,
    write_layout_run,
)
from isaaclab_arena.relations.benchmark.layout_throughput import (
    format_throughput_markdown,
    load_throughput_run,
    run_controlled_throughput,
    write_throughput_run,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _add_geometry(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-objects", type=_positive_int, default=5)
    parser.add_argument("--object-size-m", type=_positive_float, default=0.12)
    parser.add_argument("--table-bounds", type=float, nargs=4, default=DEFAULT_TABLE_XY_BOUNDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)


def _add_generation_options(parser: argparse.ArgumentParser) -> None:
    _add_geometry(parser)
    parser.add_argument("--samples", type=_positive_int, default=100)
    parser.add_argument("--max-attempts", type=_positive_int, default=100)
    parser.add_argument("--max-iterations", type=_positive_int, default=600)
    parser.add_argument("--warmup", type=int, default=0)


def _add_throughput_options(parser: argparse.ArgumentParser) -> None:
    _add_geometry(parser)
    parser.add_argument("-k", "--target-layouts", type=_positive_int, action="append", required=True)
    parser.add_argument("--repetitions", type=_positive_int, default=3)
    parser.add_argument("--max-attempts-per-layout", type=_positive_int, default=100)
    parser.add_argument("--max-iterations", type=_positive_int, default=600)
    parser.add_argument("--warmup", type=int, default=0)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("arena", "random-rejection", "explicit", "arena-scene"):
        _add_generation_options(commands.add_parser(command))
    controlled = commands.add_parser("controlled-throughput")
    controlled.add_argument("--method", choices=("arena", "random_rejection", "explicit"), required=True)
    _add_throughput_options(controlled)
    _add_throughput_options(commands.add_parser("arena-throughput"))
    analyze = commands.add_parser("analyze")
    analyze.add_argument("inputs", type=Path, nargs="+")
    analyze.add_argument("--output", type=Path)
    analyze_throughput = commands.add_parser("analyze-throughput")
    analyze_throughput.add_argument("inputs", type=Path, nargs="+")
    analyze_throughput.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if getattr(args, "warmup", 0) < 0:
        parser.error("--warmup must be non-negative")
    if hasattr(args, "table_bounds"):
        xmin, xmax, ymin, ymax = args.table_bounds
        if not xmin < xmax or not ymin < ymax:
            parser.error("--table-bounds must be XMIN XMAX YMIN YMAX with positive extents")
    return args


def _write_markdown(json_path: Path, markdown: str) -> None:
    json_path.with_suffix(".md").write_text(markdown + "\n", encoding="utf-8")


def _run_generation(args: argparse.Namespace, method: str) -> int:
    run = run_layout_generation(
        method,
        sample_count=args.samples,
        master_seed=args.seed,
        warmup=args.warmup,
        table_xy_bounds=tuple(args.table_bounds),
        object_xy_sizes=make_object_sizes(args.num_objects, args.object_size_m),
        max_attempts=args.max_attempts,
        max_iterations=args.max_iterations,
        workload="controlled-tabletop" if args.command != "arena-scene" else "arena-scene",
    )
    write_layout_run(args.output, run)
    markdown = format_layout_markdown([run])
    _write_markdown(args.output, markdown)
    print(markdown)
    return 0


def _run_throughput(args: argparse.Namespace, method: str, workload: str = "controlled-tabletop") -> int:
    run = run_controlled_throughput(
        method,
        target_layout_counts=tuple(args.target_layouts),
        repetitions=args.repetitions,
        master_seed=args.seed,
        warmup=args.warmup,
        table_xy_bounds=tuple(args.table_bounds),
        object_xy_sizes=make_object_sizes(args.num_objects, args.object_size_m),
        max_attempts_per_layout=args.max_attempts_per_layout,
        max_iterations=args.max_iterations,
        workload=workload,
    )
    write_throughput_run(args.output, run)
    markdown = format_throughput_markdown([run])
    _write_markdown(args.output, markdown)
    print(markdown)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "analyze":
            runs = [load_layout_run(path) for path in args.inputs]
            compare_layout_runs(runs)
            markdown = format_layout_markdown(runs)
            if args.output:
                args.output.write_text(markdown + "\n", encoding="utf-8")
            print(markdown)
            return 0
        if args.command == "analyze-throughput":
            runs = [load_throughput_run(path) for path in args.inputs]
            markdown = format_throughput_markdown(runs)
            if args.output:
                args.output.write_text(markdown + "\n", encoding="utf-8")
            print(markdown)
            return 0
        if args.command == "random-rejection":
            return _run_generation(args, "random_rejection")
        if args.command == "explicit":
            return _run_generation(args, "explicit")
        if args.command == "controlled-throughput":
            return _run_throughput(args, args.method)

        from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

        with SimulationAppContext(argparse.Namespace(headless=True)):
            if args.command == "arena-throughput":
                return _run_throughput(args, "arena", "arena-scene")
            return _run_generation(args, "arena")
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
