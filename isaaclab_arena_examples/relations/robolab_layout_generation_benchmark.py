# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the supplementary RoboLab-style controlled tabletop throughput baseline.

This standalone script intentionally shares only the neutral XY geometry and
output schema. It does not claim direct equivalence to a relation-based scene.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0 or not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-k", "--target-layouts", type=_positive_int, action="append", required=True)
    parser.add_argument("--repetitions", type=_positive_int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-objects", type=_positive_int, default=5)
    parser.add_argument("--object-size-m", type=_positive_float, default=0.12)
    parser.add_argument("--table-bounds", type=float, nargs=4, default=(-0.5, 0.5, -0.5, 0.5))
    parser.add_argument("--max-attempts-per-layout", type=_positive_int, default=100)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    xmin, xmax, ymin, ymax = args.table_bounds
    if not all(math.isfinite(value) for value in args.table_bounds) or not xmin < xmax or not ymin < ymax:
        parser.error("--table-bounds must be finite XMIN XMAX YMIN YMAX with positive extents")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.object_size_m > min(xmax - xmin, ymax - ymin):
        parser.error("--object-size-m must fit within the table")
    return args


def _valid_layout(
    poses: list[dict], bounds: tuple[float, float, float, float], sizes: dict[str, tuple[float, float]]
) -> bool:
    """Apply the local complete-set, containment, and XY AABB validator."""
    by_name = {pose["object_name"]: pose for pose in poses}
    if len(by_name) != len(poses) or set(by_name) != set(sizes):
        return False
    xmin, xmax, ymin, ymax = bounds
    for name, pose in by_name.items():
        width, depth = sizes[name]
        if pose["yaw"] != 0.0:
            return False
        if not xmin <= pose["x"] - width / 2 or not pose["x"] + width / 2 <= xmax:
            return False
        if not ymin <= pose["y"] - depth / 2 or not pose["y"] + depth / 2 <= ymax:
            return False
    names = tuple(sizes)
    for index, left_name in enumerate(names):
        left = by_name[left_name]
        left_width, left_depth = sizes[left_name]
        for right_name in names[index + 1 :]:
            right = by_name[right_name]
            right_width, right_depth = sizes[right_name]
            if (
                abs(left["x"] - right["x"]) < (left_width + right_width) / 2
                and abs(left["y"] - right["y"]) < (left_depth + right_depth) / 2
            ):
                return False
    return True


def _canonical_key(poses: list[dict]) -> tuple[tuple[str, float, float], ...]:
    return tuple(sorted((pose["object_name"], round(pose["x"], 6), round(pose["y"], 6)) for pose in poses))


def _sample_robolab_layout(
    generator: random.Random,
    bounds: tuple[float, float, float, float],
    sizes: dict[str, tuple[float, float]],
) -> list[dict]:
    """Sample the base-support poses used by the supplementary baseline."""
    xmin, xmax, ymin, ymax = bounds
    return [
        {
            "object_name": name,
            "x": generator.uniform(xmin + width / 2, xmax - width / 2),
            "y": generator.uniform(ymin + depth / 2, ymax - depth / 2),
            "z": 0.0,
            "yaw": 0.0,
        }
        for name, (width, depth) in sizes.items()
    ]


def _run_sample(
    seed: int,
    target: int,
    bounds: tuple[float, float, float, float],
    sizes: dict[str, tuple[float, float]],
    max_attempts_per_layout: int,
) -> dict:
    generator = random.Random(seed)
    maximum_attempts = target * max_attempts_per_layout
    attempted = accepted = 0
    unique: dict[tuple[tuple[str, float, float], ...], list[dict]] = {}
    start = time.perf_counter()
    while attempted < maximum_attempts and len(unique) < target:
        poses = _sample_robolab_layout(generator, bounds, sizes)
        attempted += 1
        if _valid_layout(poses, bounds, sizes):
            accepted += 1
            unique.setdefault(_canonical_key(poses), poses)
    elapsed_ms = (time.perf_counter() - start) * 1e3
    reached = len(unique) == target
    return {
        "method": "robolab",
        "seed": seed,
        "target_layouts": target,
        "elapsed_ms": elapsed_ms,
        "layouts_per_second": len(unique) * 1e3 / elapsed_ms if elapsed_ms > 0.0 else None,
        "validity_rate": accepted / attempted,
        "unique_layouts": len(unique),
        "attempted_layouts": attempted,
        "accepted_layouts": accepted,
        "target_reached": reached,
        "timing_applicable": True,
        "deterministic": True,
        "iterations": None,
        "gpu_peak_allocated_bytes": None,
        "gpu_peak_reserved_bytes": None,
        "layouts": list(unique.values()),
        "error": None if reached else "bounded attempt budget exhausted before reaching K unique layouts",
    }


def _source_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    bounds = tuple(args.table_bounds)
    sizes = {f"object-{index}": (args.object_size_m, args.object_size_m) for index in range(args.num_objects)}
    seeds = tuple(args.seed + index for index in range(args.repetitions))
    for index in range(args.warmup):
        _run_sample(
            args.seed - args.warmup + index,
            args.target_layouts[0],
            bounds,
            sizes,
            args.max_attempts_per_layout,
        )
    samples = [
        _run_sample(seed, target, bounds, sizes, args.max_attempts_per_layout)
        for seed in seeds
        for target in args.target_layouts
    ]
    run = {
        "workload": "controlled-tabletop",
        "method": "robolab",
        "source_commit": _source_commit(),
        "master_seed": args.seed,
        "seeds": seeds,
        "repetitions": args.repetitions,
        "target_layout_counts": tuple(args.target_layouts),
        "table_xy_bounds": bounds,
        "object_xy_sizes": sizes,
        "max_attempts_per_layout": args.max_attempts_per_layout,
        "max_iterations": None,
        "warmup": args.warmup,
        "samples": samples,
    }
    args.output.write_text(json.dumps(run, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote supplementary base-support throughput results to {args.output}")
    print("These XY base-support results do not claim direct scene or relation/height equivalence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
