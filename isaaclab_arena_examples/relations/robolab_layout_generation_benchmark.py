# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run RoboLab's SpatialSolver on the controlled tabletop benchmark.

This script intentionally has no IsaacLab-Arena imports and runs inside a
RoboLab Python environment. Its neutral JSON can be analyzed alongside Arena
results produced with the same geometry, seeds, and bounded-attempt budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections.abc import Callable
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
    parser.add_argument("--robolab-root", type=Path, required=True)
    parser.add_argument("-k", "--target-layouts", type=_positive_int, action="append", required=True)
    parser.add_argument("--repetitions", type=_positive_int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-objects", type=_positive_int, default=5)
    parser.add_argument("--object-size-m", type=_positive_float, default=0.12)
    parser.add_argument("--table-bounds", type=float, nargs=4, default=(-0.5, 0.5, -0.5, 0.5))
    parser.add_argument("--max-attempts-per-layout", type=_positive_int, default=100)
    parser.add_argument("--max-iterations", type=_positive_int, default=600)
    parser.add_argument("--collision-margin-m", type=float, default=0.0)
    parser.add_argument("--allow-relaxation", action="store_true")
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
    if not math.isfinite(args.collision_margin_m) or args.collision_margin_m < 0.0:
        parser.error("--collision-margin-m must be finite and non-negative")
    if not (args.robolab_root / "robolab" / "scene_gen" / "llm_scene_gen").is_dir():
        parser.error("--robolab-root does not contain robolab/scene_gen/llm_scene_gen")
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


def _load_robolab_solver(robolab_root: Path):
    """Import RoboLab's spatial solver without loading its physics solver."""
    sys.path.insert(0, str(robolab_root.resolve()))
    from robolab.scene_gen.llm_scene_gen.predicates import ObjectState, PlaceOnBasePredicate
    from robolab.scene_gen.llm_scene_gen.spatial_solver import SpatialSolver

    return ObjectState, PlaceOnBasePredicate, SpatialSolver


def _make_solve_layout(
    *,
    robolab_root: Path,
    bounds: tuple[float, float, float, float],
    sizes: dict[str, tuple[float, float]],
    collision_margin_m: float,
    max_iterations: int,
    allow_relaxation: bool,
) -> Callable[[int], tuple[bool, list[dict], str | None, float]]:
    """Build a seeded single-layout call to RoboLab's SpatialSolver."""
    ObjectState, PlaceOnBasePredicate, SpatialSolver = _load_robolab_solver(robolab_root)
    dimensions = {name: (width, depth, 0.1) for name, (width, depth) in sizes.items()}

    def solve_layout(seed: int) -> tuple[bool, list[dict], str | None, float]:
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed % (2**32))
        except ImportError:
            pass
        states = {
            name: ObjectState(
                name=name,
                predicates=[PlaceOnBasePredicate(name, yaw=0.0)],
            )
            for name in sizes
        }
        solver = SpatialSolver(table_bounds=bounds, collision_margin=collision_margin_m)
        start = time.perf_counter()
        success, message = solver.solve(
            states,
            dimensions,
            max_iterations=max_iterations,
            allow_relaxation=allow_relaxation,
        )
        elapsed_ms = (time.perf_counter() - start) * 1e3
        poses = [
            {
                "object_name": name,
                "x": float(state.x),
                "y": float(state.y),
                "z": dimensions[name][2] / 2.0,
                "yaw": math.radians(float(state.yaw)),
            }
            for name, state in states.items()
            if state.x is not None and state.y is not None and state.yaw is not None
        ]
        if not success:
            return False, poses, message or "RoboLab SpatialSolver failed", elapsed_ms
        if not _valid_layout(poses, bounds, sizes):
            return False, poses, "RoboLab result failed the shared XY validator", elapsed_ms
        return True, poses, None, elapsed_ms

    return solve_layout


def _run_sample(
    seed: int,
    target: int,
    max_attempts_per_layout: int,
    solve_layout: Callable[[int], tuple[bool, list[dict], str | None, float]],
) -> dict:
    maximum_attempts = target * max_attempts_per_layout
    attempted = accepted = 0
    unique: dict[tuple[tuple[str, float, float], ...], list[dict]] = {}
    last_error = None
    start = time.perf_counter()
    while attempted < maximum_attempts and len(unique) < target:
        success, poses, last_error, _ = solve_layout(seed + attempted)
        attempted += 1
        if success:
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
        "error": None if reached else last_error or "bounded attempt budget exhausted before reaching K unique layouts",
    }


def _git_output(repository_root: Path, *args: str) -> str:
    """Return stdout from one Git command."""
    return subprocess.run(
        ["git", "-C", str(repository_root), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout


def _source_revision(robolab_root: Path) -> str | None:
    try:
        commit = _git_output(robolab_root, "rev-parse", "HEAD").strip()
        if not _git_output(robolab_root, "status", "--porcelain"):
            return commit
        digest = hashlib.sha256(_git_output(robolab_root, "diff", "--binary", "HEAD").encode())
        untracked = _git_output(robolab_root, "ls-files", "--others", "--exclude-standard").splitlines()
        for relative_path in sorted(untracked):
            digest.update(relative_path.encode())
            digest.update((robolab_root / relative_path).read_bytes())
        return f"{commit}+dirty.{digest.hexdigest()[:12]}"
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    bounds = tuple(args.table_bounds)
    sizes = {f"object-{index}": (args.object_size_m, args.object_size_m) for index in range(args.num_objects)}
    seed_stride = max(args.target_layouts) * args.max_attempts_per_layout
    seeds = tuple(args.seed + index * seed_stride for index in range(args.repetitions))
    solve_layout = _make_solve_layout(
        robolab_root=args.robolab_root,
        bounds=bounds,
        sizes=sizes,
        collision_margin_m=args.collision_margin_m,
        max_iterations=args.max_iterations,
        allow_relaxation=args.allow_relaxation,
    )
    for index in range(args.warmup):
        _run_sample(
            args.seed - (index + 1) * seed_stride,
            args.target_layouts[0],
            args.max_attempts_per_layout,
            solve_layout,
        )
    samples = [
        _run_sample(seed, target, args.max_attempts_per_layout, solve_layout)
        for seed in seeds
        for target in args.target_layouts
    ]
    run = {
        "workload": "controlled-tabletop",
        "method": "robolab",
        "source_commit": _source_revision(args.robolab_root),
        "master_seed": args.seed,
        "seeds": seeds,
        "repetitions": args.repetitions,
        "target_layout_counts": tuple(args.target_layouts),
        "table_xy_bounds": bounds,
        "object_xy_sizes": sizes,
        "max_attempts_per_layout": args.max_attempts_per_layout,
        "max_iterations": args.max_iterations,
        "warmup": args.warmup,
        "solver_config": {
            "name": "SpatialSolver",
            "execution": "serial-single-layout-calls",
            "timing_scope": "complete-exact-k-generation",
            "collision_margin_m": args.collision_margin_m,
            "collision_model": "max-xy-radius-circle",
            "allow_relaxation": args.allow_relaxation,
            "random_yaw": False,
            "benchmark_revision": _source_revision(Path(__file__).resolve().parents[2]),
        },
        "samples": samples,
    }
    args.output.write_text(json.dumps(run, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote RoboLab SpatialSolver throughput results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
