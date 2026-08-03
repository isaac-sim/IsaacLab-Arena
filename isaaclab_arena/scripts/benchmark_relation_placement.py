# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reproducible, simulator-free benchmarks for BBOX relation placement."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, NextTo, On, Side
from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

BENCHMARK_REVISION = "relation-placement-benchmark-v1"
"""Version of the deterministic fixture and JSON schema."""

_DEFAULT_CHECKPOINTS = (25, 50, 100, 200, 400, 600)


def _positive_int(value: str) -> int:
    """Parse a command-line positive integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the opt-in BBOX benchmark command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device-policy",
        choices=("bbox-cpu",),
        default="bbox-cpu",
        help="BBOX is intentionally CPU-only; MESH is not benchmarked by this CLI.",
    )
    parser.add_argument("--candidates", nargs="+", type=_positive_int, required=True)
    parser.add_argument("--iterations", nargs="+", type=_positive_int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--requested-layouts", type=_positive_int, default=1)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args(argv)


def create_benchmark_objects() -> list[DummyObject]:
    """Create a small deterministic relation graph with no simulator fixtures."""
    desk = DummyObject(
        name="desk",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    desk.add_relation(IsAnchor())

    box1 = DummyObject(
        name="box1",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )
    box2 = DummyObject(
        name="box2",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.15, 0.15, 0.15)),
    )
    box1.add_relation(On(desk, clearance_m=0.01))
    box2.add_relation(On(desk, clearance_m=0.01))
    box2.add_relation(NextTo(box1, side=Side.POSITIVE_X, distance_m=0.05))
    return [desk, box1, box2]


def _checkpoint_iters(max_iterations: int) -> tuple[int, ...]:
    """Return the standard checkpoints strictly below a requested iteration cap."""
    return tuple(iteration for iteration in _DEFAULT_CHECKPOINTS if iteration < max_iterations)


def _serialize_layout(result) -> dict[str, object]:
    """Convert a placement result into a deterministic, JSON-native layout record."""
    return {
        "success": result.success,
        "loss": result.final_loss,
        "positions": {
            obj.name: list(position)
            for obj, position in sorted(result.positions.items(), key=lambda item: item[0].name)
        },
        "orientations": {
            obj.name: yaw for obj, yaw in sorted(result.orientations.items(), key=lambda item: item[0].name)
        },
    }


def run_benchmark_case(
    *,
    objects: list[DummyObject],
    seed: int,
    candidate_count: int,
    requested_layouts: int,
    max_iterations: int,
    device_policy: str,
) -> dict[str, object]:
    """Resolve one deterministic BBOX placement case and return stable counters/layouts."""
    if device_policy != "bbox-cpu":
        raise ValueError(f"Only the BBOX CPU policy is supported, got {device_policy!r}.")

    solver_params = RelationSolverParams(
        max_iters=max_iterations,
        checkpoint_iters=_checkpoint_iters(max_iterations),
        collision_mode=CollisionMode.BBOX,
        profile=True,
        verbose=False,
    )
    placer = ObjectPlacer(
        ObjectPlacerParams(
            placement_seed=seed,
            max_placement_attempts=candidate_count,
            apply_positions_to_objects=False,
            solver_params=solver_params,
        )
    )

    resolver_start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        results = placer.place_ranked_per_env(objects, num_envs=1, results_per_env=requested_layouts)[0]
    resolver_ms = (time.perf_counter() - resolver_start) * 1_000.0

    profile = placer.last_profile
    assert profile is not None, "Benchmark cases require PlacementProfile output."
    optimizer_ms = profile.checkpoints[-1].elapsed_ms
    validation_ms = sum(profile.validation_times_ms.values())
    generation_ms = max(0.0, resolver_ms - optimizer_ms - validation_ms)
    strict_count = sum(profile.strict_layouts_per_env)
    used_best_loss_fallback = profile.used_best_loss_fallback or any(not result.success for result in results)
    collision_mode = profile.collision_mode.value
    identity = {
        "revision": BENCHMARK_REVISION,
        "seed": seed,
        "collision_mode": collision_mode,
        "candidate_count": profile.candidate_count,
        "requested_layouts": requested_layouts,
        "max_iterations": max_iterations,
    }
    return {
        "revision": BENCHMARK_REVISION,
        "seed": seed,
        "collision_mode": collision_mode,
        "candidate_count": profile.candidate_count,
        "requested_layouts": requested_layouts,
        "max_iterations": max_iterations,
        "device": profile.device,
        "iterations": profile.cumulative_iterations,
        "strict_count": strict_count,
        "generation_ms": generation_ms,
        "validation_ms": validation_ms,
        "optimizer_ms": optimizer_ms,
        "resolver_ms": resolver_ms,
        "used_best_loss_fallback": used_best_loss_fallback,
        "identity": identity,
        "layouts": [_serialize_layout(result) for result in results],
    }


def _write_json_atomically(output_path: Path, payload: dict[str, object]) -> None:
    """Publish JSON through a temporary sibling so readers never see partial output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(payload, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
    try:
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _print_case_table(cases: list[dict[str, object]]) -> None:
    """Print a compact human-readable summary while JSON remains authoritative."""
    print("candidates max-iters ran strict fallback device resolver-ms")
    for case in cases:
        print(
            f"{case['candidate_count']:>10} {case['max_iterations']:>9} {case['iterations']:>3} "
            f"{case['strict_count']:>6} {str(case['used_best_loss_fallback']):>8} "
            f"{case['device']:>6} {case['resolver_ms']:>11.3f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run a warmed BBOX benchmark matrix and atomically publish its JSON report."""
    startup_start = time.perf_counter()
    args = parse_args(argv)

    run_benchmark_case(
        objects=create_benchmark_objects(),
        seed=args.seed,
        candidate_count=1,
        requested_layouts=1,
        max_iterations=min(args.iterations),
        device_policy=args.device_policy,
    )
    cold_startup_ms = (time.perf_counter() - startup_start) * 1_000.0

    cases = [
        run_benchmark_case(
            objects=create_benchmark_objects(),
            seed=args.seed,
            candidate_count=candidate_count,
            requested_layouts=args.requested_layouts,
            max_iterations=max_iterations,
            device_policy=args.device_policy,
        )
        for candidate_count in args.candidates
        for max_iterations in args.iterations
    ]
    payload = {
        "revision": BENCHMARK_REVISION,
        "configuration": {
            "device_policy": args.device_policy,
            "seed": args.seed,
            "candidates": args.candidates,
            "iterations": args.iterations,
            "requested_layouts": args.requested_layouts,
        },
        "cold_startup_ms": cold_startup_ms,
        "cases": cases,
    }
    _write_json_atomically(args.json_output, payload)
    _print_case_table(cases)
    for case in cases:
        if case["used_best_loss_fallback"]:
            print(
                "WARNING: best-loss fallback was selected; this layout is not strict.",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
