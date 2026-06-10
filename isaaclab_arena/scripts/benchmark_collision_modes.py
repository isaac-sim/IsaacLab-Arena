#!/usr/bin/env python3
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark script comparing BBOX vs MESH collision modes.

Measures wall-clock time, iterations to convergence, and placement success
for a configurable number of objects and sphere budgets.

Usage:
    python isaaclab_arena/scripts/benchmark_collision_modes.py
"""

from __future__ import annotations

import numpy as np
import time
import trimesh

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.relation_solver_params import CollisionMode, RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


def _make_cylinder(name: str, radius: float = 0.033, height: float = 0.1) -> DummyObject:
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
    return DummyObject(
        name=name,
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-radius, -radius, -height / 2),
            max_point=(radius, radius, height / 2),
        ),
        collision_mesh=mesh,
    )


def _make_table() -> DummyObject:
    mesh = trimesh.creation.box(extents=(0.6, 0.6, 0.05))
    table = DummyObject(
        name="table",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.3, -0.3, -0.025), max_point=(0.3, 0.3, 0.025)),
        collision_mesh=mesh,
    )
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())
    return table


def _build_scene(num_objects: int) -> list[DummyObject]:
    table = _make_table()
    objects = [table]
    for i in range(num_objects):
        obj = _make_cylinder(f"cyl_{i}", radius=0.025 + 0.005 * (i % 3))
        obj.add_relation(On(table))
        objects.append(obj)
    return objects


def _run_benchmark(
    mode: CollisionMode,
    num_objects: int,
    num_spheres: int,
    max_iters: int = 400,
    attempts: int = 5,
) -> dict:
    objects = _build_scene(num_objects)
    params = ObjectPlacerParams(
        solver_params=RelationSolverParams(
            collision_mode=mode,
            num_spheres=num_spheres,
            max_iters=max_iters,
            verbose=False,
        ),
        max_placement_attempts=attempts,
        verbose=False,
    )
    placer = ObjectPlacer(params=params)

    t0 = time.perf_counter()
    result = placer.place(objects)
    elapsed = time.perf_counter() - t0

    return {
        "mode": mode.value,
        "num_objects": num_objects,
        "num_spheres": num_spheres,
        "time_s": elapsed,
        "valid": result.success,
        "loss": result.final_loss,
    }


def main():
    print("=" * 70)
    print("Collision Mode Benchmark: BBOX vs MESH")
    print("=" * 70)
    print()

    configs = [
        (5, [10, 30, 50]),
        (8, [10, 30, 50]),
    ]

    results = []

    for num_objects, sphere_counts in configs:
        # BBOX baseline
        r = _run_benchmark(CollisionMode.BBOX, num_objects, num_spheres=30)
        results.append(r)
        print(f"BBOX | {num_objects} objects | time={r['time_s']:.3f}s | valid={r['valid']} | loss={r['loss']:.6f}")

        # MESH variants
        for ns in sphere_counts:
            r = _run_benchmark(CollisionMode.MESH, num_objects, num_spheres=ns)
            results.append(r)
            print(
                f"MESH | {num_objects} objects | spheres={ns:3d} | "
                f"time={r['time_s']:.3f}s | valid={r['valid']} | loss={r['loss']:.6f}"
            )
        print()

    print("-" * 70)
    print("Summary:")
    bbox_times = [r["time_s"] for r in results if r["mode"] == "bbox"]
    mesh_times = [r["time_s"] for r in results if r["mode"] == "mesh"]
    if bbox_times and mesh_times:
        print(f"  BBOX avg: {np.mean(bbox_times):.3f}s")
        print(f"  MESH avg: {np.mean(mesh_times):.3f}s")
        print(f"  Slowdown: {np.mean(mesh_times) / np.mean(bbox_times):.1f}x")


if __name__ == "__main__":
    main()
