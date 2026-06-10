#!/usr/bin/env python3
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark script comparing BBOX vs MESH collision modes.

Uses realistic object shapes and dimensions from YCB and Arena object library
(cracker box, mustard bottle, soup can, mug, sugar box, power drill proxy).
Measures wall-clock time, placement success, and final loss.

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

# ---------------------------------------------------------------------------
# Realistic object proxies (dimensions from YCB dataset / Arena library)
# ---------------------------------------------------------------------------


def _make_cracker_box() -> DummyObject:
    """YCB 003_cracker_box: 0.158 x 0.213 x 0.072 m."""
    sx, sy, sz = 0.158, 0.213, 0.072
    mesh = trimesh.creation.box(extents=(sx, sy, sz))
    return DummyObject(
        name="cracker_box",
        bounding_box=AxisAlignedBoundingBox(min_point=(-sx / 2, -sy / 2, -sz / 2), max_point=(sx / 2, sy / 2, sz / 2)),
        collision_mesh=mesh,
    )


def _make_mustard_bottle() -> DummyObject:
    """YCB 006_mustard_bottle: ~cylinder r=0.029, h=0.19."""
    r, h = 0.029, 0.19
    mesh = trimesh.creation.cylinder(radius=r, height=h, sections=32)
    return DummyObject(
        name="mustard_bottle",
        bounding_box=AxisAlignedBoundingBox(min_point=(-r, -r, -h / 2), max_point=(r, r, h / 2)),
        collision_mesh=mesh,
    )


def _make_soup_can() -> DummyObject:
    """YCB 005_tomato_soup_can: cylinder r=0.033, h=0.101."""
    r, h = 0.033, 0.101
    mesh = trimesh.creation.cylinder(radius=r, height=h, sections=32)
    return DummyObject(
        name="soup_can",
        bounding_box=AxisAlignedBoundingBox(min_point=(-r, -r, -h / 2), max_point=(r, r, h / 2)),
        collision_mesh=mesh,
    )


def _make_mug() -> DummyObject:
    """Mug: approx cylinder r=0.04, h=0.095 (handle ignored in collision)."""
    r, h = 0.04, 0.095
    mesh = trimesh.creation.cylinder(radius=r, height=h, sections=32)
    return DummyObject(
        name="mug",
        bounding_box=AxisAlignedBoundingBox(min_point=(-r, -r, -h / 2), max_point=(r, r, h / 2)),
        collision_mesh=mesh,
    )


def _make_sugar_box() -> DummyObject:
    """YCB 004_sugar_box: 0.089 x 0.176 x 0.045 m."""
    sx, sy, sz = 0.089, 0.176, 0.045
    mesh = trimesh.creation.box(extents=(sx, sy, sz))
    return DummyObject(
        name="sugar_box",
        bounding_box=AxisAlignedBoundingBox(min_point=(-sx / 2, -sy / 2, -sz / 2), max_point=(sx / 2, sy / 2, sz / 2)),
        collision_mesh=mesh,
    )


def _make_power_drill() -> DummyObject:
    """Power drill: irregular shape approximated by L-shaped box union (~0.18 x 0.07 x 0.19)."""
    body = trimesh.creation.box(extents=(0.18, 0.05, 0.06))
    handle = trimesh.creation.box(extents=(0.04, 0.05, 0.12))
    handle.apply_translation([0.05, 0.0, -0.09])
    mesh = trimesh.util.concatenate([body, handle])
    sx, sy, sz = 0.18, 0.07, 0.19
    return DummyObject(
        name="power_drill",
        bounding_box=AxisAlignedBoundingBox(min_point=(-sx / 2, -sy / 2, -sz / 2), max_point=(sx / 2, sy / 2, sz / 2)),
        collision_mesh=mesh,
    )


def _make_table() -> DummyObject:
    """60x60cm table surface."""
    mesh = trimesh.creation.box(extents=(0.6, 0.6, 0.05))
    table = DummyObject(
        name="table",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.3, -0.3, -0.025), max_point=(0.3, 0.3, 0.025)),
        collision_mesh=mesh,
    )
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())
    return table


# Object factory: creates a diverse set mimicking a real GR1 tabletop scene
_OBJECT_FACTORIES = [
    _make_cracker_box,
    _make_mustard_bottle,
    _make_soup_can,
    _make_mug,
    _make_sugar_box,
    _make_power_drill,
    _make_soup_can,  # second can
    _make_mug,  # second mug
]


def _build_scene(num_objects: int) -> list[DummyObject]:
    table = _make_table()
    objects = [table]
    for i in range(num_objects):
        factory = _OBJECT_FACTORIES[i % len(_OBJECT_FACTORIES)]
        obj = factory()
        obj.name = f"{obj.name}_{i}"
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
    print("Collision Mode Benchmark: BBOX vs MESH (real object proxies)")
    print("=" * 70)
    print()
    print("Objects: cracker_box, mustard_bottle, soup_can, mug, sugar_box, power_drill")
    print()

    configs = [
        (4, [30]),
        (6, [30]),
        (8, [30]),
    ]

    results = []

    for num_objects, sphere_counts in configs:
        # BBOX baseline
        r = _run_benchmark(CollisionMode.BBOX, num_objects, num_spheres=30)
        results.append(r)
        print(f"BBOX | {num_objects} objects | time={r['time_s']:.3f}s | valid={r['valid']} | loss={r['loss']:.6f}")

        # MESH
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
