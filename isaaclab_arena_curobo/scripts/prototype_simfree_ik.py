# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone sim-free cuRobo IK-reachability demo (no SimulationApp / no Isaac Lab env).

Builds a standalone cuRobo solver for a registered embodiment straight from its cuRobo config, seeds
a bounding-box collision world (a table anchor), then batch-checks top-down grasp IK at a clearly
reachable object and a clearly unreachable (far-away) object. Prints per-pose feasibility and the
best-seed position/rotation errors, and times the per-layout solve.

Run only inside the cuRobo image (``./docker/run_docker.sh -c``); the base image has no cuRobo::

    docker exec "$ARENA_CONTAINER" su $(id -un) -c \
      "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \
       isaaclab_arena_curobo/scripts/prototype_simfree_ik.py"
"""

from __future__ import annotations

import time
import torch

from isaaclab_arena_curobo.embodiment_curobo_registry import DROID_CUROBO_CFG
from isaaclab_arena_curobo.simfree_ik import (
    SimFreeCuboid,
    SimFreeIKReachability,
    check_ik_feasibility_simfree,
    top_down_grasp_pose_simfree,
)

# Robot base pose in the world frame (config-supplied; here the DROID base at the origin, upright).
ROBOT_BASE_POS = (0.0, 0.0, 0.0)
ROBOT_BASE_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)

# A table anchor as a static bounding-box obstacle: 0.8 x 1.2 x 0.1 m, top at z = 0.
TABLE = SimFreeCuboid(name="table", center_xyz=(0.5, 0.0, -0.05), dims_xyz=(0.8, 1.2, 0.1))

# Two candidate objects on the table. The near one sits well inside a Franka's ~0.85 m reach; the far
# one is pushed 3 m out, far beyond it. Each is a small cube whose top-down grasp we IK-check.
REACHABLE_OBJ = {"name": "near_cube", "pos": (0.45, 0.0, 0.03)}
UNREACHABLE_OBJ = {"name": "far_cube", "pos": (3.45, 0.0, 0.03)}
OBJ_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)
GRASP_Z_OFFSET = 0.05


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else None

    t0 = time.perf_counter()
    solver = SimFreeIKReachability(DROID_CUROBO_CFG, device=device, use_motion_gen=True, debug=False)
    print(f"[build+warmup] {time.perf_counter() - t0:.2f} s")

    # One layout: seed the collision world from the table bounding box, in the robot base frame.
    t0 = time.perf_counter()
    solver.update_world([TABLE], ROBOT_BASE_POS, ROBOT_BASE_QUAT_XYZW)
    print(f"[update_world] {(time.perf_counter() - t0) * 1e3:.1f} ms")

    grasp_poses = torch.stack([
        top_down_grasp_pose_simfree(
            obj["pos"], OBJ_QUAT_XYZW, ROBOT_BASE_POS, ROBOT_BASE_QUAT_XYZW, GRASP_Z_OFFSET, device=device
        )
        for obj in (REACHABLE_OBJ, UNREACHABLE_OBJ)
    ])

    # Warm solve (captures/aligns the CUDA graph at this batch size), then a timed solve.
    check_ik_feasibility_simfree(solver, grasp_poses)
    t0 = time.perf_counter()
    feasible, pos_err, rot_err = check_ik_feasibility_simfree(solver, grasp_poses)
    torch.cuda.synchronize()
    solve_ms = (time.perf_counter() - t0) * 1e3
    print(f"[solve_batch] {solve_ms:.1f} ms for {grasp_poses.shape[0]} poses")

    print("\n--- Sim-free IK reachability ---")
    for obj, feas, pe, re in zip((REACHABLE_OBJ, UNREACHABLE_OBJ), feasible, pos_err, rot_err):
        verdict = "REACHABLE" if bool(feas) else "UNREACHABLE"
        print(
            f"{obj['name']:>10} @ {obj['pos']}: {verdict:>11}  pos_err={pe.item():.4f} m  rot_err={re.item():.4f} rad"
        )

    near_ok = bool(feasible[0])
    far_ok = bool(feasible[1])
    if near_ok and not far_ok:
        print("\nRESULT: PASS - near object reachable, far object correctly rejected.")
    else:
        print(f"\nRESULT: UNEXPECTED - near_reachable={near_ok}, far_reachable={far_ok}.")


if __name__ == "__main__":
    main()
