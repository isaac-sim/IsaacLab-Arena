# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check that DROID can reach every grasp and drop pose of the balance-scale task with cuRobo IK, sim-free.

Every pose is top-down and targets the Robotiq finger-pad midpoint, 0.131 m along the approach axis from the
gripper base_link that cuRobo solves for. The script runs these checks:

- grasp: each weight's start pose, read from object-control runner logs. The hand is collision-checked against
  the other weights, the scale and the apple. A pre-grasp 10 cm higher must also clear the table.
- drop: over four slots on the right pan, at beam angles of -10, 0 and +10 deg, the lowest finger-pad height above
  the rim at which both the release pose and a pre-release 10 cm higher solve. The hand is collision-checked
  against every scale part, including the beam, hanger posts and pan rims.
- arena_check: Arena's own ik_reachable placement check on the same start poses, which puts base_link 2 cm
  above the object root with the hand links muted.

A pose passes when one of the tried gripper yaws (0 or 90 deg) solves within 1 cm and 0.2 rad, collision-free.
Scale parts are boxes bounding each USD Gprim, and the robot is cuRobo's sphere model, so the check is conservative.

Example::

    env PYTHONPATH=. .venv/bin/python isaaclab_arena_environments/balance_scale/check_reachability.py \\
        --runner_logs runs/oc_170/runner.log runs/oc_240/runner.log --output reachability.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import torch
import yaml
from pathlib import Path

import isaaclab.utils.math as math_utils
from pxr import Gf, Usd, UsdGeom

from isaaclab_arena.assets.balance_scale_usd import (
    BEAM_BODY_NAME,
    DEFAULT_GEOMETRY,
    LEFT_PAN_BODY_NAME,
    RIGHT_PAN_BODY_NAME,
    get_balance_scale_usd_path,
)
from isaaclab_arena.assets.object_library import CALIBRATED_WEIGHT_DIMENSIONS_M
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_curobo.embodiment_curobo_registry import get_curobo_cfg_by_name
from isaaclab_arena_curobo.ik_solver import CuroboIKSolver
from isaaclab_arena_curobo.utils.frame_utils import DOWN_FACING_QUAT_XYZW
from isaaclab_arena_curobo.utils.ik_solver_utils import AABBCollisionCuboid, solve_ik_feasibility

ENV_YAML = Path(__file__).parent / "balance_scale_measurement.yaml"
GRASP_POINT_OFFSET_M = 0.131
"""Robotiq base_link to finger-pad midpoint along the approach axis (DROID ee_frame offset)."""
APPROACH_LIFT_M = 0.10
POSITION_TOLERANCE_M = 0.01
ROTATION_TOLERANCE_RAD = 0.2
GRIPPER_YAWS_RAD = (0.0, math.pi / 2)
TABLE_TOP_Z_M = 0.003
ARENA_CHECK_GRASP_Z_OFFSET_M = 0.02
DROP_PAD_HEIGHTS_ABOVE_RIM_M = tuple(0.005 * i for i in range(17))
"""Finger-pad heights above the right pan's rim top tried for a release, lowest first (0 to 8 cm)."""
DROP_SLOTS_XY_W = ((0.515, -0.175), (0.555, -0.175), (0.515, -0.125), (0.555, -0.125))
BEAM_ANGLES_DEG = (-10.0, 0.0, 10.0)
IDENTITY_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)
ROBOT_BASE_POSE_W = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=IDENTITY_QUAT_XYZW)


def load_env_layout() -> tuple[tuple[float, float, float], tuple[float, float, float], dict[str, int]]:
    """Return the scale and apple positions and each weight's mass in grams from the environment YAML."""
    spec = yaml.safe_load(ENV_YAML.read_text())
    objects = {obj["id"]: obj for obj in spec["objects"]}
    t_W_scale = tuple(objects["balance_scale"]["params"]["initial_pose"]["position_xyz"])
    t_W_apple = tuple(objects["apple"]["params"]["initial_pose"]["position_xyz"])
    mass_g_per_weight = {
        obj_id: obj["params"]["mass_g"]
        for obj_id, obj in objects.items()
        if obj["registry_name"] == "calibrated_weight"
    }
    return t_W_scale, t_W_apple, mass_g_per_weight


def pan_pivot_displacement_S(beam_angle_rad: float, side_sign: float) -> tuple[float, float, float]:
    """Displacement of a pan pivot in the scale frame S when the beam turns by beam_angle_rad about +X."""
    g = DEFAULT_GEOMETRY
    y, z = side_sign * g.half_arm_m, -g.pan_pivot_drop_m
    cos_a, sin_a = math.cos(beam_angle_rad), math.sin(beam_angle_rad)
    return (0.0, y * cos_a - z * sin_a - y, y * sin_a + z * cos_a - z)


def scale_part_cuboids(t_W_scale: tuple[float, float, float], beam_angle_rad: float) -> list[AABBCollisionCuboid]:
    """World boxes bounding every Gprim of the scale with the beam at beam_angle_rad and the pans hanging level."""
    g = DEFAULT_GEOMETRY
    stage = Usd.Stage.Open(str(get_balance_scale_usd_path()))
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    T_S_beam_moved = (
        Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.0, 0.0, -g.fulcrum_height_m))
        * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), math.degrees(beam_angle_rad)))
        * Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.0, 0.0, g.fulcrum_height_m))
    )
    body_motion_S = {
        BEAM_BODY_NAME: T_S_beam_moved,
        LEFT_PAN_BODY_NAME: Gf.Matrix4d().SetTranslate(Gf.Vec3d(*pan_pivot_displacement_S(beam_angle_rad, 1.0))),
        RIGHT_PAN_BODY_NAME: Gf.Matrix4d().SetTranslate(Gf.Vec3d(*pan_pivot_displacement_S(beam_angle_rad, -1.0))),
    }
    cuboids = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        # Gprims live at /BalanceScale/<link>/<part>.
        path_parts = str(prim.GetPath()).split("/")
        link_name = path_parts[2] if len(path_parts) > 3 else None
        motion_S = body_motion_S.get(link_name, Gf.Matrix4d(1.0))
        bbox = bbox_cache.ComputeWorldBound(prim)
        T_S_box = bbox.GetMatrix() * motion_S
        corners_S = [T_S_box.Transform(bbox.GetRange().GetCorner(i)) for i in range(8)]
        lower_S = [min(corner[axis] for corner in corners_S) for axis in range(3)]
        upper_S = [max(corner[axis] for corner in corners_S) for axis in range(3)]
        center_W = tuple((lower_S[axis] + upper_S[axis]) / 2 + t_W_scale[axis] for axis in range(3))
        dims = tuple(max(upper_S[axis] - lower_S[axis], 1e-3) for axis in range(3))
        cuboids.append(
            AABBCollisionCuboid(
                name=str(prim.GetPath()).replace("/", "_"),
                dims_xyz=dims,
                pose_W_O=Pose(position_xyz=center_W, rotation_xyzw=IDENTITY_QUAT_XYZW),
            )
        )
    return cuboids


def box_cuboid(name: str, center_W: tuple[float, float, float], dims: tuple[float, float, float]):
    """Axis-aligned collision box at center_W."""
    return AABBCollisionCuboid(
        name=name, dims_xyz=dims, pose_W_O=Pose(position_xyz=center_W, rotation_xyzw=IDENTITY_QUAT_XYZW)
    )


def apple_cuboid(t_W_apple: tuple[float, float, float], beam_angle_rad: float) -> AABBCollisionCuboid:
    """Box around the apple, moved with the left pan."""
    displacement_S = pan_pivot_displacement_S(beam_angle_rad, 1.0)
    center_W = tuple(t_W_apple[axis] + displacement_S[axis] + (0.02 if axis == 2 else 0.0) for axis in range(3))
    return box_cuboid("apple", center_W, (0.09, 0.09, 0.09))


def table_cuboid() -> AABBCollisionCuboid:
    return box_cuboid("table", (0.547, 0.0, TABLE_TOP_Z_M - 0.05), (0.70, 1.0, 0.10))


def weight_cuboid(name: str, t_W_weight: tuple[float, float, float], mass_g: int) -> AABBCollisionCuboid:
    radius_m, height_m, _ = CALIBRATED_WEIGHT_DIMENSIONS_M[mass_g]
    return box_cuboid(name, t_W_weight, (2 * radius_m, 2 * radius_m, height_m))


def top_down_base_link_pose(t_W_target: tuple[float, float, float], yaw_rad: float, offset_m: float) -> torch.Tensor:
    """(1, 4, 4) top-down Robotiq base_link pose offset_m above t_W_target, with the robot base at the world origin."""
    R_down = math_utils.matrix_from_quat(torch.tensor([DOWN_FACING_QUAT_XYZW], dtype=torch.float32))[0]
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    R_yaw = torch.tensor([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])
    t_W = torch.tensor(t_W_target, dtype=torch.float32) + torch.tensor([0.0, 0.0, offset_m])
    return math_utils.make_pose(t_W.view(1, 3), (R_yaw @ R_down).view(1, 3, 3))


class ReachabilityChecker:
    """cuRobo IK for DROID against a replaceable world of boxes."""

    def __init__(self):
        self.solver = CuroboIKSolver(
            get_curobo_cfg_by_name("droid"),
            num_seeds=32,
            position_threshold=POSITION_TOLERANCE_M,
            rotation_threshold=ROTATION_TOLERANCE_RAD,
        )

    def set_world(self, cuboids: list[AABBCollisionCuboid]) -> None:
        self.solver.update_world(cuboids, ROBOT_BASE_POSE_W.position_xyz, ROBOT_BASE_POSE_W.rotation_xyzw)

    def solve_with_hand_collisions(self, T_B_E: torch.Tensor) -> dict:
        """Collision-checked IK in which the gripper links count too, unlike Arena's grasp check."""
        solver = self.solver
        T_B_E = solver._to_curobo_device(T_B_E)
        t_B_E, R_B_E = math_utils.unmake_pose(T_B_E)
        goal = solver._make_goal_tool_pose(
            solver._make_pose(position=t_B_E, quaternion=math_utils.quat_from_matrix(R_B_E), quat_is_xyzw=True)
        )
        result = solver.ik_solver.solve_pose(goal)
        position_error = result.position_error.view(-1)
        rotation_error = result.rotation_error.view(-1)
        ok = (
            (position_error < POSITION_TOLERANCE_M)
            & (rotation_error < ROTATION_TOLERANCE_RAD)
            & result.success.view(-1).bool()
        )
        best = int(torch.where(ok, position_error, position_error + 10.0).argmin())
        return {
            "ok": bool(ok.any()),
            "position_error_m": round(float(position_error[best]), 4),
            "rotation_error_rad": round(float(rotation_error[best]), 4),
        }

    def solve_arena_check(self, T_B_E: torch.Tensor) -> dict:
        """Arena's ik_reachable verdict: collision-checked IK with the gripper links muted."""
        result = solve_ik_feasibility(
            self.solver,
            T_B_E,
            position_threshold=POSITION_TOLERANCE_M,
            rotation_threshold=ROTATION_TOLERANCE_RAD,
            require_collision_free=True,
        )
        return {
            "ok": bool(result.feasible[0]),
            "position_error_m": round(float(result.position_error[0]), 4),
            "rotation_error_rad": round(float(result.rotation_error[0]), 4),
        }

    def first_reachable_yaw(self, t_W_target: tuple[float, float, float], with_approach: bool) -> dict:
        """Try each gripper yaw; return the first whose pose (and pre-pose 10 cm higher) both solve."""
        attempts = []
        for yaw_rad in GRIPPER_YAWS_RAD:
            pose = self.solve_with_hand_collisions(top_down_base_link_pose(t_W_target, yaw_rad, GRASP_POINT_OFFSET_M))
            approach = {"ok": True}
            if with_approach:
                t_W_above = (t_W_target[0], t_W_target[1], t_W_target[2] + APPROACH_LIFT_M)
                approach = self.solve_with_hand_collisions(
                    top_down_base_link_pose(t_W_above, yaw_rad, GRASP_POINT_OFFSET_M)
                )
            attempt = {"yaw_deg": round(math.degrees(yaw_rad)), "pose": pose, "approach": approach}
            attempts.append(attempt)
            if pose["ok"] and approach["ok"]:
                return {"ok": True, **attempt}
        return {"ok": False, "attempts": attempts}


def read_start_layouts(runner_logs: list[Path]) -> list[dict]:
    """Weight start positions from each runner log's first OBJECT_CONTROL_STATE record."""
    layouts = []
    for log_path in runner_logs:
        for line in log_path.read_text(errors="replace").splitlines():
            if line.startswith("OBJECT_CONTROL_STATE "):
                poses = json.loads(line.split(" ", 1)[1])["poses"]
                weights = {name: tuple(pose[0][:3]) for name, pose in poses.items() if re.match(r"weight_\d+g", name)}
                layouts.append({"log": str(log_path), "weights": weights})
                break
    return layouts


def check_grasps(checker: ReachabilityChecker, layouts: list[dict], scene: list[AABBCollisionCuboid], masses):
    """Grasp, pre-grasp and Arena-check results for every weight of every layout."""
    results = []
    for layout in layouts:
        for name, t_W_weight in layout["weights"].items():
            mass_g = masses[name]
            _, height_m, _ = CALIBRATED_WEIGHT_DIMENSIONS_M[mass_g]
            neighbours = [
                weight_cuboid(other, t_W_other, masses[other])
                for other, t_W_other in layout["weights"].items()
                if other != name
            ]
            checker.set_world(scene + neighbours + [table_cuboid()])
            arena = checker.solve_arena_check(top_down_base_link_pose(t_W_weight, 0.0, ARENA_CHECK_GRASP_Z_OFFSET_M))
            # The finger pads close on the weight's upper part. The grasp leaves the table out of the world because
            # cuRobo's fingertip spheres reach about 3 cm past the real fingertips; the pre-grasp keeps it.
            weight_top_z = t_W_weight[2] + height_m / 2
            t_W_grasp = (t_W_weight[0], t_W_weight[1], max(t_W_weight[2], weight_top_z - 0.015))
            t_W_pre_grasp = (t_W_grasp[0], t_W_grasp[1], t_W_grasp[2] + APPROACH_LIFT_M)
            attempts = []
            for yaw_rad in GRIPPER_YAWS_RAD:
                checker.set_world(scene + neighbours)
                grasp = checker.solve_with_hand_collisions(
                    top_down_base_link_pose(t_W_grasp, yaw_rad, GRASP_POINT_OFFSET_M)
                )
                checker.set_world(scene + neighbours + [table_cuboid()])
                pre_grasp = checker.solve_with_hand_collisions(
                    top_down_base_link_pose(t_W_pre_grasp, yaw_rad, GRASP_POINT_OFFSET_M)
                )
                attempts.append({"yaw_deg": round(math.degrees(yaw_rad)), "grasp": grasp, "pre_grasp": pre_grasp})
                if grasp["ok"] and pre_grasp["ok"]:
                    break
            results.append({
                "log": layout["log"],
                "weight": name,
                "t_W": [round(v, 4) for v in t_W_weight],
                "arena_check": arena,
                "physical": {
                    "ok": attempts[-1]["grasp"]["ok"] and attempts[-1]["pre_grasp"]["ok"],
                    "attempts": attempts,
                },
            })
    return results


def check_drops(checker: ReachabilityChecker, t_W_scale, t_W_apple) -> list[dict]:
    """Lowest reachable release height over each right-pan slot, for each weight size and beam angle.

    The release height is scanned upward from the rim top. The weight hangs from the finger pads, which hold it
    1.5 cm below its top as in the grasp check.
    """
    g = DEFAULT_GEOMETRY
    results = []
    for beam_angle_deg in BEAM_ANGLES_DEG:
        beam_angle_rad = math.radians(beam_angle_deg)
        right_pan_displacement_S = pan_pivot_displacement_S(beam_angle_rad, -1.0)
        floor_top_z = (
            t_W_scale[2] + g.fulcrum_height_m - g.pan_pivot_drop_m - g.pan_hanger_length_m + right_pan_displacement_S[2]
        )
        rim_top_z = floor_top_z + g.pan_rim_height_m
        checker.set_world(
            scale_part_cuboids(t_W_scale, beam_angle_rad) + [apple_cuboid(t_W_apple, beam_angle_rad), table_cuboid()]
        )
        for mass_g, (_, height_m, _) in CALIBRATED_WEIGHT_DIMENSIONS_M.items():
            for slot_x, slot_y in DROP_SLOTS_XY_W:
                t_W_slot = (slot_x, slot_y + right_pan_displacement_S[1])
                row = {
                    "beam_angle_deg": beam_angle_deg,
                    "mass_g": mass_g,
                    "slot_xy_W": [round(v, 4) for v in t_W_slot],
                    "drop": {"ok": False},
                }
                for pad_above_rim_m in DROP_PAD_HEIGHTS_ABOVE_RIM_M:
                    drop = checker.first_reachable_yaw((*t_W_slot, rim_top_z + pad_above_rim_m), with_approach=True)
                    if drop["ok"]:
                        weight_bottom_z = rim_top_z + pad_above_rim_m - (height_m - 0.015)
                        row["drop"] = drop
                        row["pad_above_rim_m"] = round(pad_above_rim_m, 3)
                        row["weight_fall_to_floor_m"] = round(weight_bottom_z - floor_top_z, 3)
                        break
                results.append(row)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runner_logs", type=Path, nargs="*", default=[], help="Object-control runner.log files.")
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the per-pose results.")
    args = parser.parse_args()

    t_W_scale, t_W_apple, masses = load_env_layout()
    checker = ReachabilityChecker()
    level_scene = scale_part_cuboids(t_W_scale, 0.0) + [apple_cuboid(t_W_apple, 0.0)]
    grasps = check_grasps(checker, read_start_layouts(args.runner_logs), level_scene, masses)
    drops = check_drops(checker, t_W_scale, t_W_apple)
    args.output.write_text(json.dumps({"grasps": grasps, "drops": drops}, indent=1))

    def count(rows, key):
        return f"{sum(row[key]['ok'] for row in rows)}/{len(rows)}"

    print(f"grasps: arena_check {count(grasps, 'arena_check')}, grasp + pre-grasp {count(grasps, 'physical')}")
    for beam_angle_deg in BEAM_ANGLES_DEG:
        rows = [row for row in drops if row["beam_angle_deg"] == beam_angle_deg]
        reached = [row for row in rows if row["drop"]["ok"]]
        worst_pad = max((row["pad_above_rim_m"] for row in reached), default=float("nan"))
        worst_fall = max((row["weight_fall_to_floor_m"] for row in reached), default=float("nan"))
        print(
            f"drops at beam {beam_angle_deg:+.0f} deg: {count(rows, 'drop')} reachable; finger pads need up to"
            f" {worst_pad * 100:.1f} cm above the rim, so a weight falls up to {worst_fall * 100:.1f} cm"
        )
    for row in drops:
        if not row["drop"]["ok"]:
            print(f"  unreachable drop: {row['mass_g']} g at {row['slot_xy_W']}, beam {row['beam_angle_deg']:+.0f} deg")
    for row in grasps:
        if not row["physical"]["ok"]:
            print(f"  unreachable grasp: {row['weight']} at {row['t_W']} ({row['log']})")
        if not row["arena_check"]["ok"]:
            print(f"  fails Arena's check: {row['weight']} at {row['t_W']} ({row['log']})")


if __name__ == "__main__":
    main()
