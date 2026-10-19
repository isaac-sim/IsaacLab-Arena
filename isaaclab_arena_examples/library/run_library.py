# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Return cataloged books to shelves and deliver a requested book using a mobile NHC12 arm."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena_examples.bento.meal import ROBOT_ASSET, TABLE_TOP
from isaaclab_arena_examples.library.mobile_controller import LibraryController
from isaaclab_arena_examples.library.scene import BOOKS, add_book_tongs, create_mobile_base, create_room


def placement_ok(book, pose, velocity, released):
    """Check measured final book position, orientation and rest after release."""
    if not released or not all(math.isfinite(v) for v in (*pose, *velocity)):
        return False
    return (
        all(abs(pose[i] - book.target[i]) < 0.008 for i in range(3))
        and abs(pose[6]) > math.cos(0.05)
        and math.sqrt(sum(v * v for v in velocity[:3])) < 0.02
        and math.sqrt(sum(v * v for v in velocity[3:])) < 0.1
    )


if TYPE_CHECKING:
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def parse_args() -> argparse.Namespace:
    """Parse and validate all arguments before launching Kit."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--robot-usd", help="Override the official NHC12 asset with a local or remote USD.")
    parser.add_argument("--max-steps", type=int, default=80000)
    parser.add_argument("--waypoint-timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/library_result.json"))
    parser.add_argument("--capture-dir", type=Path, help="Save measured geometry and poses for offline video.")
    parser.add_argument("--capture-fps", type=float, default=10.0)
    parser.add_argument("--inspect", action="store_true", help="Inspect the scene without moving the arm.")
    parser.add_argument("--keep-open", action="store_true", help="Keep the GUI open after the cycle.")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cpu")
    args = parser.parse_args()
    if args.device != "cpu":
        parser.error("This demo uses CPU PhysX for switchable grasp constraints; use --device cpu.")
    if args.max_steps <= 0 or not math.isfinite(args.waypoint_timeout) or args.waypoint_timeout <= 0:
        parser.error("--max-steps and --waypoint-timeout must be positive and finite.")
    if not math.isfinite(args.capture_fps) or not 0 < args.capture_fps <= 120:
        parser.error("--capture-fps must be finite and in (0, 120].")
    if args.capture_dir is not None and args.capture_dir.exists() and any(args.capture_dir.iterdir()):
        parser.error("--capture-dir must be missing or empty; choose a fresh directory.")
    if args.headless and (args.keep_open or args.inspect):
        parser.error("--keep-open and --inspect require the GUI.")
    if args.visualizer is None:
        args.visualizer = [] if args.headless else ["kit"]
    return args


def create_scene(args: argparse.Namespace):
    """Create library boxes, shelving, a mobile arm, and ideal grasp constraints."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from pxr import Usd, UsdPhysics

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cpu", use_fabric=False))
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 0.96, 0.90))
    light.func("/World/Light", light)

    books = create_room(sim)
    mobile_base = create_mobile_base(sim)

    usd_path = args.robot_usd or f"{ISAAC_NUCLEUS_DIR}/{ROBOT_ASSET}"
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=True,
                    enabled_self_collisions=True,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=4,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, -1.50, TABLE_TOP), joint_pos={".*": 0.0}),
            actuators={"arm": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=800.0, damping=80.0)},
        )
    )
    stage = sim.stage
    links = [
        prim
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot"))
        if prim.GetName() == "link_6_t" and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    assert len(links) == 1, "Expected the NHC12 rigid link_6_t; --robot-usd must retain the NHC12 link structure."
    ee_path = str(links[0].GetPath())
    jaw_ops = add_book_tongs(stage, ee_path, BOOKS[0].size[1] + 0.020)

    joints = {}
    for book in BOOKS:
        path = str(books[book.name].cfg.prim_path)
        # Pre-create switchable external constraints before initializing tensor views.
        joint = UsdPhysics.FixedJoint.Define(stage, f"/World/Grasp_{book.name}")
        joint.CreateBody0Rel().SetTargets([ee_path])
        joint.CreateBody1Rel().SetTargets([path])
        joint.CreateJointEnabledAttr(False)
        joint.CreateExcludeFromArticulationAttr(True)
        joint.CreateCollisionEnabledAttr(False)
        joints[book.name] = joint

    sim.reset()
    robot.update(sim.get_physics_dt())
    mobile_base.update(sim.get_physics_dt())
    for book in books.values():
        book.update(sim.get_physics_dt())
    assert robot.is_fixed_base and robot.num_joints == 6, "Expected a fixed-base, six-axis NHC12 articulation."
    sim.initialize_visualizers()
    sim.set_camera_view((1.65, -1.65, 1.9), (0.38, 0.03, 0.88))
    return sim, robot, books, joints, jaw_ops, usd_path, mobile_base


def run(args: argparse.Namespace, app: SimulationAppContext, report: dict) -> None:
    """Run a bounded cycle and populate a report from measured final body states."""
    import torch

    sim, robot, books, joints, jaw_ops, usd_path, mobile_base = create_scene(args)
    report["robot_usd"] = usd_path
    controller = LibraryController(sim, robot, books, joints, jaw_ops, args.waypoint_timeout, mobile_base)
    capture = None
    if args.capture_dir is not None:
        from isaaclab_arena_examples.library.capture import SceneCapture

        capture = SceneCapture(sim, robot, books, args.capture_dir, args.capture_fps, mobile_base=mobile_base)
        capture.record(0, controller)
    dt = sim.get_physics_dt()
    stable_time = settle_time = 0.0
    robot.set_joint_position_target_index(target=robot.data.default_joint_pos.torch)
    try:
        with torch.inference_mode():
            for step in range(args.max_steps):
                if not app.is_running() or app.is_exiting():
                    report["reason"] = "window_closed"
                    break
                started = time.monotonic()
                if not args.inspect and step >= 120:
                    controller.step()
                mobile_base.write_data_to_sim()
                robot.write_data_to_sim()
                sim.step()
                robot.update(dt)
                mobile_base.update(dt)
                for book in books.values():
                    book.update(dt)
                report["steps"] = step + 1
                if capture is not None:
                    capture.record(step + 1, controller)
                if args.inspect:
                    continue
                if controller.failure:
                    report["reason"] = controller.failure
                    break
                if controller.finished:
                    settle_time += dt
                    all_ok = all(
                        placement_ok(
                            book,
                            books[book.name].data.root_pose_w.torch[0].tolist(),
                            books[book.name].data.root_vel_w.torch[0].tolist(),
                            book.name in controller.released,
                        )
                        for book in BOOKS
                    )
                    stable_time = stable_time + dt if all_ok else 0.0
                    if stable_time >= 0.5:
                        report.update(
                            success=controller.lookup_complete, reason="returns_shelved_and_requested_book_in_loan_box"
                        )
                        break
                    if settle_time >= 3.0:
                        report["reason"] = "placement_failed_after_release"
                        break
                if not args.headless:
                    time.sleep(max(0.0, dt - (time.monotonic() - started)))
            else:
                report["reason"] = "inspection_complete" if args.inspect else "max_steps"
    finally:
        report["phase"] = controller.phase
        report["events"] = controller.events
        report["catalog_lookup_complete"] = controller.lookup_complete
        report["navigation_model"] = "prescribed_kinematic_mobile_base_route"
        report["controller_state"] = {
            "target_tcp": controller.target[0].tolist(),
            "measured_tcp": controller.tcp_pose()[0][0].tolist(),
            "joint_positions": robot.data.joint_pos.torch[0].tolist(),
            "jaw_opening": controller.jaw_opening,
            "command_tcp": controller.command_pos[0].tolist(),
            "arm_root_pose": robot.data.root_pose_w.torch[0].tolist(),
            "mobile_base_pose": mobile_base.data.root_pose_w.torch[0].tolist(),
        }
        report["books"] = {
            book.name: {
                "catalog_id": book.catalog_id,
                "shelf_slot": book.shelf_slot,
                "target": book.target,
                "pose_xyzw": books[book.name].data.root_pose_w.torch[0].tolist(),
                "velocity": books[book.name].data.root_vel_w.torch[0].tolist(),
                "released": book.name in controller.released,
                "placement_ok": placement_ok(
                    book,
                    books[book.name].data.root_pose_w.torch[0].tolist(),
                    books[book.name].data.root_vel_w.torch[0].tolist(),
                    book.name in controller.released,
                ),
            }
            for book in BOOKS
        }
        if capture is not None:
            capture.record(report.get("steps", 0), controller, force=True)
            capture.save(report)
        for joint in joints.values():
            joint.GetJointEnabledAttr().Set(False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[library] {report['reason']}; result: {args.output}", flush=True)
    if args.keep_open:
        while app.is_running() and not app.is_exiting():
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)
            time.sleep(dt)


def main() -> None:
    """Launch Kit, save success or failure, and return a nonzero exit status on failure."""
    args = parse_args()
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    report = {
        "success": False,
        "reason": "initialization_failed",
        "grasp_model": "animated_paddle_tongs_with_ideal_fixed_constraint",
        "book_model": "rigid_cuboids",
    }
    with SimulationAppContext(args) as app:
        try:
            run(args, app, report)
        except Exception as exc:
            report.update(success=False, reason=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if not args.inspect:
            assert report["success"], f"Library cycle failed: {report['reason']}. See {args.output}."


if __name__ == "__main__":
    main()
