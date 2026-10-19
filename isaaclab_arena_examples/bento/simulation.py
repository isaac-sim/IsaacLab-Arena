# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run one NHC12 bento plating cycle with detailed rigid foods and paddle tongs."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena_examples.bento.meal import (
    EE_QUAT_XYZW,
    FOODS,
    HOVER_Z,
    ROBOT_ASSET,
    TABLE_TOP,
    TCP_OFFSET,
    TRAY_FLOOR,
    placement_ok,
)

if TYPE_CHECKING:
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def parse_args() -> argparse.Namespace:
    """Parse and validate all arguments before launching Kit."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--robot-usd", help="Override the official NHC12 asset with a local or remote USD.")
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--waypoint-timeout", type=float, default=15.0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/yaskawa_bento_result.json"))
    parser.add_argument("--inspect", action="store_true", help="Inspect the scene without moving the arm.")
    parser.add_argument("--keep-open", action="store_true", help="Keep the GUI open after the cycle.")
    parser.add_argument(
        "--capture-dir", type=Path, help="Save visible USD geometry and measured poses for video rendering."
    )
    parser.add_argument("--capture-fps", type=float, default=10.0)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cpu")
    args = parser.parse_args()
    if args.device != "cpu":
        parser.error("This demo uses CPU PhysX for switchable grasp constraints; use --device cpu.")
    if args.max_steps <= 0 or not math.isfinite(args.waypoint_timeout) or args.waypoint_timeout <= 0:
        parser.error("--max-steps and --waypoint-timeout must be positive and finite.")
    if args.headless and (args.keep_open or args.inspect):
        parser.error("--keep-open and --inspect require the GUI.")
    if not math.isfinite(args.capture_fps) or not 0 < args.capture_fps <= 120:
        parser.error("--capture-fps must be finite and in (0, 120].")
    if args.capture_dir is not None and args.capture_dir.exists() and any(args.capture_dir.iterdir()):
        parser.error("--capture-dir must be missing or empty; choose a fresh directory.")
    if args.visualizer is None:
        args.visualizer = [] if args.headless else ["kit"]
    return args


def create_scene(args: argparse.Namespace):
    """Create the fixed arm, bento, detailed foods, and ideal grasp constraints."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena_examples.bento.visuals import add_bento_visual, add_food_visual, add_serving_trays, add_tongs

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cpu", use_fabric=False))
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 0.96, 0.90))
    light.func("/World/Light", light)

    def box(name, size, pos, color, proxy=False):
        cfg = sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.65),
        )
        cfg.func(f"/World/{name}", cfg, translation=pos)
        if proxy:
            UsdGeom.Imageable(sim.stage.GetPrimAtPath(f"/World/{name}")).CreatePurposeAttr(UsdGeom.Tokens.proxy)

    box("Table", (1.3, 1.1, 0.05), (0.35, 0.0, TABLE_TOP - 0.025), (0.75, 0.77, 0.75))
    box("Pedestal", (0.22, 0.22, TABLE_TOP - 0.05), (0.0, 0.0, (TABLE_TOP - 0.05) / 2), (0.2, 0.22, 0.24))
    box("BentoFloor", (0.4, 0.32, 0.012), (0.57, 0.24, TABLE_TOP + 0.006), (0.12, 0.075, 0.055), proxy=True)
    for name, size, pos in (
        ("LeftWall", (0.012, 0.32, 0.055), (0.37, 0.24, TRAY_FLOOR + 0.0275)),
        ("RightWall", (0.012, 0.32, 0.055), (0.77, 0.24, TRAY_FLOOR + 0.0275)),
        ("FrontWall", (0.4, 0.012, 0.055), (0.57, 0.08, TRAY_FLOOR + 0.0275)),
        ("BackWall", (0.4, 0.012, 0.055), (0.57, 0.40, TRAY_FLOOR + 0.0275)),
        ("DividerX", (0.012, 0.32, 0.032), (0.57, 0.24, TRAY_FLOOR + 0.016)),
        ("DividerY", (0.4, 0.012, 0.032), (0.57, 0.24, TRAY_FLOOR + 0.016)),
    ):
        box(name, size, pos, (0.20, 0.12, 0.075), proxy=True)
    add_bento_visual(sim.stage, TRAY_FLOOR)
    add_serving_trays(sim.stage, TABLE_TOP, FOODS)

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
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_TOP), joint_pos={".*": 0.0}),
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
    jaw_ops = add_tongs(stage, ee_path, FOODS[0].size[1] + 0.020)

    foods, joints = {}, {}
    for food in FOODS:
        kwargs = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                # Approximate resistance to rolling for the spherical food portions.
                angular_damping=2.0 if food.sphere else 0.1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=food.mass),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=food.color, roughness=0.8),
        )
        spawn = (
            sim_utils.SphereCfg(radius=food.size[0] / 2, **kwargs)
            if food.sphere
            else sim_utils.CuboidCfg(size=food.size, **kwargs)
        )
        path = f"/World/Food_{food.name}"
        foods[food.name] = RigidObject(
            RigidObjectCfg(
                prim_path=path,
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(*food.source_xy, TABLE_TOP + food.size[2] / 2 + 0.002)),
            )
        )
        # Keep simple rigid-body collision proxies while rendering detailed food meshes.
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if prim.IsA(UsdGeom.Gprim):
                UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.proxy)
        add_food_visual(stage, path, food)
        # Pre-create switchable external constraints before initializing tensor views.
        joint = UsdPhysics.FixedJoint.Define(stage, f"/World/Grasp_{food.name}")
        joint.CreateBody0Rel().SetTargets([ee_path])
        joint.CreateBody1Rel().SetTargets([path])
        joint.CreateJointEnabledAttr(False)
        joint.CreateExcludeFromArticulationAttr(True)
        joint.CreateCollisionEnabledAttr(False)
        joints[food.name] = joint

    sim.reset()
    robot.update(sim.get_physics_dt())
    for food in foods.values():
        food.update(sim.get_physics_dt())
    assert robot.is_fixed_base and robot.num_joints == 6, "Expected a fixed-base, six-axis NHC12 articulation."
    sim.initialize_visualizers()
    sim.set_camera_view((1.65, -1.65, 1.9), (0.38, 0.03, 0.88))
    return sim, robot, foods, joints, jaw_ops, usd_path


class PlatingController:
    """Advance waypoints only after the measured TCP reaches and settles at them."""

    def __init__(self, sim, robot, foods, joints, jaw_ops, timeout):
        import torch

        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        self.sim, self.robot, self.foods, self.joints = sim, robot, foods, joints
        self.jaw_ops = jaw_ops
        self.jaw_opening = FOODS[0].size[1] + 0.020
        self.jaw_target = self.jaw_opening
        self.timeout = timeout
        self.body_id = robot.find_bodies("link_6_t")[0][0]
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=1,
            device=sim.device,
        )
        self.offset = torch.tensor([TCP_OFFSET], device=sim.device, dtype=torch.float32)
        self.orientation = torch.tensor([EE_QUAT_XYZW], device=sim.device, dtype=torch.float32)
        self.released = set()
        self.food_index = 0
        self.phase = "approach"
        self.elapsed = self.dwell = 0.0
        self.finished = False
        self.failure = None
        self.command_pos = self.tcp_pose()[0].clone()
        self.target = self.make_target()

    def tcp_pose(self):
        """Return the world TCP position and the flange's XYZW quaternion."""
        from isaaclab.utils.math import quat_apply

        pose = self.robot.data.body_link_pose_w.torch[:, self.body_id]
        return pose[:, :3] + quat_apply(pose[:, 3:7], self.offset), pose[:, 3:7]

    def make_target(self):
        """Build a waypoint from the measured food pose or its assigned destination."""
        import torch

        food = FOODS[self.food_index]
        pick = self.foods[food.name].data.root_pose_w.torch[0, :3].tolist()
        if self.phase in ("approach", "descend", "close", "lift"):
            target = [pick[0], pick[1], HOVER_Z]
            if self.phase in ("descend", "close"):
                target[2] = pick[2] + 0.003
        else:
            target = [*food.target_xy, HOVER_Z]
            if self.phase in ("place", "open"):
                # Use the actual captured food-to-TCP offset to place its bottom above the floor.
                target[2] = TRAY_FLOOR + food.size[2] / 2 + self.grasp_height + 0.003
        return torch.tensor([target], dtype=torch.float32, device=self.sim.device)

    def attach(self) -> bool:
        """Enable ideal holding after the paddles close around the measured food pose."""
        import torch

        from isaaclab.utils.math import subtract_frame_transforms
        from pxr import Gf

        food = FOODS[self.food_index]
        ee = self.robot.data.body_link_pose_w.torch[:, self.body_id]
        portion = self.foods[food.name].data.root_pose_w.torch
        if torch.linalg.vector_norm(self.tcp_pose()[0] - portion[:, :3]).item() > 0.014:
            self.failure = f"food_moved_during_pick: {food.name}"
            self.finished = True
            return False
        local_pos, local_quat = subtract_frame_transforms(ee[:, :3], ee[:, 3:7], portion[:, :3], portion[:, 3:7])
        pos, quat = local_pos[0].tolist(), local_quat[0].tolist()
        joint = self.joints[food.name]
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*pos))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(quat[3], Gf.Vec3f(*quat[:3])))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        self.grasp_height = float((self.tcp_pose()[0][0, 2] - portion[0, 2]).item())
        joint.GetJointEnabledAttr().Set(True)
        return True

    def advance(self):
        """Attach, release, or advance to the next portion after a successful waypoint."""
        food = FOODS[self.food_index]
        if self.phase == "close" and not self.attach():
            return
        if self.phase == "place":
            self.joints[food.name].GetJointEnabledAttr().Set(False)
            self.released.add(food.name)
        phases = ("approach", "descend", "close", "lift", "transfer", "place", "open", "retreat")
        if self.phase == "retreat":
            self.food_index += 1
            self.finished = self.food_index == len(FOODS)
            self.phase = "done" if self.finished else "approach"
        else:
            self.phase = phases[phases.index(self.phase) + 1]
        active_food = FOODS[min(self.food_index, len(FOODS) - 1)]
        self.jaw_target = active_food.size[1] + (
            0.010 if self.phase in ("close", "lift", "transfer", "place") else 0.020
        )
        self.elapsed = self.dwell = 0.0
        if not self.finished:
            self.target = self.make_target()
            print(f"[bento] {FOODS[self.food_index].name}: {self.phase}", flush=True)
        else:
            print("[bento] all portions: done", flush=True)

    def step(self):
        """Compute bounded joint targets and check measured waypoint convergence."""
        import torch

        from isaaclab.utils.math import matrix_from_quat, quat_apply, skew_symmetric_matrix, subtract_frame_transforms

        dt = self.sim.get_physics_dt()
        from pxr import Gf

        self.jaw_opening += max(-0.07 * dt, min(0.07 * dt, self.jaw_target - self.jaw_opening))
        for side, op in zip((-1, 1), self.jaw_ops):
            op.Set(Gf.Vec3d(side * self.jaw_opening / 2, 0, TCP_OFFSET[2]))
        tcp_pos, ee_quat = self.tcp_pose()
        if not self.finished:
            self.elapsed += dt
            distance = torch.linalg.vector_norm(self.target - tcp_pos).item()
            angle = 2 * math.acos(min(1.0, abs(float((ee_quat * self.orientation).sum().item()))))
            speed = torch.linalg.vector_norm(self.robot.data.body_link_vel_w.torch[0, self.body_id, :3]).item()
            jaws_ready = abs(self.jaw_target - self.jaw_opening) < 0.0005
            position_tolerance = 0.003 if self.phase in ("transfer", "place") else 0.006
            settled = distance < position_tolerance and angle < 0.05 and speed < 0.035 and jaws_ready
            self.dwell = self.dwell + dt if settled else 0.0
            if self.dwell >= (0.4 if self.phase in ("close", "open") else 0.2):
                self.advance()
            elif self.elapsed > self.timeout:
                self.failure = f"waypoint_timeout: {FOODS[self.food_index].name}/{self.phase}"
                self.finished = True
            delta = self.target - self.command_pos
            # Lower bulky portions slowly and pause the command when the real arm lags.
            # Otherwise a lateral tracking error can leave the rice riding on the box rim.
            tracking_error = torch.linalg.vector_norm(self.command_pos - tcp_pos).item()
            if self.phase != "place" or tracking_error < 0.006:
                speed_limit = 0.05 if self.phase == "place" else 0.12
                self.command_pos += delta * min(
                    1.0, speed_limit * dt / max(torch.linalg.vector_norm(delta).item(), 1e-8)
                )

        root = self.robot.data.root_pose_w.torch
        pos_b, quat_b = subtract_frame_transforms(root[:, :3], root[:, 3:7], tcp_pos, ee_quat)
        goal_b, goal_quat_b = subtract_frame_transforms(root[:, :3], root[:, 3:7], self.command_pos, self.orientation)
        self.ik.set_command(torch.cat((goal_b, goal_quat_b), dim=-1))
        jacobian = self.robot.data.body_link_jacobian_w.torch[:, self.body_id - 1].clone()
        # Shift the link-origin Jacobian to the paddle center, then express it in the base frame.
        offset_w = quat_apply(ee_quat, self.offset)
        jacobian[:, :3] -= torch.bmm(skew_symmetric_matrix(offset_w), jacobian[:, 3:])
        inverse_rotation = matrix_from_quat(root[:, 3:7]).transpose(1, 2)
        jacobian[:, :3] = torch.bmm(inverse_rotation, jacobian[:, :3])
        jacobian[:, 3:] = torch.bmm(inverse_rotation, jacobian[:, 3:])
        joint_pos = self.robot.data.joint_pos.torch
        target = self.ik.compute(pos_b, quat_b, jacobian, joint_pos)
        target = joint_pos + (target - joint_pos).clamp(-0.012, 0.012)
        limits = self.robot.data.soft_joint_pos_limits.torch
        target = torch.maximum(torch.minimum(target, limits[..., 1]), limits[..., 0])
        assert torch.isfinite(target).all(), "IK returned non-finite joint targets."
        self.robot.set_joint_position_target_index(target=target)


def run(args: argparse.Namespace, app: SimulationAppContext, report: dict) -> None:
    """Run a bounded cycle and populate a report from measured final body states."""
    import torch

    sim, robot, foods, joints, jaw_ops, usd_path = create_scene(args)
    report["robot_usd"] = usd_path
    controller = PlatingController(sim, robot, foods, joints, jaw_ops, args.waypoint_timeout)
    capture = None
    if args.capture_dir is not None:
        from isaaclab_arena_examples.bento.capture import SceneCapture

        capture = SceneCapture(sim, robot, foods, args.capture_dir, args.capture_fps)
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
                robot.write_data_to_sim()
                sim.step()
                robot.update(dt)
                for portion in foods.values():
                    portion.update(dt)
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
                            food,
                            foods[food.name].data.root_pose_w.torch[0].tolist(),
                            foods[food.name].data.root_vel_w.torch[0].tolist(),
                            food.name in controller.released,
                        )
                        for food in FOODS
                    )
                    stable_time = stable_time + dt if all_ok else 0.0
                    if stable_time >= 0.5:
                        report.update(success=True, reason="all_portions_placed_and_stable")
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
        report["controller_state"] = {
            "target_tcp": controller.target[0].tolist(),
            "measured_tcp": controller.tcp_pose()[0][0].tolist(),
            "joint_positions": robot.data.joint_pos.torch[0].tolist(),
            "jaw_opening": controller.jaw_opening,
        }
        report["portions"] = {
            food.name: {
                "pose_xyzw": foods[food.name].data.root_pose_w.torch[0].tolist(),
                "velocity": foods[food.name].data.root_vel_w.torch[0].tolist(),
                "released": food.name in controller.released,
                "placement_ok": placement_ok(
                    food,
                    foods[food.name].data.root_pose_w.torch[0].tolist(),
                    foods[food.name].data.root_vel_w.torch[0].tolist(),
                    food.name in controller.released,
                ),
            }
            for food in FOODS
        }
        for joint in joints.values():
            joint.GetJointEnabledAttr().Set(False)
        if capture is not None:
            capture.record(report.get("steps", 0), controller, force=True)
            capture.save(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[bento] {report['reason']}; result: {args.output}", flush=True)
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
        "food_model": "rigid_cohesive_portions_with_detailed_visual_meshes",
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
            assert report["success"], f"Bento plating failed: {report['reason']}. See {args.output}."


if __name__ == "__main__":
    main()
