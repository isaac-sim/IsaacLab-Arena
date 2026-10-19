# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Use a pair of NHC12 arms to open an occupied shelf gap and support leaning books."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab_arena_examples.library.crowded_scene import create_crowded_scene

ORIENTATION = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
GRASP_HEIGHT = 0.095
RETURN_TARGET = (-0.19, 0.52, 0.86)


def parse_args():
    """Parse options before starting Kit."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=48000)
    parser.add_argument("--no-capture", action="store_true", help="Run a short physics check without geometry capture.")
    parser.add_argument("--stop-after", help="Finish a diagnostic run after entering the named phase.")
    parser.add_argument("--capture-fps", type=float, default=10)
    parser.add_argument("--waypoint-timeout", type=float, default=25)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cpu")
    args = parser.parse_args()
    if args.device != "cpu":
        parser.error("Use --device cpu for switchable ideal grasp constraints.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a missing or empty output directory.")
    if args.max_steps <= 0 or not 0 < args.capture_fps <= 120 or not math.isfinite(args.capture_fps):
        parser.error("max-steps and capture-fps must be positive, capture-fps at most 120.")
    if not math.isfinite(args.waypoint_timeout) or args.waypoint_timeout <= 0:
        parser.error("waypoint-timeout must be positive and finite.")
    if args.visualizer is None:
        args.visualizer = [] if args.headless else ["kit"]
    return args


def create_scene():
    """Spawn a handling arm and a contact-support arm around the occupied shelf."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

    from isaaclab_arena_examples.bento.meal import ROBOT_ASSET
    from isaaclab_arena_examples.library.scene import add_book_tongs

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cpu", use_fabric=False))
    light = sim_utils.DomeLightCfg(intensity=2500, color=(1.0, 0.96, 0.90))
    light.func("/World/Light", light)
    objects, metadata = create_crowded_scene(sim)
    robots, paths = {}, {}
    for side, position, quaternion in (
        ("left", (-0.65, -0.10, 0.75), (0, 0, 0, 1)),
        ("right", (0.60, -0.20, 1.10), (0, 0, 1, 0)),
    ):
        robot = Articulation(
            ArticulationCfg(
                prim_path="/World/" + side.capitalize() + "Robot",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/{ROBOT_ASSET}",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        fix_root_link=True,
                        enabled_self_collisions=True,
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=8,
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(pos=position, rot=quaternion, joint_pos={".*": 0.0}),
                actuators={"arm": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=800, damping=80)},
            )
        )
        robots[side] = robot
        paths[side] = next(
            str(prim.GetPath())
            for prim in Usd.PrimRange(sim.stage.GetPrimAtPath(robot.cfg.prim_path))
            if prim.GetName() == "link_6_t" and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        )
    jaw_ops = add_book_tongs(sim.stage, paths["left"], 0.18, paddle_length=0.02)
    material = sim_utils.RigidBodyMaterialCfg(static_friction=0.12, dynamic_friction=0.08, friction_combine_mode="min")
    # The thin paddle reaches below the support wrist so the handling wrist can pass above the book.
    for name, size, position in (
        ("SupportLowerStem", (0.012, 0.006, 0.20), (0, 0, -0.58)),
        ("SupportBridge", (0.012, 0.051, 0.012), (0, 0.0225, -0.48)),
        ("SupportUpperStem", (0.012, 0.006, 0.28), (0, 0.045, -0.34)),
    ):
        part = sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0),
            physics_material=material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.57, 0.64), roughness=0.5),
        )
        part.func(paths["right"] + "/" + name, part, translation=position)
    # Symmetric tapered edges enter a leaning-book gap before the full paddle width.
    paddle = UsdGeom.Mesh.Define(sim.stage, paths["right"] + "/SupportPaddle")
    outline = [(-0.065, -0.00025), (-0.065, 0.00025), (0, 0.004), (0.065, 0.00025), (0.065, -0.00025), (0, -0.004)]
    points = [Gf.Vec3f(x, y, z - 0.68) for z in (-0.07, 0.07) for x, y in outline]
    faces = [tuple(reversed(range(6))), tuple(range(6, 12))]
    faces.extend((i, (i + 1) % 6, (i + 1) % 6 + 6, i + 6) for i in range(6))
    paddle.CreatePointsAttr(points)
    paddle.CreateFaceVertexCountsAttr([len(face) for face in faces])
    paddle.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    paddle.CreateSubdivisionSchemeAttr("none")
    paddle.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.57, 0.64)])
    UsdPhysics.CollisionAPI.Apply(paddle.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(paddle.GetPrim()).CreateApproximationAttr("convexHull")
    contact = PhysxSchema.PhysxCollisionAPI.Apply(paddle.GetPrim())
    contact.CreateContactOffsetAttr(0.0002)
    contact.CreateRestOffsetAttr(0)
    physics_material = UsdShade.Material.Define(sim.stage, "/World/SupportMaterial")
    physics = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
    physics.CreateStaticFrictionAttr(0.12)
    physics.CreateDynamicFrictionAttr(0.08)
    physics.CreateRestitutionAttr(0)
    PhysxSchema.PhysxMaterialAPI.Apply(physics_material.GetPrim()).CreateFrictionCombineModeAttr("min")
    UsdShade.MaterialBindingAPI.Apply(paddle.GetPrim()).Bind(physics_material, materialPurpose="physics")
    joints = {}
    for name in ("return_book", "loan_book"):
        joint = UsdPhysics.FixedJoint.Define(sim.stage, "/World/Grasp_" + name)
        joint.CreateBody0Rel().SetTargets([paths["left"]])
        joint.CreateBody1Rel().SetTargets([str(objects[name].cfg.prim_path)])
        joint.CreateJointEnabledAttr(False)
        joint.CreateExcludeFromArticulationAttr(True)
        joint.CreateCollisionEnabledAttr(False)
        joints[name] = joint
    sim.reset()
    for robot in robots.values():
        robot.update(sim.get_physics_dt())
    for obj in objects.values():
        obj.update(sim.get_physics_dt())
    sim.initialize_visualizers()
    sim.set_camera_view((1.5, -1.7, 1.9), (0, 0.25, 1.0))
    metadata["return_book"]["target"] = list(RETURN_TARGET)
    metadata["helper_push_distance"] = 0.085
    metadata["jaw_ops"] = jaw_ops
    return sim, robots, objects, joints, metadata


class ArmIK:
    """Track a measured tool center with the six joints of one arm."""

    def __init__(self, sim, robot, side):
        import torch

        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        self.sim, self.robot = sim, robot
        self.body = robot.find_bodies("link_6_t")[0][0]
        self.joints = list(range(robot.num_joints))
        self.controller = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=1,
            device=sim.device,
        )
        self.offset = torch.tensor([[0.0, 0.0, -0.20]], device=sim.device)
        self.command = self.pose()[0].clone()
        desired = ORIENTATION if side == "left" else (0.0, 0.0, math.sqrt(0.5), -math.sqrt(0.5))
        self.orientation = torch.tensor([desired], device=sim.device, dtype=torch.float32)

    def pose(self):
        from isaaclab.utils.math import quat_apply

        value = self.robot.data.body_link_pose_w.torch[:, self.body]
        return value[:, :3] + quat_apply(value[:, 3:], self.offset), value[:, 3:]

    def step(self, target, finger, speed=0.10):
        import torch

        from isaaclab.utils.math import matrix_from_quat, quat_apply, skew_symmetric_matrix, subtract_frame_transforms

        position, quaternion = self.pose()
        target = torch.tensor([target], device=self.sim.device, dtype=torch.float32)
        delta = target - self.command
        tracking = float((self.command - position).norm())
        if tracking < 0.015:
            self.command += delta * min(1.0, speed * self.sim.get_physics_dt() / max(float(delta.norm()), 1e-8))
        root = self.robot.data.root_pose_w.torch
        local, local_q = subtract_frame_transforms(root[:, :3], root[:, 3:], position, quaternion)
        goal, goal_q = subtract_frame_transforms(root[:, :3], root[:, 3:], self.command, self.orientation)
        self.controller.set_command(torch.cat((goal, goal_q), dim=-1))
        jacobian = self.robot.data.body_link_jacobian_w.torch[:, self.body - 1].clone()
        offset = quat_apply(quaternion, self.offset)
        jacobian[:, :3] -= torch.bmm(skew_symmetric_matrix(offset), jacobian[:, 3:])
        rotation = matrix_from_quat(root[:, 3:]).transpose(1, 2)
        jacobian[:, :3] = torch.bmm(rotation, jacobian[:, :3])
        jacobian[:, 3:] = torch.bmm(rotation, jacobian[:, 3:])
        current = self.robot.data.joint_pos.torch
        goal = self.controller.compute(local, local_q, jacobian, current)
        goal = current + (goal - current).clamp(-0.012, 0.012)
        limits = self.robot.data.soft_joint_pos_limits.torch
        goal = torch.maximum(torch.minimum(goal, limits[..., 1]), limits[..., 0])
        assert torch.isfinite(goal).all(), "Nonfinite arm target"
        self.robot.set_joint_position_target_index(target=goal)
        distance = float((target - position).norm())
        angle = 2 * math.acos(min(1.0, abs(float((quaternion * self.orientation).sum()))))
        velocity = float(self.robot.data.body_link_vel_w.torch[0, self.body, :3].norm())
        return distance, angle, velocity


class CrowdedController:
    """Open a real gap with the right arm and manipulate books with the left arm."""

    def __init__(self, sim, robot, objects, joints, metadata, timeout):
        self.sim, self.robot, self.objects, self.joints, self.metadata = sim, robot, objects, joints, metadata
        self.left, self.right = ArmIK(sim, robot["left"], "left"), ArmIK(sim, robot["right"], "right")
        self.jaw_ops = metadata.pop("jaw_ops")
        self.jaw_opening = 0.18
        self.timeout = timeout
        self.index = 0
        self.elapsed = self.dwell = 0.0
        self.failure = None
        self.finished = False
        self.released = set()
        self.events = []
        self.max_gap = self.gap()
        self.initial_gap = self.gap()
        self.hold_offset = GRASP_HEIGHT
        self.targets = {}
        self.phases = [
            "home",
            "pickup_hover",
            "pickup_return",
            "grip_return",
            "lift_return",
            "carry_return",
            "helper_approach_gap",
            "helper_enter_gap",
            "helper_spread_gap",
            "align_return",
            "insert_return",
            "seat_return",
            "release_return",
            "clear_return",
            "withdraw_return",
            "helper_unload",
            "helper_retract",
            "helper_park",
            "helper_rotate_mid",
            "helper_rotate_loan",
            "helper_approach_lean",
            "helper_enter_lean",
            "helper_support_lean",
            "loan_hover",
            "loan_align",
            "loan_grasp",
            "grip_loan",
            "raise_loan",
            "extract_loan",
            "loan_carry",
            "loan_box_hover",
            "loan_box_lower",
            "release_loan",
            "loan_clear",
            "done",
        ]
        self.left_target = (-0.30, -0.05, 1.12)
        self.right_target = (0.30, 0.12, 1.43)
        self.left_finger = self.right_finger = 0.04
        self.set_phase_targets()

    @property
    def phase(self):
        return self.phases[self.index]

    def gap(self):
        """Measure free width over the insertion depth, accounting for displaced or rotated neighbors."""
        import torch

        from isaaclab.utils.math import matrix_from_quat

        left_names = self.metadata["return_left_neighbors"]
        right_names = self.metadata["return_right_neighbors"]
        poses = torch.cat([self.objects[name].data.root_pose_w.torch for name in left_names + right_names], dim=0)
        rotations = matrix_from_quat(poses[:, 3:])
        half = torch.tensor([0.0175, 0.08, 0.11], device=self.sim.device)
        extents = torch.matmul(rotations.abs(), half)
        lower = poses[:, :3] - extents
        upper = poses[:, :3] + extents
        in_lane = (upper[:, 1] > 0.438) & (lower[:, 1] < 0.602) & (upper[:, 2] > 0.75) & (lower[:, 2] < 0.98)
        split = len(left_names)
        left = upper[:split, 0][in_lane[:split]]
        right = lower[split:, 0][in_lane[split:]]
        left_edge = float(left.max()) if len(left) else -0.45
        right_edge = float(right.min()) if len(right) else 0.45
        return right_edge - left_edge

    def measured(self, name):
        return self.objects[name].data.root_pose_w.torch[0, :3].tolist()

    def set_phase_targets(self):
        """Set front-entry waypoints; neighboring books are never assigned scripted poses."""
        p = self.phase
        if p == "helper_spread_gap":
            self.gap_before_push = self.gap()
            self.row_before_push = self.measured(self.metadata["return_right_neighbors"][0])
        if p == "helper_approach_lean":
            import torch

            from isaaclab.utils.math import matrix_from_quat

            pose = self.objects[self.metadata["leaning_neighbor"]].data.root_pose_w.torch[0]
            rotation = matrix_from_quat(pose[3:])
            half = torch.tensor([0.0175, 0.08, 0.11], device=self.sim.device)
            vertical = rotation[2, :].abs()
            top = float(pose[2] + (vertical * half).sum())
            center_top = float(pose[2] + (half / vertical.clamp_min(1e-6)).min())
            self.support_x = float(pose[0])
            self.support_y = float(pose[1])
            self.support_clear_z = top + 0.55 + 0.025
            self.support_touch_z = center_top + 0.55 - 0.001
        if p == "helper_support_lean":
            self.lean_before_support = self.measured(self.metadata["leaning_neighbor"])
        source = self.measured("return_book")
        loan = self.measured("loan_book")
        held_z = 0.86 + self.hold_offset
        left_targets = {
            "pickup_hover": (-0.30, -0.25, 1.10),
            "pickup_return": (source[0], source[1], source[2] + GRASP_HEIGHT),
            "grip_return": (source[0], source[1], source[2] + GRASP_HEIGHT),
            "lift_return": (-0.30, -0.25, 1.12),
            "carry_return": (-0.30, 0.10, 1.15),
            "align_return": (-0.19, 0.25, held_z + 0.04),
            "insert_return": (-0.19, 0.52, held_z + 0.04),
            "seat_return": (-0.19, 0.52, held_z + 0.003),
            "release_return": (-0.19, 0.52, held_z + 0.003),
            "clear_return": (-0.19, 0.52, 1.01),
            "withdraw_return": (-0.30, 0.10, 1.15),
            "loan_hover": (loan[0], 0.25, 1.10),
            "loan_align": (loan[0], 0.25, loan[2] + GRASP_HEIGHT),
            "loan_grasp": (loan[0], loan[1], loan[2] + GRASP_HEIGHT),
            "grip_loan": (loan[0], loan[1], loan[2] + GRASP_HEIGHT),
            "raise_loan": (loan[0], loan[1], loan[2] + GRASP_HEIGHT + 0.04),
            "extract_loan": (loan[0], 0.25, loan[2] + GRASP_HEIGHT),
            "loan_carry": (0.20, 0.15, 1.12),
            "loan_box_hover": (0.30, -0.25, 1.12),
            "loan_box_lower": (0.30, -0.25, held_z + 0.003),
            "release_loan": (0.30, -0.25, held_z + 0.003),
            "loan_clear": (0.30, -0.25, 1.12),
            "done": (0.30, -0.25, 1.12),
        }
        right_targets = {
            "helper_approach_gap": (-0.2100, 0.25, 1.33),
            "helper_enter_gap": (-0.2100, 0.52, 1.33),
            "helper_spread_gap": (-0.1250, 0.52, 1.33),
            "helper_unload": (-0.145, 0.52, 1.33),
            "helper_retract": (-0.145, 0.23, 1.33),
            "helper_park": (0.30, 0.12, 1.43),
        }
        if p == "helper_approach_lean":
            right_targets[p] = (self.support_x, 0.25, self.support_clear_z)
        if p == "helper_enter_lean":
            right_targets[p] = (self.support_x, self.support_y, self.support_clear_z)
        if p == "helper_support_lean":
            right_targets[p] = (self.support_x, self.support_y, self.support_touch_z)
        if p in left_targets:
            self.left_target = left_targets[p]
        if p in right_targets:
            self.right_target = right_targets[p]
        if p == "helper_rotate_mid":
            import torch

            self.right.orientation = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=self.sim.device)
        if p == "helper_rotate_loan":
            import torch

            self.right.orientation = torch.tensor([ORIENTATION], device=self.sim.device, dtype=torch.float32)
        if p in (
            "grip_return",
            "lift_return",
            "carry_return",
            "helper_approach_gap",
            "helper_enter_gap",
            "helper_spread_gap",
            "align_return",
            "insert_return",
            "seat_return",
        ):
            self.left_finger = 0.019
        elif p in ("release_return", "helper_unload", "helper_retract", "clear_return"):
            self.left_finger = 0.0215
        elif p in ("grip_loan", "raise_loan", "extract_loan", "loan_carry", "loan_box_hover", "loan_box_lower"):
            self.left_finger = 0.019
        elif p == "release_loan":
            self.left_finger = 0.0215
        else:
            self.left_finger = 0.04
        self.right_finger = 0.0
        print("[crowded] " + p, flush=True)

    def attach(self, name):
        """Enable ideal holding after the physical gripper reaches the measured upper book region."""
        from isaaclab.utils.math import subtract_frame_transforms
        from pxr import Gf

        pose = self.objects[name].data.root_pose_w.torch
        grasp = pose[:, :3].clone()
        grasp[:, 2] += GRASP_HEIGHT
        if float((self.left.pose()[0] - grasp).norm()) > 0.025:
            self.failure = "book_moved_before_grasp: " + name
            self.finished = True
            return
        tool = self.left.robot.data.body_link_pose_w.torch[:, self.left.body]
        position, quaternion = subtract_frame_transforms(tool[:, :3], tool[:, 3:], pose[:, :3], pose[:, 3:])
        xyz, q = position[0].tolist(), quaternion[0].tolist()
        joint = self.joints[name]
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*xyz))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1))
        joint.GetJointEnabledAttr().Set(True)
        self.hold_offset = float(self.left.pose()[0][0, 2] - pose[0, 2])
        self.events.append({"event": "grasp", "book": name})

    def advance(self):
        if self.phase == "grip_return":
            self.attach("return_book")
        if self.phase == "grip_loan":
            self.attach("loan_book")
        if self.failure:
            return
        if self.phase in ("release_return", "release_loan"):
            name = "return_book" if self.phase == "release_return" else "loan_book"
            self.joints[name].GetJointEnabledAttr().Set(False)
            self.released.add(name)
            self.events.append({"event": "released", "book": name})
        if self.phase == "helper_spread_gap":
            if self.gap() < 0.060:
                self.failure = f"gap_not_opened: {self.gap():.4f}m"
                self.finished = True
                return
            self.events.append({
                "event": "gap_opened_by_contact",
                "gap_before_m": self.gap_before_push,
                "gap_m": self.gap(),
                "right_neighbor_x_displacement_m": (
                    self.measured(self.metadata["return_right_neighbors"][0])[0] - self.row_before_push[0]
                ),
            })
        if self.phase == "helper_support_lean":
            self.events.append({
                "event": "leaning_neighbor_supported",
                "before": self.lean_before_support,
                "after": self.measured(self.metadata["leaning_neighbor"]),
            })
        self.index += 1
        self.elapsed = self.dwell = 0.0
        self.finished = self.phase == "done"
        self.set_phase_targets()

    def step(self):
        from pxr import Gf

        dt = self.sim.get_physics_dt()
        opening = 0.17 if self.left_finger < 0.025 else 0.18
        self.jaw_opening += max(-0.07 * dt, min(0.07 * dt, opening - self.jaw_opening))
        for side, op in zip((-1, 1), self.jaw_ops):
            op.Set(Gf.Vec3d(side * self.jaw_opening / 2, 0, -0.20))
        self.elapsed += dt
        careful = self.phase in (
            "helper_spread_gap",
            "helper_support_lean",
            "insert_return",
            "seat_return",
            "loan_grasp",
            "loan_box_lower",
            "raise_loan",
            "extract_loan",
        )
        left = self.left.step(self.left_target, self.left_finger, 0.035 if careful else 0.10)
        right = self.right.step(
            self.right_target,
            self.right_finger,
            (
                0.012
                if self.phase in ("helper_spread_gap", "helper_support_lean")
                else 0.02 if self.phase in ("helper_enter_gap", "helper_enter_lean") else 0.08
            ),
        )
        self.max_gap = max(self.max_gap, self.gap())
        if self.finished:
            return
        precise = self.phase in ("helper_approach_gap", "helper_enter_gap", "helper_approach_lean", "helper_enter_lean")
        # The support is allowed to stop against a book before its nominal depth.
        tolerance = 0.012 if self.phase == "helper_support_lean" else 0.0025 if precise else 0.006
        angle_tolerance = 0.03 if precise else 0.08
        support_active = any(event["event"] == "leaning_neighbor_supported" for event in self.events)
        settled = all(
            distance < allowed and angle < angle_tolerance and speed < 0.045
            for (distance, angle, speed), allowed in (
                (left, tolerance),
                (right, 0.012 if support_active else tolerance),
            )
        )
        self.dwell = self.dwell + self.sim.get_physics_dt() if settled else 0.0
        if self.dwell >= (
            0.8 if self.phase in ("grip_return", "grip_loan", "release_return", "release_loan") else 0.35
        ):
            self.advance()
        elif self.elapsed > self.timeout:
            self.failure = f"waypoint_timeout: {self.phase}; left={left}; right={right}"
            self.finished = True


def run(args, app, report):
    """Record actual contact motion and validate the opened gap and final book destinations."""
    import torch

    from isaaclab_arena_examples.library.crowded_capture import GenericCapture

    sim, robot, objects, joints, metadata = create_scene()
    ctrl = CrowdedController(sim, robot, objects, joints, metadata, args.waypoint_timeout)
    capture = None if args.no_capture else GenericCapture(sim, robot, objects, args.output_dir, args.capture_fps)
    dt = sim.get_physics_dt()
    if capture is not None:
        capture.record(0, ctrl.phase, {"gap_m": ctrl.gap()})
    settle = 0.0
    try:
        with torch.inference_mode():
            for step in range(args.max_steps):
                if not app.is_running() or app.is_exiting():
                    report["reason"] = "window_closed"
                    break
                if step >= 120:
                    ctrl.step()
                for arm in robot.values():
                    arm.write_data_to_sim()
                sim.step()
                for arm in robot.values():
                    arm.update(dt)
                for obj in objects.values():
                    obj.update(dt)
                report["steps"] = step + 1
                if capture is not None:
                    capture.record(step + 1, ctrl.phase, {"gap_m": ctrl.gap()})
                if args.stop_after and ctrl.phase == args.stop_after:
                    report.update(probe=True, reason="probe_complete: " + ctrl.phase)
                    break
                if ctrl.failure:
                    report["reason"] = ctrl.failure
                    break
                if ctrl.finished:
                    settle += dt
                    if settle > 1.0:
                        return_pose = objects["return_book"].data.root_pose_w.torch[0]
                        loan_pose = objects["loan_book"].data.root_pose_w.torch[0]
                        # Lateral contact may shift a seated book within its assigned slot.
                        return_ok = all(
                            abs(float(return_pose[i]) - RETURN_TARGET[i]) < limit
                            for i, limit in enumerate((0.020, 0.015, 0.015))
                        )
                        loan_ok = (
                            abs(float(loan_pose[0]) - 0.30) < 0.055
                            and abs(float(loan_pose[1]) + 0.25) < 0.035
                            and abs(float(loan_pose[2]) - 0.86) < 0.02
                        )
                        stable = all(
                            float(objects[n].data.root_vel_w.torch.norm()) < 0.10 for n in ("return_book", "loan_book")
                        )
                        ok = return_ok and loan_ok and stable and len(ctrl.released) == 2 and ctrl.max_gap > 0.060
                        report.update(
                            success=ok,
                            reason="contact_gap_and_supported_loan_completed" if ok else "final_validation_failed",
                        )
                        break
            else:
                report["reason"] = "max_steps"
    finally:
        report.update(
            phase=ctrl.phase,
            events=ctrl.events,
            initial_gap_m=ctrl.initial_gap,
            max_gap_m=ctrl.max_gap,
            final_gap_m=ctrl.gap(),
            scene=metadata,
            gap_before_push_m=getattr(ctrl, "gap_before_push", None),
            row_before_push=getattr(ctrl, "row_before_push", None),
            grasp_model="ideal_fixed_constraint_with_visual_tongs",
            neighbor_model="unconstrained_dynamic_rigid_books_with_contact_support",
        )
        report["objects"] = {
            name: {
                "pose_xyzw": obj.data.root_pose_w.torch[0].tolist(),
                "velocity": obj.data.root_vel_w.torch[0].tolist(),
            }
            for name, obj in objects.items()
        }
        report["arms"] = {
            side: {
                "joints": arm.data.joint_pos.torch[0].tolist(),
                "root_pose": arm.data.root_pose_w.torch[0].tolist(),
                "ee_pose": arm.data.body_link_pose_w.torch[0, arm.find_bodies("link_6_t")[0][0]].tolist(),
            }
            for side, arm in robot.items()
        }
        report["tcp"] = {
            "left": ctrl.left.pose()[0][0].tolist(),
            "right": ctrl.right.pose()[0][0].tolist(),
            "target_left": ctrl.left_target,
            "target_right": ctrl.right_target,
            "command_left": ctrl.left.command[0].tolist(),
            "command_right": ctrl.right.command[0].tolist(),
        }
        if capture is None and ctrl.failure:
            capture = GenericCapture(sim, robot, objects, args.output_dir, args.capture_fps)
        if capture is not None:
            capture.record(report.get("steps", 0), ctrl.phase, {"gap_m": ctrl.gap()}, force=True)
            capture.save(report)
        (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"success": report["success"], "reason": report["reason"], "directory": str(args.output_dir)}),
        flush=True,
    )


def main():
    """Start the existing runtime and fail the process if the physical task did not complete."""
    args = parse_args()
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    report = {"success": False, "reason": "initialization_failed"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with SimulationAppContext(args) as app:
        try:
            run(args, app, report)
        finally:
            (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        assert report["success"] or report.get("probe"), report["reason"]


if __name__ == "__main__":
    main()
