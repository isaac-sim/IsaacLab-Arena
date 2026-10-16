# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measured-state BURVIK assembly using only YAM TCP poses and finger commands."""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.embodiments.yam.control import absolute_pose_actions, tcp_pose_w
from isaaclab_arena.embodiments.yam.gripper import YamGripper
from isaaclab_arena.embodiments.yam.kinematics import YamForwardKinematics
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena.tasks.predicates.gripper import gripper_not_grasping, gripper_released
from isaaclab_arena.tasks.predicates.spatial import contact_force_is_upward_support
from isaaclab_arena_environments.burvik_assembly.connections import attachment_enabled
from isaaclab_arena_environments.burvik_assembly.geometry import attachment_pose
from isaaclab_arena_environments.burvik_assembly.task import installed_part


@dataclass
class BurvikYamAssemblyPolicyCfg(PolicyCfg):
    """Configure the deterministic, single-environment manipulation policy."""

    specification_path: str = "artifacts/ikea_burvik/specification.json"
    translation_speed: float = 0.10
    rotation_speed: float = 0.8
    carry_pose_gain: float = 0.2
    """Fraction of measured hand error commanded while supporting the shared assembly."""
    stop_after_leg: int = 0
    """Diagnostic partial rollout; zero requests the entire task."""


def compose(a, b):
    """Compose T_A_B and T_B_C, represented by position and scipy rotation."""
    return a[0] + a[1].apply(b[0]), a[1] * b[1]


def relative(a, b):
    """Return T_A_B from world poses T_W_A and T_W_B."""
    return a[1].inv().apply(b[0] - a[0]), a[1].inv() * b[1]


@register_policy
class BurvikYamAssemblyPolicy(PolicyBase[BurvikYamAssemblyPolicyCfg]):
    """Execute guarded manipulation states through Arena's pose-action interface."""

    name = "burvik_yam_assembly"

    def __init__(self, config: BurvikYamAssemblyPolicyCfg):
        super().__init__(config)
        self.spec = json.loads(Path(config.specification_path).read_text())
        assert 0 < config.carry_pose_gain <= 1, "Carry pose gain must be in (0, 1]"
        self.hands = {
            side: YamGripper(articulation_name=f"{side}_robot", frame_transformer_name=f"{side}_ee_frame")
            for side in ("left", "right")
        }
        self.reset()

    def reset(self, env_ids=None):
        self.program = None
        self.steps = 0
        self.phase = "initialize"
        self.transitions = []
        self.finished = False
        self.failure = None
        self.grippers = {"left": 0.0, "right": 0.0}
        self.compliant_hands = set()
        self.carry_z_bias = 0.0

    def pose(self, name):
        T_W_P = self.base.arena_world.get_pose_w(name)[0].cpu().numpy()
        return T_W_P[:3].copy(), Rotation.from_quat(T_W_P[3:])

    def tcp(self, side):
        T_W_G = tcp_pose_w(self.base, side)[0].cpu().numpy()
        return T_W_G[:3].copy(), Rotation.from_quat(T_W_G[3:])

    def enter(self, phase):
        self.phase = phase
        self.transitions.append({"state": phase, "time_s": self.steps * self.base.step_dt})
        print(f"[YAM POLICY] {self.steps * self.base.step_dt:.2f}s {phase}", flush=True)

    def hold(self, phase, seconds):
        self.enter(phase)
        for _ in range(round(seconds / self.base.step_dt)):
            yield

    def check_grasp(self, name, side, closing_error=0.002):
        """Require measured jaw width and a closing stall before carrying an object."""
        width = 0.012 if name == "handle" else 0.018
        if bool(gripper_not_grasping(self.base, self.hands[side], width, 0.008, closing_error)[0]):
            raise RuntimeError(f"{name}: {side} fingers did not close around the tube")

    def move_grippers(self, phase, sides, command):
        """Ramp finger commands to limit contact impulses during grasp and release."""
        self.enter(phase)
        initial = {side: self.grippers[side] for side in sides}
        ticks = round(0.8 / self.base.step_dt)
        for tick in range(ticks):
            t = (tick + 1) / ticks
            fraction = t * t * (3 - 2 * t)
            for side in sides:
                self.grippers[side] = initial[side] + fraction * (command - initial[side])
            yield
        yield from self.hold(f"{phase}: settle", 0.2)

    def flip_wrist(self, side):
        """Reorient open symmetric jaws through the wrist's neutral angle."""
        robot = self.base.scene[f"{side}_robot"]
        index = robot.joint_names.index("joint6")
        direction = -1.0 if float(robot.data.joint_pos.torch[0, index]) > 0 else 1.0
        p, r = self.tcp(side)
        for angle in (90, 180):
            rotation = r * Rotation.from_euler("z", direction * angle, degrees=True)
            yield from self.move(
                f"turn: unwind {side} wrist {angle} degrees", {side: (p, rotation)}, orientation_tolerance_rad=0.06
            )
            self.check_installed_legs()

    def close_on_table(self, phase, side, T_T_G):
        """Follow the measured table pose while closing on an attached leg."""
        self.enter(f"{phase}: align with moving leg")
        aligned_steps = 0
        for _ in range(round(4.0 / self.base.step_dt)):
            goal = compose(self.pose("tray"), T_T_G)
            self.targets[side] = goal
            actual = self.tcp(side)
            aligned = np.linalg.norm(goal[0] - actual[0]) < 0.0015 and (goal[1] * actual[1].inv()).magnitude() < 0.035
            aligned_steps = aligned_steps + 1 if aligned else 0
            if aligned_steps >= round(0.15 / self.base.step_dt):
                break
            yield
        else:
            raise RuntimeError(f"{phase}: open jaws did not align with the moving leg")
        for _ in self.move_grippers(phase, [side], 0.9):
            self.targets[side] = compose(self.pose("tray"), T_T_G)
            yield
        self.check_installed_legs()

    def move_with_table(self, phase, side, T_T_G):
        """Approach a table-relative grasp while following the table's live pose."""
        self.enter(phase)
        start = relative(self.pose("tray"), self.tcp(side))
        duration = max(
            0.4,
            1.5 * np.linalg.norm(T_T_G[0] - start[0]) / self.config.translation_speed,
            1.5 * (T_T_G[1] * start[1].inv()).magnitude() / self.config.rotation_speed,
        )
        curve = Slerp([0, 1], Rotation.concatenate([start[1], T_T_G[1]]))
        ticks = max(1, round(duration / self.base.step_dt))
        for tick in range(ticks):
            t = (tick + 1) / ticks
            fraction = t * t * (3 - 2 * t)
            intermediate = start[0] + fraction * (T_T_G[0] - start[0]), curve([fraction])[0]
            self.targets[side] = compose(self.pose("tray"), intermediate)
            yield
        for _ in range(round(4.0 / self.base.step_dt)):
            goal = compose(self.pose("tray"), T_T_G)
            self.targets[side] = goal
            actual = self.tcp(side)
            if np.linalg.norm(goal[0] - actual[0]) < 0.004 and (goal[1] * actual[1].inv()).magnitude() < 0.035:
                return
            yield
        raise RuntimeError(f"{phase}: hand did not converge to the moving table")

    def move_table(self, phase, goal, duration=1.0, tolerance=0.012):
        """Move one object-frame trajectory through the two fixed measured grasps."""
        self.enter(phase)
        start = self.pose("tray")
        duration = max(
            duration,
            1.5 * np.linalg.norm(goal[0] - start[0]) / min(self.config.translation_speed, 0.05),
            1.5 * (goal[1] * start[1].inv()).magnitude() / min(self.config.rotation_speed, 0.5),
        )
        curve = Slerp([0, 1], Rotation.concatenate([start[1], goal[1]]))
        ticks = max(1, round(duration / self.base.step_dt))
        for tick in range(ticks + round(6.0 / self.base.step_dt)):
            self.check_installed_legs()
            fraction = min(1.0, (tick + 1) / ticks)
            fraction = fraction * fraction * (3 - 2 * fraction)
            wanted = start[0] + fraction * (goal[0] - start[0]), curve([fraction])[0]
            if tick >= ticks:
                # Reject payload sag with one common vertical correction, preserving the two grasp transforms.
                self.carry_z_bias = float(
                    np.clip(
                        self.carry_z_bias + 0.5 * self.base.step_dt * (goal[0][2] - self.pose("tray")[0][2]),
                        -0.01,
                        0.01,
                    )
                )
            wanted[0][2] += self.carry_z_bias
            for side, grasp in self.carry_grasps.items():
                self.targets[side] = compose(wanted, grasp)
                self.check_grasp("leg_4" if side == "left" else "leg_2", side)
            actual = self.pose("tray")
            if (
                tick >= ticks
                and np.linalg.norm(goal[0] - actual[0]) < tolerance
                and (goal[1] * actual[1].inv()).magnitude() < 0.04
            ):
                return
            yield
        raise RuntimeError(f"{phase}: table pose did not converge")

    def check_installed_legs(self):
        for index in range(1, 5):
            name = f"leg_{index}"
            p, r = attachment_pose(name, self.spec)
            if not bool(installed_part(self.base, name, p.tolist(), r.as_quat().tolist())[0]):
                raise RuntimeError(f"{self.phase}: retention lost on {name}")

    def move(self, phase, goals, duration=None, tolerance=0.004, orientation_tolerance_rad=0.035):
        """Interpolate pose commands, then require measured hand convergence."""
        self.enter(phase)
        starts = {side: self.tcp(side) for side in goals}
        duration = duration or 0.4
        curves = {}
        for side, goal in goals.items():
            start = starts[side]
            duration = max(
                duration,
                1.5 * np.linalg.norm(goal[0] - start[0]) / self.config.translation_speed,
                1.5 * (goal[1] * start[1].inv()).magnitude() / self.config.rotation_speed,
            )
            curves[side] = Slerp([0, 1], Rotation.concatenate([start[1], goal[1]]))
        ticks = max(1, round(duration / self.base.step_dt))
        for tick in range(ticks):
            t = (tick + 1) / ticks
            fraction = t * t * (3 - 2 * t)
            for side, goal in goals.items():
                self.targets[side] = (
                    starts[side][0] + fraction * (goal[0] - starts[side][0]),
                    curves[side]([fraction])[0],
                )
            yield
        errors = {}
        for _ in range(round(4.0 / self.base.step_dt)):
            errors = {}
            for side, goal in goals.items():
                actual = self.tcp(side)
                errors[side] = (
                    float(np.linalg.norm(goal[0] - actual[0])),
                    float((goal[1] * actual[1].inv()).magnitude()),
                )
            if all(p < tolerance and r < orientation_tolerance_rad for p, r in errors.values()):
                return
            yield
        raise RuntimeError(f"{phase}: TCP failed to converge: {errors}")

    def move_empty_arm(self, phase, side, joint_goal):
        """Follow reachable FK waypoints using pose actions while the fingers are open."""
        assert self.grippers[side] == 0.0, "Only use the clearance path with an open gripper"
        robot = self.base.scene[f"{side}_robot"]
        ids = [robot.joint_names.index(f"joint{i}") for i in range(1, 7)]
        start = robot.data.joint_pos.torch[0, ids].cpu().numpy().copy()
        limits = robot.data.soft_joint_pos_limits.torch[0, ids].cpu().numpy()
        goal = np.asarray(joint_goal)
        assert ((goal >= limits[:, 0]) & (goal <= limits[:, 1])).all(), "Clearance configuration exceeds joint limits"
        model = YamForwardKinematics(self.base, side)
        position, rotation = model.tcp_pose(start)
        actual = self.tcp(side)
        assert np.linalg.norm(position - actual[0]) < 0.0001, "YAM FK and measured TCP frames disagree"
        assert (rotation * actual[1].inv()).magnitude() < 0.001, "YAM FK and measured TCP orientations disagree"
        self.enter(phase)
        ticks = round(max(1.0, 1.5 * np.abs(goal - start).max() / 0.6) / self.base.step_dt)
        for tick in range(ticks):
            t = (tick + 1) / ticks
            self.targets[side] = model.tcp_pose(start + t * t * (3 - 2 * t) * (goal - start))
            yield
        yield from self.move(f"{phase}: settle", {side: model.tcp_pose(goal)})

    def grasp(self, name, side, point, rotation):
        """Close on the measured tube and verify that a lift carries the part."""
        part = self.pose(name)
        target = part[0] + part[1].apply(point) + [0, 0, 0.003 if name == "handle" else 0.011], rotation
        self.grippers[side] = 0.0
        if name == "handle":
            tcp = self.tcp(side)
            raised = tcp[0].copy()
            raised[2] = max(raised[2], 1.20)
            yield from self.move("handle: raise empty hand", {side: (raised, tcp[1])})
            above = target[0].copy()
            above[2] = 1.00
            yield from self.move("handle: above pickup", {side: (above, target[1])})
        yield from self.move(f"{name}: pregrasp", {side: (target[0] + [0, 0, 0.06], target[1])})
        # Re-measure after settling, before approaching the tube.
        part = self.pose(name)
        target = part[0] + part[1].apply(point) + [0, 0, 0.003 if name == "handle" else 0.011], rotation
        yield from self.move(f"{name}: approach", {side: target}, tolerance=0.0015)
        yield from self.move_grippers(f"{name}: close {side} fingers", [side], 1.0 if name == "handle" else 0.9)
        self.check_grasp(name, side)
        initial_z = self.pose(name)[0][2]
        tcp = self.tcp(side)
        if name == "handle":
            yield from self.move("handle: lift clear of presentation rails", {side: (tcp[0] + [0, 0, 0.06], tcp[1])})
            self.check_grasp(name, side)
            tcp = self.tcp(side)
            target = compose(self.pose("tray"), attachment_pose(name, self.spec))
            pre = target[0] + 0.07 * target[1].apply([0, 0, 1]), target[1]
            yield from self.transport_handle(side, pre, relative(self.pose(name), tcp))
        else:
            lift_offset = [0, 0.06 if side == "left" else -0.06, 0.15]
            yield from self.move(f"{name}: lift", {side: (tcp[0] + lift_offset, tcp[1])})
        lifted = self.pose(name)[0][2] - initial_z
        if lifted < 0.10:
            raise RuntimeError(f"{name}: grasp lost; part lifted only {lifted:.4f} m")
        self.check_grasp(name, side)
        return relative(self.pose(name), self.tcp(side))

    def seat(self, name, side, T_P_G):
        """Servo the held part toward its live socket until its physical latch engages."""
        T_T_P = attachment_pose(name, self.spec)
        target = compose(self.pose("tray"), T_T_P)
        outward = target[1].apply([0, 0, 1 if name == "handle" else -1])
        pre = target[0] + 0.07 * outward, target[1]
        pre_tcp = compose(pre, T_P_G)
        if name != "handle":
            yield from self.move(f"{name}: above socket", {side: (pre_tcp[0] + [0, 0, 0.06], pre_tcp[1])})
        self.check_grasp(name, side)
        yield from self.move(f"{name}: align socket", {side: pre_tcp})
        self.check_grasp(name, side)
        self.enter(f"{name}: insert")
        if name == "handle":
            self.compliant_hands.add(side)
        duration = 10.0 if name == "handle" else 6.0
        handle_offset = 0.07
        position_bias = np.zeros(3)
        rotation_bias = np.zeros(3)
        for tick in range(round((40.0 if name == "handle" else 16.0) / self.base.step_dt)):
            self.check_grasp(name, side)
            if bool(attachment_enabled(self.base, name)[0]):
                yield from self.hold(f"{name}: retention verified", 0.2)
                yield from self.move_grippers(f"{name}: release", [side], 0.0)
                width = 0.012 if name == "handle" else 0.018
                if not bool(gripper_released(self.base, self.hands[side], width)[0]):
                    raise RuntimeError(f"{name}: fingers did not release")
                tcp = self.tcp(side)
                retreat = tcp[0] - tcp[1].apply([0, 0, 0.10 if name == "handle" else 0.035])
                if name != "handle":
                    retreat += [0, 0, 0.08]
                yield from self.move(f"{name}: retreat", {side: (retreat, tcp[1])})
                return
            target = compose(self.pose("tray"), T_T_P)
            fraction = min(1.0, tick * self.base.step_dt / duration)
            offset = 0.07 * (1.0 - fraction)
            outward = target[1].apply([0, 0, 1 if name == "handle" else -1])
            measured = self.pose(name)
            if name == "handle":
                error = target[0] - measured[0]
                radial = error - np.dot(error, outward) * outward
                angular = (target[1] * measured[1].inv()).magnitude()
                # Align both pins before entering either bore; advance only while alignment holds.
                if np.linalg.norm(radial) < 0.0003 and angular < np.deg2rad(0.2):
                    handle_offset = max(0.0, handle_offset - 0.007 * self.base.step_dt)
                offset = handle_offset
            # State feedback corrects hand tracking error and minor slip without moving a rigid body directly.
            measured_grasp = relative(measured, self.tcp(side))
            self.targets[side] = compose((target[0] + offset * outward, target[1]), measured_grasp)
            if fraction == 1.0 or name == "handle":
                # Bounded integral correction rejects static servo droop at the narrow pin clearance.
                position_bias += 3.0 * self.base.step_dt * (target[0] + offset * outward - measured[0])
                position_bias *= min(1.0, 0.003 / max(np.linalg.norm(position_bias), 1e-9))
                rotation_bias += 2.0 * self.base.step_dt * (target[1] * measured[1].inv()).as_rotvec()
                rotation_bias *= min(1.0, 0.02 / max(np.linalg.norm(rotation_bias), 1e-9))
                hand_p, hand_r = self.targets[side]
                self.targets[side] = hand_p + position_bias, Rotation.from_rotvec(rotation_bias) * hand_r
            yield
        raise RuntimeError(f"{name}: pin did not seat")

    def transport_handle(self, side, goal, grasp):
        """Keep the handle's full bounds clear of the bench and the tray's outer face."""
        start = self.pose("handle")
        corners = self.base.arena_world.get_aabb_in_local_frame("handle").get_corners_at()[0].cpu().numpy()
        goals = self.handle_transfer_waypoints(start, goal, grasp, corners, self.pose("tray")[0][0])
        for index, hand in enumerate(goals, 1):
            yield from self.move(f"handle: transfer {index}/16", {side: hand}, duration=0.5)
            self.check_grasp("handle", side)

    @staticmethod
    def handle_transfer_waypoints(start, goal, grasp, corners, tray_x):
        """Plan clearance-aware hand poses through the right arm's reachable folded workspace."""
        start_hand = compose(start, grasp)
        goal_hand = compose(goal, grasp)
        curve = Slerp([0, 1], Rotation.concatenate([start[1], goal[1]]))
        for index in range(1, 17):
            fraction = index / 16
            rotation = curve([fraction])[0]
            bounds = rotation.apply(corners)
            position = start[0] + fraction * (goal[0] - start[0])
            clearance = 0.79 + 0.03 * min(1.0, 2 * fraction)
            position[2] = max(position[2], clearance - bounds[:, 2].min())
            # Withdraw toward the right base while the hand points down; extend once it tilts sideways.
            stage_xy = np.array([-0.22, -0.48])
            t = fraction / 0.25 if fraction <= 0.25 else min(1.0, (fraction - 0.25) / 0.5)
            blend = t * t * (3 - 2 * t)
            xy_a, xy_b = (start_hand[0][:2], stage_xy) if fraction <= 0.25 else (stage_xy, goal_hand[0][:2])
            hand_xy = xy_a + blend * (xy_b - xy_a)
            position[:2] = hand_xy - rotation.apply(grasp[0])[:2]
            position[0] = min(position[0], tray_x - 0.045 - bounds[:, 0].max())
            yield compose((position, rotation), grasp)

    def assemble(self):
        yield from self.hold("settle", 1.0)
        self.homes = {side: (self.tcp(side)[0] + [0, 0, 0.12], self.tcp(side)[1]) for side in self.targets}
        yield from self.move("clear workbench", self.homes)
        for index, side in ((1, "left"), (2, "right"), (3, "left"), (4, "left")):
            name = f"leg_{index}"
            local_p, local_r = attachment_pose(name, self.spec)
            if bool(installed_part(self.base, name, local_p.tolist(), local_r.as_quat().tolist())[0]):
                continue
            down = Rotation.from_euler("x" if side == "left" else "y", 180, degrees=True)
            grasp = yield from self.grasp(name, side, np.array([0.0, 0.0, -0.22]), down)
            yield from self.seat(name, side, grasp)
            if index < 4:
                yield from self.move(f"{name}: park", {side: self.homes[side]})
            if self.config.stop_after_leg == index:
                return
        yield from self.finish_table()

    def finish_table(self):
        """Turn the table about opposing leg grasps, install its handle, and set it down."""
        self.compliant_hands = {"left", "right"}
        names = {"left": "leg_4", "right": "leg_2"}
        contacts = {}
        for side, name in names.items():
            part = self.pose(name)
            # These hand-to-tube orientations match the successful leg insertions.
            q_P_G = (
                Rotation.from_euler("x", -90, degrees=True)
                if side == "left"
                else Rotation.from_euler("x", -90, degrees=True) * Rotation.from_euler("y", 180, degrees=True)
            )
            contacts[side] = compose(part, (np.array([0.0, 0.0, -0.22]), q_P_G))
        approach = {side: (p - r.apply([0, 0, 0.035]) + [0, 0, 0.08], r) for side, (p, r) in contacts.items()}
        yield from self.move("turn: approach opposite legs", approach)
        yield from self.move("turn: grasp opposite legs", contacts)
        yield from self.move_grippers("turn: close both grippers", ["left", "right"], 0.9)
        for side, name in names.items():
            self.check_grasp(name, side)
        self.check_installed_legs()
        tray = self.pose("tray")
        grasps = {side: relative(tray, self.tcp(side)) for side in names}
        pivot = (self.tcp("left")[0] + self.tcp("right")[0]) / 2
        pivot_local = tray[1].inv().apply(pivot - tray[0])
        pivot_goal = pivot.copy()
        pivot_goal[1] = 0.0
        pivot_goal[2] = 1.10

        def tray_at(angle, center=pivot_goal):
            rotation = Rotation.from_euler("y", angle, degrees=True) * tray[1]
            return center - rotation.apply(pivot_local), rotation

        target = tray_at(0)
        yield from self.move("turn: lift table", {s: compose(target, g) for s, g in grasps.items()}, tolerance=0.012)
        if self.pose("tray")[0][2] - tray[0][2] < 0.05:
            raise RuntimeError("Both grippers failed to lift the assembled tray")
        self.check_installed_legs()
        # Short increments follow a circular path about the grasp line rather than a chord through the bench.
        for angle in range(10, 91, 10):
            target = tray_at(angle)
            yield from self.move(
                f"turn: {angle} degrees",
                {s: compose(target, g) for s, g in grasps.items()},
                duration=1.0,
                tolerance=0.012,
            )
            self.check_installed_legs()
        # At this side angle, the rim and the lowest foot can both support the table.
        support = tray_at(88.1)
        support[0][0] += 0.30
        support[0][2] = 0.985
        yield from self.move(
            "handle: approach bench support", {s: compose(support, g) for s, g in grasps.items()}, tolerance=0.012
        )
        self.enter("handle: lower until bench supports table")
        supported = False
        for _ in range(round(8.0 / self.base.step_dt)):
            force = sum(
                self.base.scene[f"{name}_bench_contact"].data.normal_force_matrix_w.torch[:, 0, 0]
                for name in ("tray", "leg_1")
            )
            if bool(contact_force_is_upward_support(force, force_threshold=8.0, support_cone_half_angle_rad=0.3)[0]):
                supported = True
                break
            support[0][2] -= 0.008 * self.base.step_dt
            self.targets.update({s: compose(support, g) for s, g in grasps.items()})
            yield
        if not supported:
            raise RuntimeError("The side-resting table has no measured bench support")
        # Stop the descent on contact instead of forcing a geometric goal into the bench.
        self.targets.update({side: self.tcp(side) for side in names})
        yield from self.hold("handle: settle on bench", 0.3)
        self.check_installed_legs()
        yield from self.move_grippers("handle: release right hand", ["right"], 0.0)
        self.check_installed_legs()
        tcp = self.tcp("right")
        center = compose(self.pose("tray"), grasps["right"])[0]
        yield from self.move("handle: center opened jaws", {"right": (center, tcp[1])})
        tcp = self.tcp("right")
        yield from self.move(
            "handle: clear right hand", {"right": (tcp[0] + [0.08, 0, 0] - tcp[1].apply([0, 0, 0.035]), tcp[1])}
        )
        self.compliant_hands.remove("right")
        # Grasp the upper rail in the side-resting assembly to clear the workbench.
        grasp = yield from self.grasp(
            "handle",
            "right",
            np.array([-0.1446, -0.002, 0.216]),
            Rotation.from_euler("z", -90, degrees=True)
            * Rotation.from_euler("x", 180, degrees=True)
            * Rotation.from_euler("y", -15, degrees=True),
        )
        yield from self.seat("handle", "right", grasp)
        yield from self.finish_upright(grasps, tray_at)

    def finish_upright(self, grasps, tray_at):
        """Regrasp the retained assembly, turn it upright, and verify released stability."""
        yield from self.move_empty_arm("turn: clear tray rim", "right", [1.0, 0.85, 0.60, 0.0, 0.0, 0.0])

        # Symmetric jaws can regrasp with a half-turn about their approach axis, avoiding wrist limits.
        contact = compose(self.pose("tray"), grasps["right"])
        contact = contact[0], contact[1] * Rotation.from_euler("z", 180, degrees=True)
        # Approach the round tube obliquely to accommodate the table's measured resting tilt.
        axis = self.pose("leg_2")[1].apply([0, 0, 1])
        contact = contact[0] + 0.10 * axis, Rotation.from_rotvec(np.deg2rad(15.0) * axis) * contact[1]
        tracked_grasp = relative(self.pose("tray"), contact)
        self.compliant_hands.add("right")
        yield from self.move(
            "turn: right regrasp approach",
            {"right": (contact[0] + [0.08, 0, 0] - contact[1].apply([0, 0, 0.035]), contact[1])},
        )
        yield from self.move_with_table("turn: right regrasp", "right", tracked_grasp)
        yield from self.close_on_table("turn: secure right regrasp", "right", tracked_grasp)
        self.check_grasp("leg_2", "right")
        yield from self.reorient_left(grasps, tray_at)

    def reorient_left(self, grasps, tray_at):
        """Withdraw the open left fingers fully before rotating and regrasping."""
        yield from self.move_grippers("turn: release left wrist", ["left"], 0.0)
        contact = self.tcp("left")
        center = compose(self.pose("tray"), grasps["left"])[0]
        yield from self.move("turn: center opened left jaws", {"left": (center, contact[1])})
        contact = self.tcp("left")
        away = contact[0] + [0.08, 0, 0] - contact[1].apply([0, 0, 0.095])
        yield from self.move("turn: withdraw left wrist", {"left": (away, contact[1])})
        yield from self.flip_wrist("left")
        yield from self.regrasp_left_and_place(grasps, tray_at)

    def regrasp_left_and_place(self, grasps, tray_at):
        """Approach the free left tube grasp, then turn and place the assembly."""
        contact = compose(self.pose("tray"), grasps["left"])
        flipped = contact[1] * Rotation.from_euler("z", 180, degrees=True)
        axis = self.pose("leg_4")[1].apply([0, 0, 1])
        flipped = Rotation.from_rotvec(np.deg2rad(-15.0) * axis) * flipped
        tracked_grasp = relative(self.pose("tray"), (contact[0] + 0.10 * axis, flipped))
        axis_T = attachment_pose("leg_4", self.spec)[1].apply([0, 0, -1])
        pregrasp = tracked_grasp[0] + 0.08 * axis_T - tracked_grasp[1].apply([0, 0, 0.06]), tracked_grasp[1]
        yield from self.move_with_table("turn: left regrasp approach", "left", pregrasp)
        yield from self.move_with_table("turn: left regrasp", "left", tracked_grasp)
        yield from self.close_on_table("turn: secure left regrasp", "left", tracked_grasp)
        self.check_grasp("leg_4", "left")
        yield from self.move_grippers("turn: secure carry grasps", ["left", "right"], 1.0)
        yield from self.carry_and_place(tray_at)

    def carry_and_place(self, tray_at, lift=True):
        """Follow the measured grasps to turn and release the complete table."""
        names = ("left", "right")
        measured = self.pose("tray")
        self.carry_grasps = {side: relative(measured, self.tcp(side)) for side in names}
        target = measured[0].copy(), measured[1]
        target[0][2] += 0.15
        if lift:
            yield from self.move_table("turn: lift clear of bench", target, duration=6.0)
        yield from self.move_table("turn: return to center", tray_at(90))
        for angle in range(100, 181, 10):
            yield from self.move_table(f"turn: {angle} degrees", tray_at(angle))
        target = tray_at(180)
        target = target[0].copy(), target[1]
        target[0][2] = 1.172
        yield from self.move_table("place: lower feet onto workbench", target)
        yield from self.move_grippers("place: release both hands", ["left", "right"], 0.0)
        retreat = {
            side: (self.tcp(side)[0] - self.tcp(side)[1].apply([0, 0, 0.10]), self.tcp(side)[1]) for side in names
        }
        yield from self.move("place: withdraw both hands", retreat)
        yield from self.hold("verify: unsupported by robot for five seconds", 5.0)
        if bool(self.base.progress_tracker.is_complete()[0]):
            return
        self.enter("verify: wait for Arena settled-success condition")
        for _ in range(round(10.0 / self.base.step_dt)):
            yield
            if bool(self.base.progress_tracker.is_complete()[0]):
                return
        raise RuntimeError("The released table does not satisfy all five assembly predicates")

    def get_action(self, env, observation):
        self.base = env.unwrapped
        assert self.base.num_envs == 1, "The scripted assembly policy supports one environment"
        assert self.base.action_manager.total_action_dim == 16, "Use the YAM ik_pose action mode"
        if self.program is None:
            self.targets = {side: self.tcp(side) for side in ("left", "right")}
            self.program = self.assemble()
        if not self.finished:
            try:
                next(self.program)
            except StopIteration:
                self.finished = True
            except Exception as error:
                self.failure = str(error)
                raise
        self.steps += 1
        poses = {}
        for side, (p, r) in self.targets.items():
            if side in self.compliant_hands:
                actual_p, actual_r = self.tcp(side)
                gain = self.config.carry_pose_gain
                p = actual_p + gain * (p - actual_p)
                r = Rotation.from_rotvec(gain * (r * actual_r.inv()).as_rotvec()) * actual_r
            poses[side] = np.r_[p, r.as_quat()]
        return absolute_pose_actions(self.base, poses, self.grippers)
