# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Coordinate catalog lookup, mobile transport, and upright shelf insertion."""

import math

from isaaclab_arena_examples.bento.meal import EE_QUAT_XYZW, TCP_OFFSET
from isaaclab_arena_examples.library.scene import BOOKS, CATALOG

HOVER = 1.15
GRASP_OFFSET = 0.095
SHELF_FRONT = 1.05
RETURN_STOP = -1.50
SHELF_STOP = 0.90
LOAN_STOP = -0.65

PHASES = (
    "navigate_pick",
    "search_catalog",
    "approach_pick",
    "lower_pick",
    "insert_pick",
    "close",
    "raise_pick",
    "extract",
    "lift",
    "stow",
    "navigate_drop",
    "approach_drop",
    "lower_drop",
    "insert_drop",
    "seat_drop",
    "release",
    "withdraw",
    "retreat",
)


class LibraryController:
    """Execute ID-directed returns and loans with a prescribed mobile-base route."""

    def __init__(self, sim, robot, books, joints, jaw_ops, timeout, mobile_base):
        import torch

        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        self.sim, self.robot, self.books, self.joints = sim, robot, books, joints
        self.mobile_base, self.jaw_ops, self.timeout = mobile_base, jaw_ops, timeout
        self.body_id = robot.find_bodies("link_6_t")[0][0]
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=1,
            device=sim.device,
        )
        self.offset = torch.tensor([TCP_OFFSET], device=sim.device, dtype=torch.float32)
        self.orientation = torch.tensor([EE_QUAT_XYZW], device=sim.device, dtype=torch.float32)
        self.jaw_opening = self.jaw_target = BOOKS[0].size[1] + 0.020
        self.released = set()
        self.events = []
        self.book_index = 0
        self.phase = "navigate_pick"
        self.elapsed = self.dwell = 0.0
        self.finished, self.failure = False, None
        self.command_pos = self.tcp_pose()[0].clone()
        self.target = self.command_pos.clone()
        self.travel_joint_target = robot.data.joint_pos.torch.clone()
        self.grasp_height = GRASP_OFFSET
        self.lookup_complete = False

    def tcp_pose(self):
        """Return measured world TCP position and XYZW orientation."""
        from isaaclab.utils.math import quat_apply

        pose = self.robot.data.body_link_pose_w.torch[:, self.body_id]
        return pose[:, :3] + quat_apply(pose[:, 3:7], self.offset), pose[:, 3:7]

    @property
    def book(self):
        """Return the currently scheduled book."""
        return BOOKS[min(self.book_index, len(BOOKS) - 1)]

    def make_target(self):
        """Build approach and insertion waypoints from catalog and measured book poses."""
        import torch

        book = self.book
        source = self.books[book.name].data.root_pose_w.torch[0, :3].tolist()
        shelf_pick = book.name == "retrieve"
        shelf_drop = not shelf_pick
        root = self.robot.data.root_pose_w.torch[0]
        if self.phase in ("navigate_pick", "navigate_drop", "search_catalog"):
            return self.tcp_pose()[0].clone()
        if self.phase in ("approach_pick", "lower_pick", "insert_pick", "close", "raise_pick", "extract", "lift"):
            point = [source[0], SHELF_FRONT if shelf_pick else source[1], HOVER]
            if self.phase in ("lower_pick", "insert_pick", "close", "raise_pick", "extract"):
                point[2] = source[2] + GRASP_OFFSET
            if self.phase in ("insert_pick", "close", "raise_pick"):
                point[1] = source[1]
            if self.phase == "raise_pick":
                point[2] += 0.04
            if self.phase == "extract" and not shelf_pick:
                point[2] = HOVER
        elif self.phase == "stow":
            point = [float(root[0]) + 0.40, float(root[1]) + 0.03, HOVER]
        else:
            target = CATALOG[book.catalog_id]["shelf_position"] if shelf_drop else book.target
            point = [target[0], SHELF_FRONT if shelf_drop else target[1], HOVER]
            if self.phase in ("lower_drop", "insert_drop", "seat_drop", "release", "withdraw"):
                point[2] = target[2] + self.grasp_height + 0.003
            if shelf_drop and self.phase in ("lower_drop", "insert_drop"):
                point[2] += 0.04
            if self.phase in ("insert_drop", "seat_drop", "release"):
                point[1] = target[1]
            if self.phase == "withdraw" and not shelf_drop:
                point[2] = HOVER
        return torch.tensor([point], dtype=torch.float32, device=self.sim.device)

    def attach(self):
        """Hold the book at its measured relative pose after reaching its upper grasp region."""
        import torch

        from isaaclab.utils.math import subtract_frame_transforms
        from pxr import Gf

        ee = self.robot.data.body_link_pose_w.torch[:, self.body_id]
        pose = self.books[self.book.name].data.root_pose_w.torch
        grasp = pose[:, :3].clone()
        grasp[:, 2] += GRASP_OFFSET
        if torch.linalg.vector_norm(self.tcp_pose()[0] - grasp).item() > 0.014:
            self.failure = f"book_moved_during_pick: {self.book.name}"
            self.finished = True
            return False
        position, rotation = subtract_frame_transforms(ee[:, :3], ee[:, 3:7], pose[:, :3], pose[:, 3:7])
        position, rotation = position[0].tolist(), rotation[0].tolist()
        joint = self.joints[self.book.name]
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*position))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(rotation[3], Gf.Vec3f(*rotation[:3])))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1))
        self.grasp_height = float(self.tcp_pose()[0][0, 2] - pose[0, 2])
        joint.GetJointEnabledAttr().Set(True)
        self.events.append({"event": "picked", "catalog_id": self.book.catalog_id, "phase": self.phase})
        return True

    def advance(self):
        """Record semantic task events and advance to the next measured waypoint."""
        if self.phase == "close" and not self.attach():
            return
        if self.phase == "release":
            self.joints[self.book.name].GetJointEnabledAttr().Set(False)
            self.released.add(self.book.name)
            self.events.append({
                "event": "loan_box_placed" if self.book.name == "retrieve" else "shelf_returned",
                "catalog_id": self.book.catalog_id,
                "shelf_slot": self.book.shelf_slot,
            })
        if self.phase == "retreat":
            self.book_index += 1
            self.finished = self.book_index == len(BOOKS)
            self.phase = "done" if self.finished else "navigate_pick"
        else:
            self.phase = PHASES[PHASES.index(self.phase) + 1]
        self.jaw_target = self.book.size[1] + (
            0.010
            if self.phase
            in (
                "close",
                "raise_pick",
                "extract",
                "lift",
                "stow",
                "navigate_drop",
                "approach_drop",
                "lower_drop",
                "insert_drop",
                "seat_drop",
                "release",
            )
            else 0.020
        )
        if self.phase in ("withdraw", "retreat", "done"):
            self.jaw_target = self.book.size[1] + 0.020
        self.elapsed = self.dwell = 0.0
        if self.phase.startswith("navigate"):
            self.travel_joint_target = self.robot.data.joint_pos.torch.clone()
        self.command_pos = self.tcp_pose()[0].clone()
        self.target = self.make_target()
        print(f"[library] {self.book.catalog_id}/{self.book.shelf_slot}: {self.phase}", flush=True)

    def navigate(self):
        """Translate the base and arm root continuously along the open aisle."""
        import torch

        picking = self.phase == "navigate_pick"
        goal_y = (
            (SHELF_STOP if self.book.name == "retrieve" else RETURN_STOP)
            if picking
            else (LOAN_STOP if self.book.name == "retrieve" else SHELF_STOP)
        )
        pose = self.robot.data.root_pose_w.torch.clone()
        error = goal_y - float(pose[0, 1])
        increment = max(-0.22 * self.sim.get_physics_dt(), min(0.22 * self.sim.get_physics_dt(), error))
        pose[0, 1] += increment
        self.robot.write_root_pose_to_sim_index(root_pose=pose)
        base_pose = self.mobile_base.data.root_pose_w.torch.clone()
        base_pose[0, :2] = pose[0, :2]
        self.mobile_base.write_root_pose_to_sim_index(root_pose=base_pose)
        self.robot.set_joint_position_target_index(target=self.travel_joint_target)
        assert torch.isfinite(pose).all(), "Nonfinite mobile root pose."
        self.elapsed += self.sim.get_physics_dt()
        if abs(error) < 0.001:
            self.dwell += self.sim.get_physics_dt()
            if self.dwell > 0.5:
                self.advance()
        elif self.elapsed > 40:
            self.failure, self.finished = "navigation_timeout", True

    def step(self):
        """Control navigation or bounded differential IK and verify waypoint convergence."""
        import torch

        from isaaclab.utils.math import matrix_from_quat, quat_apply, skew_symmetric_matrix, subtract_frame_transforms
        from pxr import Gf

        dt = self.sim.get_physics_dt()
        self.jaw_opening += max(-0.07 * dt, min(0.07 * dt, self.jaw_target - self.jaw_opening))
        for side, op in zip((-1, 1), self.jaw_ops):
            op.Set(Gf.Vec3d(side * self.jaw_opening / 2, 0, TCP_OFFSET[2]))
        if self.phase.startswith("navigate") and not self.finished:
            self.navigate()
            return
        if self.phase == "search_catalog":
            self.elapsed += dt
            if self.elapsed > (2.5 if self.book.name == "retrieve" else 0.5):
                entry = CATALOG[self.book.catalog_id]
                assert entry["name"] == self.book.name, "Catalog ID resolves to a different book."
                self.lookup_complete = self.lookup_complete or self.book.name == "retrieve"
                self.events.append(
                    {"event": "catalog_match", "catalog_id": self.book.catalog_id, "shelf_slot": entry["shelf_slot"]}
                )
                self.advance()
            return
        tcp, orientation = self.tcp_pose()
        if not self.finished:
            self.elapsed += dt
            distance = torch.linalg.vector_norm(self.target - tcp).item()
            angle = 2 * math.acos(min(1.0, abs(float((orientation * self.orientation).sum()))))
            speed = torch.linalg.vector_norm(self.robot.data.body_link_vel_w.torch[0, self.body_id, :3]).item()
            tolerance = 0.003 if self.phase in ("insert_pick", "insert_drop", "lower_drop", "seat_drop") else 0.006
            settled = (
                distance < tolerance
                and angle < 0.05
                and speed < 0.035
                and abs(self.jaw_target - self.jaw_opening) < 0.0005
            )
            self.dwell = self.dwell + dt if settled else 0.0
            if self.dwell >= (0.5 if self.phase in ("close", "release", "withdraw") else 0.2):
                self.advance()
                if self.phase.startswith("navigate") or self.phase == "search_catalog":
                    return
            elif self.elapsed > self.timeout:
                self.failure = f"waypoint_timeout: {self.book.name}/{self.phase}"
                self.finished = True
            delta = self.target - self.command_pos
            tracking_error = torch.linalg.vector_norm(self.command_pos - tcp).item()
            careful = self.phase in (
                "lower_pick",
                "insert_pick",
                "lower_drop",
                "insert_drop",
                "seat_drop",
                "raise_pick",
                "extract",
                "withdraw",
            )
            if not careful or tracking_error < 0.006:
                speed_limit = 0.055 if careful else 0.16
                self.command_pos += delta * min(
                    1.0, speed_limit * dt / max(torch.linalg.vector_norm(delta).item(), 1e-8)
                )
        root = self.robot.data.root_pose_w.torch
        position_b, rotation_b = subtract_frame_transforms(root[:, :3], root[:, 3:7], tcp, orientation)
        goal_b, goal_quat_b = subtract_frame_transforms(root[:, :3], root[:, 3:7], self.command_pos, self.orientation)
        self.ik.set_command(torch.cat((goal_b, goal_quat_b), dim=-1))
        jacobian = self.robot.data.body_link_jacobian_w.torch[:, self.body_id - 1].clone()
        offset = quat_apply(orientation, self.offset)
        jacobian[:, :3] -= torch.bmm(skew_symmetric_matrix(offset), jacobian[:, 3:])
        inverse_rotation = matrix_from_quat(root[:, 3:7]).transpose(1, 2)
        jacobian[:, :3] = torch.bmm(inverse_rotation, jacobian[:, :3])
        jacobian[:, 3:] = torch.bmm(inverse_rotation, jacobian[:, 3:])
        position = self.robot.data.joint_pos.torch
        goal = self.ik.compute(position_b, rotation_b, jacobian, position)
        goal = position + (goal - position).clamp(-0.012, 0.012)
        limits = self.robot.data.soft_joint_pos_limits.torch
        goal = torch.maximum(torch.minimum(goal, limits[..., 1]), limits[..., 0])
        assert torch.isfinite(goal).all(), "Nonfinite IK output."
        self.robot.set_joint_position_target_index(target=goal)
