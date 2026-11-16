# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task 1: put the mug away past the open cabinet door.

A Franka on a counter picks an upright mug and places it on a target shelf of a cabinet whose door stands open beside
the opening, swung towards the robot. While the mug is inside the cabinet the wrist camera looks into the
compartment, so the door and the front edges of the compartment are out of view, yet the hand, wrist and forearm
pass right by them.

Phases: approach and grasp the mug (the cabinet and door are in view), lift, move in front of the opening, insert
(action phase: tool inside the cabinet), lower, release and withdraw.
Randomised: cabinet position, base height, target compartment, door angle (around the knob value), mug position.
Difficulty knob: door opening angle in degrees (smaller is harder; 90 is perpendicular to the cabinet front).
Success: mug upright and at rest on the target shelf inside the cabinet, gripper open and withdrawn, no collision.
"""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.controller import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    Waypoint,
    pitched_approach,
    pitched_quat,
)
from isaaclab_arena_environments.spatial_memory.geometry import FACE_NX, FACE_PX
from isaaclab_arena_environments.spatial_memory.layout_utils import (
    BOARD_T_M,
    choice,
    counter_pose,
    counter_spec,
    pose,
    uniform,
)
from isaaclab_arena_environments.spatial_memory.parts import PartSpec, board, cylinder_item
from isaaclab_arena_environments.spatial_memory.task_logic import LayoutSample, PlanMode, SpatialMemoryTaskLogic
from isaaclab_arena_environments.spatial_memory.visibility import VisibilityTarget

CABINET_WIDTH_M = 0.40
CABINET_DEPTH_M = 0.30
COMPARTMENT_HEIGHT_M = 0.30
PLINTH_HEIGHT_M = 0.24
MUG_DIAMETER_M = 0.055
MUG_HEIGHT_M = 0.09
GRASP_HEIGHT_M = 0.05
"""Tool height above the mug's base when grasping."""
APPROACH_PITCH_RAD = math.radians(45.0)
"""The hand approaches 45 degrees below horizontal, both at the counter and into the cabinet."""
INSERT_DEPTH_M = 0.12
TRANSIT_HEIGHT_M = 0.26
"""Lowest tool height for sideways moves over the counter."""
"""Distance of the placed mug's centre behind the cabinet front."""
DOOR_ANGLE_JITTER_DEG = 6.0
REST_JOINT_POSE = (0.15, 0.3, 0.0, -2.1, 0.0, 2.9, 0.785)
"""Forward-pitched arm posture used as the null-space rest pose."""


class MugPastDoorLogic(SpatialMemoryTaskLogic):
    """Logic of task 1 (see module docstring)."""

    name = "spatial_memory_mug_past_door"
    difficulty_name = "door_angle_deg"
    default_difficulty = 90.0
    episode_length_s = 40.0
    task_description = "Pick up the mug and put it on the indicated shelf inside the open cabinet."

    def cabinet_height(self) -> float:
        return 2.0 * COMPARTMENT_HEIGHT_M + 3.0 * BOARD_T_M

    def build_part_specs(self) -> list[PartSpec]:
        t = BOARD_T_M
        inner_width = CABINET_WIDTH_M - 2.0 * t
        return [
            counter_spec(),
            board("plinth", (CABINET_DEPTH_M, CABINET_WIDTH_M, PLINTH_HEIGHT_M), colour="dark"),
            board("cab_bottom", (CABINET_DEPTH_M, inner_width, t)),
            board("cab_shelf", (CABINET_DEPTH_M - t, inner_width, t)),
            board("cab_top", (CABINET_DEPTH_M, CABINET_WIDTH_M, t)),
            board("cab_left", (CABINET_DEPTH_M, t, self.cabinet_height())),
            board("cab_right", (CABINET_DEPTH_M, t, self.cabinet_height())),
            board("cab_back", (t, inner_width, self.cabinet_height())),
            board("door", (t, CABINET_WIDTH_M, self.cabinet_height()), colour="grey"),
            cylinder_item("mug", MUG_DIAMETER_M, MUG_HEIGHT_M, colour="red", mass_kg=0.15),
        ]

    def sample_layout(self, generator: torch.Generator) -> LayoutSample:
        t = BOARD_T_M
        table = self.new_pose_table()
        front_x = uniform(generator, 0.53, 0.57)
        center_y = uniform(generator, -0.08, -0.02)
        base_z = uniform(generator, 0.13, 0.16)  # top face of the bottom board
        target = choice(generator, 2)
        door_angle = math.radians(self.difficulty + uniform(generator, -DOOR_ANGLE_JITTER_DEG, DOOR_ANGLE_JITTER_DEG))
        center_x = front_x + 0.5 * CABINET_DEPTH_M
        height = self.cabinet_height()
        bottom_z = base_z - t  # underside of the bottom board
        table["counter"] = counter_pose()
        table["plinth"] = pose(center_x, center_y, bottom_z - 0.5 * PLINTH_HEIGHT_M)
        table["cab_bottom"] = pose(center_x, center_y, base_z - 0.5 * t)
        shelf_top = base_z + COMPARTMENT_HEIGHT_M + t
        table["cab_shelf"] = pose(center_x + 0.5 * t, center_y, shelf_top - 0.5 * t)
        top_bottom = shelf_top + COMPARTMENT_HEIGHT_M
        table["cab_top"] = pose(center_x, center_y, top_bottom + 0.5 * t)
        side_z = bottom_z + 0.5 * height
        table["cab_left"] = pose(center_x, center_y + 0.5 * CABINET_WIDTH_M - 0.5 * t, side_z)
        table["cab_right"] = pose(center_x, center_y - 0.5 * CABINET_WIDTH_M + 0.5 * t, side_z)
        table["cab_back"] = pose(front_x + CABINET_DEPTH_M - 0.5 * t, center_y, side_z)
        # The door hangs on the right front edge and swings towards the robot by door_angle (0 = closed).
        hinge_x = front_x - 0.5 * t
        hinge_y = center_y - 0.5 * CABINET_WIDTH_M
        direction = (-math.sin(door_angle), math.cos(door_angle))
        door_center_x = hinge_x + 0.5 * CABINET_WIDTH_M * direction[0]
        door_center_y = hinge_y + 0.5 * CABINET_WIDTH_M * direction[1]
        door_yaw = math.atan2(direction[1], direction[0]) - 0.5 * math.pi
        table["door"] = pose(door_center_x, door_center_y, side_z, door_yaw)
        mug_x = uniform(generator, 0.34, 0.41)
        mug_y = uniform(generator, 0.24, 0.34)
        table["mug"] = pose(mug_x, mug_y, 0.5 * MUG_HEIGHT_M + 0.001)
        support_z = base_z if target == 0 else shelf_top
        params = {
            "front_x": torch.tensor(front_x),
            "center_y": torch.tensor(center_y),
            "support_z": torch.tensor(support_z),
            "target_compartment": torch.tensor(target),
            "door_angle_deg": torch.tensor(math.degrees(door_angle)),
            "mug_start_xy": torch.tensor([mug_x, mug_y]),
        }
        return LayoutSample(part_poses_e=self.pose_table_to_tensor(table), params=params)

    # ----------------------------------------------------------------------------------------------------------------

    def visibility_targets(self) -> list[VisibilityTarget]:
        return [
            VisibilityTarget("door", "door", (FACE_PX, FACE_NX)),
            VisibilityTarget("bottom_front_edge", "cab_bottom", (FACE_NX,), spacing_m=0.02),
            VisibilityTarget("shelf_front_edge", "cab_shelf", (FACE_NX,), spacing_m=0.02),
            VisibilityTarget("top_front_edge", "cab_top", (FACE_NX,), spacing_m=0.02),
        ]

    def hidden_target_names(self) -> list[str]:
        return ["door", "bottom_front_edge", "shelf_front_edge", "top_front_edge"]

    def manipulated_item_names(self) -> list[str]:
        return ["mug"]

    def item_support_names(self) -> list[str]:
        return ["counter", "cab_bottom", "cab_shelf"]

    def action_phase(self, env, runtime) -> torch.Tensor:
        tool_x_e = self.tool_pose_w(env)[:, 0] - env.scene.env_origins[:, 0]
        return tool_x_e > runtime.params["front_x"]

    def success(self, env, runtime) -> torch.Tensor:
        params = runtime.params
        mug_pose_e = env.arena_world.get_pose_e("mug")
        mug_velocity = env.arena_world.get_root_linear_velocity_w("mug").norm(dim=-1)
        inside_x = (mug_pose_e[:, 0] > params["front_x"] + 0.04) & (
            mug_pose_e[:, 0] < params["front_x"] + CABINET_DEPTH_M - 0.02
        )
        inside_y = (mug_pose_e[:, 1] - params["center_y"]).abs() < 0.5 * CABINET_WIDTH_M - 0.03
        on_support = (mug_pose_e[:, 2] - 0.5 * MUG_HEIGHT_M - params["support_z"]).abs() < 0.012
        mug_up_w = _z_axis(mug_pose_e[:, 3:7])
        upright = mug_up_w[:, 2] > math.cos(math.radians(15.0))
        tool_x_e = self.tool_pose_w(env)[:, 0] - env.scene.env_origins[:, 0]
        withdrawn = (tool_x_e < params["front_x"] - 0.03) & ~self.gripper_closed(env)
        return inside_x & inside_y & on_support & upright & (mug_velocity < 0.02) & withdrawn

    def extra_failures(self, env, runtime) -> torch.Tensor:
        mug_z_e = env.arena_world.get_pose_e("mug")[:, 2]
        return mug_z_e < -0.05

    # ----------------------------------------------------------------------------------------------------------------

    def plan(self, env, runtime, env_id: int, mode: str) -> list[Waypoint]:
        params = {key: value[env_id] for key, value in runtime.params.items()}
        front_x = float(params["front_x"])
        center_y = float(params["center_y"])
        support_z = float(params["support_z"])
        mug_x, mug_y = (float(v) for v in params["mug_start_xy"])
        forward = pitched_quat(0.0, APPROACH_PITCH_RAD)
        approach = pitched_approach(0.0, APPROACH_PITCH_RAD)
        grasp_z = GRASP_HEIGHT_M
        place_z = support_z + GRASP_HEIGHT_M + 0.004
        carry_z = place_z + 0.015
        place_x = front_x + INSERT_DEPTH_M
        pregrasp = (mug_x - 0.10 * approach[0], mug_y, grasp_z - 0.10 * approach[2])
        plan = [
            Waypoint(
                (pregrasp[0], pregrasp[1], pregrasp[2] + 0.08),
                forward,
                label="above_pregrasp",
                tolerance_m=0.03,
                tolerance_rad=0.15,
            ),
            Waypoint(pregrasp, forward, label="pregrasp", max_speed_mps=0.2),
            Waypoint((mug_x, mug_y, grasp_z), forward, label="grasp", max_speed_mps=0.08, tolerance_m=0.006),
            Waypoint((mug_x, mug_y, grasp_z), forward, GRIPPER_CLOSE, hold_steps=15, label="close"),
        ]
        if mode == PlanMode.ORACLE:
            # Lift to the compartment height first, line up in a lane clear of the door, then insert straight.
            lane_y = center_y + 0.06
            transit_z = max(carry_z, TRANSIT_HEIGHT_M)
            plan += [
                Waypoint((mug_x, mug_y, transit_z), forward, GRIPPER_CLOSE, label="lift", max_speed_mps=0.1),
                Waypoint(
                    (front_x - 0.14, lane_y, transit_z), forward, GRIPPER_CLOSE, label="transit", max_speed_mps=0.2
                ),
                Waypoint(
                    (front_x - 0.12, lane_y, carry_z), forward, GRIPPER_CLOSE, label="preinsert", max_speed_mps=0.15
                ),
                Waypoint((place_x, lane_y, carry_z), forward, GRIPPER_CLOSE, label="insert", max_speed_mps=0.1),
                Waypoint((place_x, lane_y, place_z), forward, GRIPPER_CLOSE, label="lower", max_speed_mps=0.05),
            ]
            release_y = lane_y
        else:
            # Lift a nominal amount and move straight to the goal at the compartment centre.
            plan += [
                Waypoint((mug_x, mug_y, grasp_z + 0.08), forward, GRIPPER_CLOSE, label="lift", max_speed_mps=0.15),
                Waypoint((place_x, center_y, place_z), forward, GRIPPER_CLOSE, label="straight", max_speed_mps=0.15),
            ]
            release_y = center_y
        plan += [
            Waypoint((place_x, release_y, place_z), forward, GRIPPER_OPEN, hold_steps=8, label="release"),
            Waypoint((front_x - 0.12, release_y, place_z), forward, label="withdraw", max_speed_mps=0.12),
            Waypoint((front_x - 0.12, release_y, place_z), forward, label="done", max_steps=10_000),
        ]
        return plan


def _z_axis(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Return the body z axis (N, 3) of each orientation."""
    import isaaclab.utils.math as math_utils

    z = torch.zeros(quat_xyzw.shape[0], 3, device=quat_xyzw.device)
    z[:, 2] = 1.0
    return math_utils.quat_apply(quat_xyzw, z)
