# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Put the mug away past the open cabinet door.

A Franka on a counter picks an upright mug and places it on a target shelf of a cabinet whose door stands open beside
the opening, swung towards the robot. While the mug is inside the cabinet the wrist camera looks into the
compartment, so the door and the front edges of the compartment are out of view, yet the hand, wrist and forearm
pass right by them.

Phases: approach and grasp the mug (the cabinet and door are in view), lift, move in front of the opening, insert
(hidden phase: tool more than 5 cm inside the cabinet), lower, release and withdraw.
Randomized: cabinet position, base height, target compartment, door angle (around the knob value), mug position.
Difficulty knob: door opening angle in degrees (smaller is harder; 90 is perpendicular to the cabinet front).
Success: mug upright and at rest on the target shelf inside the cabinet, gripper open and withdrawn, no collision.
"""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.franka_controller import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    HOLD_FOREVER_STEPS,
    Waypoint,
    pitched_approach,
    pitched_quat,
)
from isaaclab_arena_environments.spatial_memory.geometry import FACE_NX, FACE_PX
from isaaclab_arena_environments.spatial_memory.layout_sampling import (
    BOARD_THICKNESS_M,
    choice,
    counter_pose,
    counter_spec,
    pose,
    uniform,
)
from isaaclab_arena_environments.spatial_memory.parts import PartSpec, board, cylinder_item
from isaaclab_arena_environments.spatial_memory.predicates import part_upright
from isaaclab_arena_environments.spatial_memory.task_logic import LayoutSample, PlanMode, SpatialMemoryTaskLogic
from isaaclab_arena_environments.spatial_memory.visibility import VisibilityTarget

CABINET_WIDTH_M = 0.40
"""Outer width of the cabinet along y."""
CABINET_DEPTH_M = 0.30
"""Outer depth of the cabinet along x."""
COMPARTMENT_HEIGHT_M = 0.30
"""Clear height of each of the cabinet's two compartments."""
PLINTH_HEIGHT_M = 0.24
"""Height of the plinth the cabinet stands on."""
MUG_DIAMETER_M = 0.055
"""Diameter of the mug."""
MUG_HEIGHT_M = 0.09
"""Height of the mug."""
GRASP_HEIGHT_M = 0.05
"""Height of the tool center point above the mug's base when grasping."""
APPROACH_PITCH_RAD = math.radians(45.0)
"""Downward pitch of the hand's approach direction, both at the counter and into the cabinet."""
INSERT_DEPTH_M = 0.12
"""Distance of the placed mug's center behind the cabinet front."""
HIDDEN_PHASE_DEPTH_M = 0.05
"""Depth of the tool center point inside the cabinet from which the hidden phase starts."""
TRANSIT_HEIGHT_M = 0.26
"""Lowest tool height for sideways moves over the counter."""
DOOR_ANGLE_JITTER_DEG = 6.0
"""Maximum random deviation of the door angle from the difficulty knob's value."""
REST_JOINT_POSE = (0.15, 0.3, 0.0, -2.1, 0.0, 2.9, 0.785)
"""Forward-pitched Franka arm posture the scripted controller pulls towards in the inverse-kinematics null space."""
INSIDE_FRONT_MARGIN_M = 0.04
"""Minimum distance of the placed mug's center behind the cabinet front."""
INSIDE_BACK_MARGIN_M = 0.02
"""Minimum distance of the placed mug's center in front of the cabinet's back."""
INSIDE_SIDE_MARGIN_M = 0.03
"""Minimum y distance between the placed mug's center and the cabinet's sides."""
ON_SUPPORT_TOLERANCE_M = 0.012
"""Maximum vertical distance between the mug's base and the target shelf for the mug to rest on it."""
MUG_AT_REST_SPEED_MPS = 0.02
"""Linear speed below which the mug counts as at rest."""
WITHDRAWN_MARGIN_M = 0.03
"""Distance in front of the cabinet front the tool center point must be for the arm to count as withdrawn."""
DROPPED_Z_M = -0.05
"""Environment-frame height below which the mug counts as dropped."""


class MugPastDoorLogic(SpatialMemoryTaskLogic):
    """Logic of the mug-past-the-door task (see the module docstring)."""

    name = "spatial_memory_mug_past_door"
    rest_joint_pose = REST_JOINT_POSE
    difficulty_name = "door_angle_deg"
    default_difficulty = 90.0
    episode_length_s = 40.0
    task_description = "Pick up the mug and put it on the indicated shelf inside the open cabinet."

    def door_angle_deg(self) -> float:
        """Return the nominal door opening angle in degrees (the difficulty knob)."""
        return self.difficulty

    def cabinet_height(self) -> float:
        """Return the outer height of the cabinet."""
        return 2.0 * COMPARTMENT_HEIGHT_M + 3.0 * BOARD_THICKNESS_M

    def build_part_specs(self) -> list[PartSpec]:
        t = BOARD_THICKNESS_M
        inner_width = CABINET_WIDTH_M - 2.0 * t
        return [
            counter_spec(),
            board("plinth", (CABINET_DEPTH_M, CABINET_WIDTH_M, PLINTH_HEIGHT_M), color="dark"),
            board("cab_bottom", (CABINET_DEPTH_M, inner_width, t)),
            board("cab_shelf", (CABINET_DEPTH_M - t, inner_width, t)),
            board("cab_top", (CABINET_DEPTH_M, CABINET_WIDTH_M, t)),
            board("cab_left", (CABINET_DEPTH_M, t, self.cabinet_height())),
            board("cab_right", (CABINET_DEPTH_M, t, self.cabinet_height())),
            board("cab_back", (t, inner_width, self.cabinet_height())),
            board("door", (t, CABINET_WIDTH_M, self.cabinet_height()), color="gray"),
            cylinder_item("mug", MUG_DIAMETER_M, MUG_HEIGHT_M, color="red", mass_kg=0.15),
        ]

    def sample_layout(self, generator: torch.Generator) -> LayoutSample:
        t = BOARD_THICKNESS_M
        poses = self.parked_part_poses()
        front_x = uniform(generator, 0.53, 0.57)
        center_y = uniform(generator, -0.08, -0.02)
        base_z = uniform(generator, 0.13, 0.16)  # top face of the bottom board
        target = choice(generator, 2)
        door_angle = math.radians(
            self.door_angle_deg() + uniform(generator, -DOOR_ANGLE_JITTER_DEG, DOOR_ANGLE_JITTER_DEG)
        )
        center_x = front_x + 0.5 * CABINET_DEPTH_M
        height = self.cabinet_height()
        bottom_z = base_z - t  # underside of the bottom board
        poses["counter"] = counter_pose()
        poses["plinth"] = pose(center_x, center_y, bottom_z - 0.5 * PLINTH_HEIGHT_M)
        poses["cab_bottom"] = pose(center_x, center_y, base_z - 0.5 * t)
        shelf_top = base_z + COMPARTMENT_HEIGHT_M + t
        poses["cab_shelf"] = pose(center_x + 0.5 * t, center_y, shelf_top - 0.5 * t)
        top_bottom = shelf_top + COMPARTMENT_HEIGHT_M
        poses["cab_top"] = pose(center_x, center_y, top_bottom + 0.5 * t)
        side_z = bottom_z + 0.5 * height
        poses["cab_left"] = pose(center_x, center_y + 0.5 * CABINET_WIDTH_M - 0.5 * t, side_z)
        poses["cab_right"] = pose(center_x, center_y - 0.5 * CABINET_WIDTH_M + 0.5 * t, side_z)
        poses["cab_back"] = pose(front_x + CABINET_DEPTH_M - 0.5 * t, center_y, side_z)
        # The door hangs on the right front edge and swings towards the robot by door_angle (0 = closed).
        hinge_x = front_x - 0.5 * t
        hinge_y = center_y - 0.5 * CABINET_WIDTH_M
        direction = (-math.sin(door_angle), math.cos(door_angle))
        door_center_x = hinge_x + 0.5 * CABINET_WIDTH_M * direction[0]
        door_center_y = hinge_y + 0.5 * CABINET_WIDTH_M * direction[1]
        door_yaw = math.atan2(direction[1], direction[0]) - 0.5 * math.pi
        poses["door"] = pose(door_center_x, door_center_y, side_z, door_yaw)
        mug_x = uniform(generator, 0.34, 0.41)
        mug_y = uniform(generator, 0.24, 0.34)
        poses["mug"] = pose(mug_x, mug_y, 0.5 * MUG_HEIGHT_M + 0.001)
        support_z = base_z if target == 0 else shelf_top
        params = {
            "front_x": torch.tensor(front_x),
            "center_y": torch.tensor(center_y),
            "support_z": torch.tensor(support_z),
            "target_compartment": torch.tensor(target),
            "door_angle_deg": torch.tensor(math.degrees(door_angle)),
            "mug_start_xy": torch.tensor([mug_x, mug_y]),
        }
        return LayoutSample(part_poses_e=self.part_poses_tensor(poses), layout_params=params)

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

    def remembered_target_names(self) -> list[str]:
        """Return the door."""
        return ["door"]

    def manipulated_item_names(self) -> list[str]:
        return ["mug"]

    def item_support_names(self) -> list[str]:
        return ["counter", "cab_bottom", "cab_shelf"]

    def hidden_phase(self, env, episode) -> torch.Tensor:
        t_E_tcp_x = self.tool_pose_w(env)[:, 0] - env.scene.env_origins[:, 0]
        return t_E_tcp_x > episode.layout_params["front_x"] + HIDDEN_PHASE_DEPTH_M

    def success_terms(self, env, episode) -> dict[str, torch.Tensor]:
        params = episode.layout_params
        t_E_mug = env.arena_world.get_pose_e("mug")[:, :3]
        mug_speed = env.arena_world.get_root_linear_velocity_w("mug").norm(dim=-1)
        inside_x = (t_E_mug[:, 0] > params["front_x"] + INSIDE_FRONT_MARGIN_M) & (
            t_E_mug[:, 0] < params["front_x"] + CABINET_DEPTH_M - INSIDE_BACK_MARGIN_M
        )
        inside_y = (t_E_mug[:, 1] - params["center_y"]).abs() < 0.5 * CABINET_WIDTH_M - INSIDE_SIDE_MARGIN_M
        on_support = (t_E_mug[:, 2] - 0.5 * MUG_HEIGHT_M - params["support_z"]).abs() < ON_SUPPORT_TOLERANCE_M
        t_E_tcp_x = self.tool_pose_w(env)[:, 0] - env.scene.env_origins[:, 0]
        withdrawn = (t_E_tcp_x < params["front_x"] - WITHDRAWN_MARGIN_M) & ~self.gripper_closed(env)
        return {
            "mug_inside_cabinet": inside_x & inside_y,
            "mug_on_target_shelf": on_support,
            "mug_upright": part_upright(env, "mug"),
            "mug_at_rest": mug_speed < MUG_AT_REST_SPEED_MPS,
            "released_and_withdrawn": withdrawn,
        }

    def extra_failures(self, env, episode) -> torch.Tensor:
        t_E_mug_z = env.arena_world.get_pose_e("mug")[:, 2]
        return t_E_mug_z < DROPPED_Z_M

    # ----------------------------------------------------------------------------------------------------------------

    def plan(self, env, episode, env_id: int, mode: str) -> list[Waypoint]:
        params = {key: value[env_id] for key, value in episode.layout_params.items()}
        front_x = float(params["front_x"])
        center_y = float(params["center_y"])
        support_z = float(params["support_z"])
        mug_x, mug_y = (float(v) for v in params["mug_start_xy"])
        q_E_tcp = pitched_quat(0.0, APPROACH_PITCH_RAD)
        approach_E = pitched_approach(0.0, APPROACH_PITCH_RAD)
        grasp_z = GRASP_HEIGHT_M
        place_z = support_z + GRASP_HEIGHT_M + 0.004
        carry_z = place_z + 0.015
        place_x = front_x + INSERT_DEPTH_M
        pregrasp = (mug_x - 0.10 * approach_E[0], mug_y, grasp_z - 0.10 * approach_E[2])
        plan = [
            Waypoint(
                (pregrasp[0], pregrasp[1], pregrasp[2] + 0.08),
                q_E_tcp,
                label="above_pregrasp",
                tolerance_m=0.03,
                tolerance_rad=0.15,
            ),
            Waypoint(pregrasp, q_E_tcp, label="pregrasp", max_speed_mps=0.2),
            Waypoint((mug_x, mug_y, grasp_z), q_E_tcp, label="grasp", max_speed_mps=0.08, tolerance_m=0.006),
            Waypoint((mug_x, mug_y, grasp_z), q_E_tcp, GRIPPER_CLOSE, hold_steps=15, label="close"),
        ]
        if mode == PlanMode.ORACLE:
            # Lift to the compartment height first, line up in a lane clear of the door, then insert straight.
            lane_y = center_y + 0.06
            transit_z = max(carry_z, TRANSIT_HEIGHT_M)
            plan += [
                Waypoint((mug_x, mug_y, transit_z), q_E_tcp, GRIPPER_CLOSE, label="lift", max_speed_mps=0.1),
                Waypoint(
                    (front_x - 0.14, lane_y, transit_z), q_E_tcp, GRIPPER_CLOSE, label="transit", max_speed_mps=0.2
                ),
                Waypoint(
                    (front_x - 0.12, lane_y, carry_z), q_E_tcp, GRIPPER_CLOSE, label="preinsert", max_speed_mps=0.15
                ),
                Waypoint((place_x, lane_y, carry_z), q_E_tcp, GRIPPER_CLOSE, label="insert", max_speed_mps=0.1),
                Waypoint((place_x, lane_y, place_z), q_E_tcp, GRIPPER_CLOSE, label="lower", max_speed_mps=0.05),
            ]
            release_y = lane_y
        else:
            # Lift a nominal amount and move straight to the goal at the compartment center.
            plan += [
                Waypoint((mug_x, mug_y, grasp_z + 0.08), q_E_tcp, GRIPPER_CLOSE, label="lift", max_speed_mps=0.15),
                Waypoint((place_x, center_y, place_z), q_E_tcp, GRIPPER_CLOSE, label="straight", max_speed_mps=0.15),
            ]
            release_y = center_y
        plan += [
            Waypoint((place_x, release_y, place_z), q_E_tcp, GRIPPER_OPEN, hold_steps=8, label="release"),
            Waypoint((front_x - 0.12, release_y, place_z), q_E_tcp, label="withdraw", max_speed_mps=0.12),
            Waypoint((front_x - 0.12, release_y, place_z), q_E_tcp, label="done", max_steps=HOLD_FOREVER_STEPS),
        ]
        return plan
