# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Take the item from the back of the shelf.

A deep shelf compartment (with a top, so it cannot be reached from above) holds a front row of items with one gap, and a
short target item at the back, behind a front item next to the gap. The Franka sees the target and the gap while it
lines up in front of the shelf, reaches in through the gap, moves sideways behind the front row to the target, grasps
it and brings it out the same way. Once the hand is past the front row, the front items and the edges of the gap are
beside and behind the wrist camera, while the hand must move behind them without touching them.

Phases: line up in front of the gap (target and front row in view), reach in (hidden phase: hand past the front row),
grasp, pull the target out.
Randomized: shelf position and height, gap position, which side of the gap the target is on and how far.
Difficulty knob: gap width between the front items in meters (smaller is harder; the Franka hand is 0.2 m wide).
Success: the target is held and pulled out in front of the shelf, no front item moved by more than 1.5 cm or tipped,
no collision.
"""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.franka_controller import (
    GRIPPER_CLOSE,
    HOLD_FOREVER_STEPS,
    Waypoint,
    pitched_quat,
)
from isaaclab_arena_environments.spatial_memory.geometry import ALL_FACES
from isaaclab_arena_environments.spatial_memory.layout_sampling import (
    BOARD_THICKNESS_M,
    counter_pose,
    counter_spec,
    pose,
    uniform,
)
from isaaclab_arena_environments.spatial_memory.parts import PartSpec, board, box_item, cylinder_item
from isaaclab_arena_environments.spatial_memory.predicates import part_upright, tool_position_e
from isaaclab_arena_environments.spatial_memory.task_logic import LayoutSample, PlanMode, SpatialMemoryTaskLogic
from isaaclab_arena_environments.spatial_memory.visibility import VisibilityTarget

SHELF_WIDTH_M = 1.00
"""Outer width of the shelf compartment."""
SHELF_DEPTH_M = 0.40
"""Outer depth of the shelf compartment."""
COMPARTMENT_HEIGHT_M = 0.50
"""Clear height between the shelf bottom and the shelf top."""
PLINTH_HEIGHT_M = 0.24
"""Height of the plinth the shelf stands on."""
FRONT_ITEM_SIZE_M = (0.06, 0.08, 0.09)
"""Size of each front-row item."""
FRONT_ROW_DEPTH_M = 0.05
"""Distance of the front row's center behind the shelf front."""
FRONT_ITEM_SPACING_M = 0.095
"""Distance between the centers of adjacent front items on the same side of the gap."""
NUM_FRONT_ITEMS_PER_SIDE = 2
"""Number of front items on each side of the gap."""
TARGET_DIAMETER_M = 0.05
"""Diameter of the cylindrical target."""
TARGET_HEIGHT_M = 0.10
"""Height of the cylindrical target."""
TARGET_DEPTH_M = 0.30
"""Distance of the target's center behind the shelf front."""
TARGET_OFFSET_RANGE_M = (0.12, 0.16)
"""Lateral offset of the target from the gap center: it stands behind a front item, not in line with the gap."""
HAND_PAST_ROW_M = 0.12
"""Distance of the tool center point past the front row's center beyond which the whole hand is past the front row."""
CARRY_LIFT_M = 0.05
"""Height the target is lifted before it is carried out, clear of the shelf floor."""
PREGRASP_SETBACK_M = 0.08
"""Distance in front of the target at which the hand moves sideways behind the front row."""
GRASP_HEIGHT_M = 0.075
"""Height of the grasp above the shelf bottom's top face."""
REST_JOINT_POSE = (0.0, 0.3, 0.0, -2.1, 0.0, 2.9, 0.785)
"""Franka arm posture the scripted controller pulls towards in the inverse-kinematics null space."""
APPROACH_PITCH_RAD = math.radians(55.0)
"""Downward pitch of the approach direction, steep enough to keep the forearm above the front row."""
PULLED_OUT_MARGIN_M = 0.02
"""Distance in front of the shelf front the target's center must be to count as pulled out."""
LIFTED_MARGIN_M = 0.01
"""Height of the target's bottom above the shelf bottom above which it counts as lifted."""
FRONT_ITEM_MAX_SHIFT_M = 0.015
"""Horizontal displacement of a front item above which it counts as disturbed."""
FRONT_ITEM_MAX_TILT_DEG = 10.0
"""Tilt of a front item above which it counts as tipped."""
TARGET_DROPPED_Z_M = -0.05
"""Environment-frame height below which the target counts as dropped."""


class BackOfShelfLogic(SpatialMemoryTaskLogic):
    """Logic of the back-of-shelf task (see the module docstring)."""

    name = "spatial_memory_back_of_shelf"
    rest_joint_pose = REST_JOINT_POSE
    difficulty_name = "gap_width_m"
    default_difficulty = 0.30
    episode_length_s = 60.0
    task_description = "Take the item from the back of the shelf without disturbing the items in front of it."

    def front_item_names(self) -> list[str]:
        """Return the names of the front-row items, left side first."""
        names = []
        for side in ("left", "right"):
            for index in range(NUM_FRONT_ITEMS_PER_SIDE):
                names.append(f"front_{side}_{index}")
        return names

    def build_part_specs(self) -> list[PartSpec]:
        t = BOARD_THICKNESS_M
        height = COMPARTMENT_HEIGHT_M + 2 * t
        parts = [
            counter_spec(),
            board("plinth", (SHELF_DEPTH_M, SHELF_WIDTH_M, PLINTH_HEIGHT_M), color="dark"),
            board("shelf_bottom", (SHELF_DEPTH_M, SHELF_WIDTH_M, t)),
            board("shelf_top", (SHELF_DEPTH_M, SHELF_WIDTH_M, t)),
            board("shelf_back", (t, SHELF_WIDTH_M, height)),
            board("shelf_left", (SHELF_DEPTH_M, t, height)),
            board("shelf_right", (SHELF_DEPTH_M, t, height)),
        ]
        colors = ("yellow", "green", "purple", "orange")
        for index, name in enumerate(self.front_item_names()):
            parts.append(box_item(name, FRONT_ITEM_SIZE_M, color=colors[index], mass_kg=0.3))
        parts.append(cylinder_item("target", TARGET_DIAMETER_M, TARGET_HEIGHT_M, color="red", mass_kg=0.1))
        return parts

    def sample_layout(self, generator: torch.Generator) -> LayoutSample:
        t = BOARD_THICKNESS_M
        poses = self.parked_part_poses()
        front_x = uniform(generator, 0.40, 0.44)
        center_y = uniform(generator, -0.04, 0.04)
        base_z = uniform(generator, 0.22, 0.26)  # top face of the shelf bottom
        gap_y = center_y + uniform(generator, -0.15, 0.15)
        center_x = front_x + 0.5 * SHELF_DEPTH_M
        height = COMPARTMENT_HEIGHT_M + 2 * t
        side_z = base_z - t + 0.5 * height
        poses["counter"] = counter_pose()
        poses["plinth"] = pose(center_x, center_y, base_z - t - 0.5 * PLINTH_HEIGHT_M)
        poses["shelf_bottom"] = pose(center_x, center_y, base_z - 0.5 * t)
        poses["shelf_top"] = pose(center_x, center_y, base_z + COMPARTMENT_HEIGHT_M + 0.5 * t)
        poses["shelf_back"] = pose(front_x + SHELF_DEPTH_M - 0.5 * t, center_y, side_z)
        poses["shelf_left"] = pose(center_x, center_y + 0.5 * SHELF_WIDTH_M - 0.5 * t, side_z)
        poses["shelf_right"] = pose(center_x, center_y - 0.5 * SHELF_WIDTH_M + 0.5 * t, side_z)
        row_x = front_x + FRONT_ROW_DEPTH_M
        item_z = base_z + 0.5 * FRONT_ITEM_SIZE_M[2] + 0.001
        half_gap = 0.5 * self.difficulty
        front_xy = []
        for index in range(NUM_FRONT_ITEMS_PER_SIDE):
            offset = half_gap + 0.5 * FRONT_ITEM_SIZE_M[1] + index * FRONT_ITEM_SPACING_M
            poses[f"front_left_{index}"] = pose(row_x, gap_y + offset, item_z)
            poses[f"front_right_{index}"] = pose(row_x, gap_y - offset, item_z)
        for name in self.front_item_names():
            front_xy.append(poses[name][:2])
        target_x = front_x + TARGET_DEPTH_M + uniform(generator, -0.02, 0.02)
        side = 1.0 if float(torch.rand(1, generator=generator)) < 0.5 else -1.0
        target_y = gap_y + side * uniform(generator, TARGET_OFFSET_RANGE_M[0], TARGET_OFFSET_RANGE_M[1])
        poses["target"] = pose(target_x, target_y, base_z + 0.5 * TARGET_HEIGHT_M + 0.001)
        params = {
            "front_x": torch.tensor(front_x),
            "base_z": torch.tensor(base_z),
            "gap_y": torch.tensor(gap_y),
            "target_xyz": torch.tensor([target_x, target_y, base_z]),
            "front_xy": torch.tensor(front_xy),
        }
        return LayoutSample(part_poses_e=self.part_poses_tensor(poses), layout_params=params)

    # ----------------------------------------------------------------------------------------------------------------

    def visibility_targets(self) -> list[VisibilityTarget]:
        targets = [VisibilityTarget(name, name, ALL_FACES, spacing_m=0.02) for name in self.front_item_names()]
        targets.append(VisibilityTarget("target", "target", ALL_FACES, spacing_m=0.02))
        return targets

    def hidden_target_names(self) -> list[str]:
        return self.front_item_names()

    def remembered_target_names(self) -> list[str]:
        """Return the front items on either side of the gap."""
        return [f"front_{side}_0" for side in ("left", "right")]

    def manipulated_item_names(self) -> list[str]:
        return ["target"]

    def item_support_names(self) -> list[str]:
        return ["shelf_bottom", "shelf_left", "shelf_right", "shelf_back"]

    def forbidden_item_contacts(self) -> list[tuple[str, str]]:
        return [("target", name) for name in self.front_item_names()]

    def hidden_phase(self, env, episode) -> torch.Tensor:
        t_E_tcp_x = tool_position_e(env, self)[:, 0]
        return t_E_tcp_x > episode.layout_params["front_x"] + FRONT_ROW_DEPTH_M + HAND_PAST_ROW_M

    def success_terms(self, env, episode) -> dict[str, torch.Tensor]:
        params = episode.layout_params
        t_E_target = env.arena_world.get_pose_e("target")[:, :3]
        pulled_out = t_E_target[:, 0] < params["front_x"] - PULLED_OUT_MARGIN_M
        lifted = t_E_target[:, 2] - 0.5 * TARGET_HEIGHT_M > params["base_z"] + LIFTED_MARGIN_M
        moved = []
        for index, name in enumerate(self.front_item_names()):
            t_E_item_xy = env.arena_world.get_pose_e(name)[:, :2]
            displacement = (t_E_item_xy - params["front_xy"][:, index]).norm(dim=-1)
            tipped = ~part_upright(env, name, max_tilt_deg=FRONT_ITEM_MAX_TILT_DEG)
            moved.append((displacement > FRONT_ITEM_MAX_SHIFT_M) | tipped)
        return {
            "target_pulled_out": pulled_out & lifted & self.gripper_closed(env),
            "front_items_undisturbed": ~torch.stack(moved, dim=-1).any(dim=-1),
        }

    def extra_failures(self, env, episode) -> torch.Tensor:
        return env.arena_world.get_pose_e("target")[:, 2] < TARGET_DROPPED_Z_M

    # ----------------------------------------------------------------------------------------------------------------

    def plan(self, env, episode, env_id: int, mode: str) -> list[Waypoint]:
        params = {key: value[env_id] for key, value in episode.layout_params.items()}
        front_x = float(params["front_x"])
        base_z = float(params["base_z"])
        target_x, target_y, _ = (float(v) for v in params["target_xyz"])
        q_E_tcp = pitched_quat(0.0, APPROACH_PITCH_RAD)
        grasp_z = base_z + GRASP_HEIGHT_M
        gap_y = float(params["gap_y"])
        pre_x = front_x - 0.10
        # In front of the target, but deep enough that the whole hand is past the front row before it moves
        # sideways; the hand then moves straight onto the target along its approach axis.
        behind_row_x = target_x - PREGRASP_SETBACK_M
        if mode == PlanMode.ORACLE:
            # In through the gap, sideways behind the front row, and back out the same way.
            plan = [
                Waypoint(
                    (pre_x, gap_y, grasp_z + 0.10),
                    q_E_tcp,
                    label="above_preinsert",
                    tolerance_m=0.03,
                    tolerance_rad=0.15,
                ),
                Waypoint((pre_x, gap_y, grasp_z), q_E_tcp, label="preinsert", max_speed_mps=0.2),
                Waypoint(
                    (behind_row_x, gap_y, grasp_z), q_E_tcp, label="reach_in", max_speed_mps=0.05, tolerance_m=0.006
                ),
                Waypoint(
                    (behind_row_x, target_y, grasp_z), q_E_tcp, label="sideways", max_speed_mps=0.03, tolerance_m=0.006
                ),
                Waypoint(
                    (target_x, target_y, grasp_z), q_E_tcp, label="onto_target", max_speed_mps=0.03, tolerance_m=0.006
                ),
                Waypoint((target_x, target_y, grasp_z), q_E_tcp, GRIPPER_CLOSE, hold_steps=12, label="close"),
                Waypoint(
                    (target_x, target_y, grasp_z + CARRY_LIFT_M),
                    q_E_tcp,
                    GRIPPER_CLOSE,
                    label="lift",
                    max_speed_mps=0.05,
                ),
                Waypoint(
                    (behind_row_x, target_y, grasp_z + CARRY_LIFT_M),
                    q_E_tcp,
                    GRIPPER_CLOSE,
                    label="back_off",
                    max_speed_mps=0.05,
                ),
                Waypoint(
                    (behind_row_x, gap_y, grasp_z + CARRY_LIFT_M),
                    q_E_tcp,
                    GRIPPER_CLOSE,
                    label="back_to_gap",
                    max_speed_mps=0.05,
                ),
                Waypoint(
                    (pre_x, gap_y, grasp_z + CARRY_LIFT_M), q_E_tcp, GRIPPER_CLOSE, label="pull_out", max_speed_mps=0.08
                ),
            ]
            exit_y = gap_y
        else:
            # Line up with the visible target and reach straight in, ignoring the front items in the way.
            plan = [
                Waypoint(
                    (pre_x, target_y, grasp_z + 0.10),
                    q_E_tcp,
                    label="above_preinsert",
                    tolerance_m=0.03,
                    tolerance_rad=0.15,
                ),
                Waypoint((pre_x, target_y, grasp_z), q_E_tcp, label="preinsert", max_speed_mps=0.2),
                Waypoint(
                    (target_x, target_y, grasp_z), q_E_tcp, label="straight", max_speed_mps=0.08, tolerance_m=0.006
                ),
                Waypoint((target_x, target_y, grasp_z), q_E_tcp, GRIPPER_CLOSE, hold_steps=12, label="close"),
                Waypoint(
                    (target_x, target_y, grasp_z + CARRY_LIFT_M),
                    q_E_tcp,
                    GRIPPER_CLOSE,
                    label="lift",
                    max_speed_mps=0.05,
                ),
                Waypoint(
                    (pre_x, target_y, grasp_z + CARRY_LIFT_M),
                    q_E_tcp,
                    GRIPPER_CLOSE,
                    label="pull_out",
                    max_speed_mps=0.08,
                ),
            ]
            exit_y = target_y
        plan += [
            Waypoint((pre_x - 0.06, exit_y, grasp_z + 0.06), q_E_tcp, GRIPPER_CLOSE, label="clear", max_speed_mps=0.1),
            Waypoint(
                (pre_x - 0.06, exit_y, grasp_z + 0.06),
                q_E_tcp,
                GRIPPER_CLOSE,
                label="hold",
                max_steps=HOLD_FOREVER_STEPS,
            ),
        ]
        return plan
