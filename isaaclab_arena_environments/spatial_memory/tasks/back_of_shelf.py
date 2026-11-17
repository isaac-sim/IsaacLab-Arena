# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task 5: take the item from the back of the shelf.

A deep shelf compartment (with a top, so it cannot be reached from above) holds a front row of items with one gap, and a
short target item at the back, behind a front item next to the gap. The Franka sees the target and the gap while it
lines up in front of the shelf, reaches in through the gap, moves sideways behind the front row to the target, grasps
it and brings it out the same way. Once the hand is past the front row, the front items and the edges of the gap are
beside and behind the wrist camera, while the hand must move behind them without touching them.

Phases: line up in front of the gap (target and front row in view), reach in (action phase: hand past the front row),
grasp, pull the target out.
Randomised: shelf position and height, gap position, which side of the gap the target is on and how far.
Difficulty knob: gap width between the front items in metres (smaller is harder; the Franka hand is 0.2 m wide).
Success: the target is held and pulled out in front of the shelf, no front item moved by more than 1.5 cm or tipped,
no collision.
"""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.controller import GRIPPER_CLOSE, Waypoint, pitched_quat
from isaaclab_arena_environments.spatial_memory.geometry import ALL_FACES
from isaaclab_arena_environments.spatial_memory.layout_utils import BOARD_T_M, counter_pose, counter_spec, pose, uniform
from isaaclab_arena_environments.spatial_memory.parts import PartSpec, board, box_item, cylinder_item
from isaaclab_arena_environments.spatial_memory.predicates import item_pose_e, item_upright, tool_position_e
from isaaclab_arena_environments.spatial_memory.task_logic import LayoutSample, PlanMode, SpatialMemoryTaskLogic
from isaaclab_arena_environments.spatial_memory.visibility import VisibilityTarget

SHELF_WIDTH_M = 1.00
SHELF_DEPTH_M = 0.40
COMPARTMENT_HEIGHT_M = 0.50
PLINTH_HEIGHT_M = 0.24
FRONT_ITEM_SIZE_M = (0.06, 0.08, 0.09)
FRONT_ROW_DEPTH_M = 0.05
"""Distance of the front row's centre behind the shelf front."""
FRONT_ITEM_SPACING_M = 0.095
NUM_FRONT_ITEMS_PER_SIDE = 2
TARGET_DIAMETER_M = 0.05
TARGET_HEIGHT_M = 0.10
TARGET_DEPTH_M = 0.30
"""Distance of the target's centre behind the shelf front."""
TARGET_OFFSET_RANGE_M = (0.12, 0.16)
"""Lateral offset of the target from the gap centre: it stands behind a front item, not in line with the gap."""
CARRY_LIFT_M = 0.05
"""Height the target is lifted before it is carried out, clear of the shelf floor."""
PREGRASP_SETBACK_M = 0.08
"""How far in front of the target the hand moves sideways."""
GRASP_HEIGHT_M = 0.075
REST_JOINT_POSE = (0.0, 0.3, 0.0, -2.1, 0.0, 2.9, 0.785)
APPROACH_PITCH_RAD = math.radians(55.0)
"""A steeper approach keeps the forearm above the front row."""


class BackOfShelfLogic(SpatialMemoryTaskLogic):
    """Logic of task 5 (see module docstring)."""

    name = "spatial_memory_back_of_shelf"
    difficulty_name = "gap_width_m"
    default_difficulty = 0.30
    episode_length_s = 60.0
    task_description = "Take the item from the back of the shelf without disturbing the items in front of it."

    def front_item_names(self) -> list[str]:
        return [f"front_{side}_{index}" for side in ("left", "right") for index in range(NUM_FRONT_ITEMS_PER_SIDE)]

    def build_part_specs(self) -> list[PartSpec]:
        t = BOARD_T_M
        height = COMPARTMENT_HEIGHT_M + 2 * t
        parts = [
            counter_spec(),
            board("plinth", (SHELF_DEPTH_M, SHELF_WIDTH_M, PLINTH_HEIGHT_M), colour="dark"),
            board("shelf_bottom", (SHELF_DEPTH_M, SHELF_WIDTH_M, t)),
            board("shelf_top", (SHELF_DEPTH_M, SHELF_WIDTH_M, t)),
            board("shelf_back", (t, SHELF_WIDTH_M, height)),
            board("shelf_left", (SHELF_DEPTH_M, t, height)),
            board("shelf_right", (SHELF_DEPTH_M, t, height)),
        ]
        colours = ("yellow", "green", "purple", "orange")
        for index, name in enumerate(self.front_item_names()):
            parts.append(box_item(name, FRONT_ITEM_SIZE_M, colour=colours[index], mass_kg=0.3))
        parts.append(cylinder_item("target", TARGET_DIAMETER_M, TARGET_HEIGHT_M, colour="red", mass_kg=0.1))
        return parts

    def sample_layout(self, generator: torch.Generator) -> LayoutSample:
        t = BOARD_T_M
        table = self.new_pose_table()
        front_x = uniform(generator, 0.40, 0.44)
        center_y = uniform(generator, -0.04, 0.04)
        base_z = uniform(generator, 0.22, 0.26)  # top face of the shelf bottom
        gap_y = center_y + uniform(generator, -0.15, 0.15)
        center_x = front_x + 0.5 * SHELF_DEPTH_M
        height = COMPARTMENT_HEIGHT_M + 2 * t
        side_z = base_z - t + 0.5 * height
        table["counter"] = counter_pose()
        table["plinth"] = pose(center_x, center_y, base_z - t - 0.5 * PLINTH_HEIGHT_M)
        table["shelf_bottom"] = pose(center_x, center_y, base_z - 0.5 * t)
        table["shelf_top"] = pose(center_x, center_y, base_z + COMPARTMENT_HEIGHT_M + 0.5 * t)
        table["shelf_back"] = pose(front_x + SHELF_DEPTH_M - 0.5 * t, center_y, side_z)
        table["shelf_left"] = pose(center_x, center_y + 0.5 * SHELF_WIDTH_M - 0.5 * t, side_z)
        table["shelf_right"] = pose(center_x, center_y - 0.5 * SHELF_WIDTH_M + 0.5 * t, side_z)
        row_x = front_x + FRONT_ROW_DEPTH_M
        item_z = base_z + 0.5 * FRONT_ITEM_SIZE_M[2] + 0.001
        half_gap = 0.5 * self.difficulty
        front_xy = []
        for index in range(NUM_FRONT_ITEMS_PER_SIDE):
            offset = half_gap + 0.5 * FRONT_ITEM_SIZE_M[1] + index * FRONT_ITEM_SPACING_M
            table[f"front_left_{index}"] = pose(row_x, gap_y + offset, item_z)
            table[f"front_right_{index}"] = pose(row_x, gap_y - offset, item_z)
        for name in self.front_item_names():
            front_xy.append(table[name][:2])
        target_x = front_x + TARGET_DEPTH_M + uniform(generator, -0.02, 0.02)
        side = 1.0 if float(torch.rand(1, generator=generator)) < 0.5 else -1.0
        target_y = gap_y + side * uniform(generator, TARGET_OFFSET_RANGE_M[0], TARGET_OFFSET_RANGE_M[1])
        table["target"] = pose(target_x, target_y, base_z + 0.5 * TARGET_HEIGHT_M + 0.001)
        params = {
            "front_x": torch.tensor(front_x),
            "base_z": torch.tensor(base_z),
            "gap_y": torch.tensor(gap_y),
            "target_xyz": torch.tensor([target_x, target_y, base_z]),
            "front_xy": torch.tensor(front_xy),
        }
        return LayoutSample(part_poses_e=self.pose_table_to_tensor(table), params=params)

    # ----------------------------------------------------------------------------------------------------------------

    def visibility_targets(self) -> list[VisibilityTarget]:
        targets = [VisibilityTarget(name, name, ALL_FACES, spacing_m=0.02) for name in self.front_item_names()]
        targets.append(VisibilityTarget("target", "target", ALL_FACES, spacing_m=0.02))
        return targets

    def hidden_target_names(self) -> list[str]:
        return self.front_item_names()

    def manipulated_item_names(self) -> list[str]:
        return ["target"]

    def item_support_names(self) -> list[str]:
        return ["shelf_bottom", "shelf_left", "shelf_right", "shelf_back"]

    def allowed_contacts(self, env, runtime) -> torch.Tensor:
        """The robot may touch only the target; front items may rest on the shelf and touch each other."""
        from isaaclab_arena_environments.spatial_memory.collision import filtered_sensor_labels

        labels = filtered_sensor_labels(self)
        allowed = torch.zeros(len(labels), len(runtime.part_names), dtype=torch.bool, device=env.device)
        target_id = runtime.part_index["target"]
        item_ids = runtime.part_ids(self.item_names())
        support_ids = runtime.part_ids(self.item_support_names())
        for sensor_index, label in enumerate(labels):
            if label in self.embodiment.arm_links or label in self.embodiment.gripper_bodies:
                allowed[sensor_index, target_id] = True
            elif label == "target":
                allowed[sensor_index, support_ids] = True
            else:
                allowed[sensor_index, item_ids] = True
                allowed[sensor_index, support_ids] = True
        return allowed.unsqueeze(0).expand(env.num_envs, -1, -1)

    def action_phase(self, env, runtime) -> torch.Tensor:
        tool_x = tool_position_e(env, self)[:, 0]
        # The hand, not just the fingertips, is past the front row.
        return tool_x > runtime.params["front_x"] + FRONT_ROW_DEPTH_M + 0.12

    def success_terms(self, env, runtime) -> dict[str, torch.Tensor]:
        params = runtime.params
        target = item_pose_e(env, "target")
        pulled_out = target[:, 0] < params["front_x"] - 0.02
        lifted = target[:, 2] - 0.5 * TARGET_HEIGHT_M > params["base_z"] + 0.01
        moved = []
        for index, name in enumerate(self.front_item_names()):
            displacement = (item_pose_e(env, name)[:, :2] - params["front_xy"][:, index]).norm(dim=-1)
            moved.append((displacement > 0.015) | ~item_upright(env, name, max_tilt_deg=10.0))
        return {
            "target_pulled_out": pulled_out & lifted & self.gripper_closed(env),
            "front_items_undisturbed": ~torch.stack(moved, dim=-1).any(dim=-1),
        }

    def extra_failures(self, env, runtime) -> torch.Tensor:
        return item_pose_e(env, "target")[:, 2] < -0.05

    # ----------------------------------------------------------------------------------------------------------------

    def plan(self, env, runtime, env_id: int, mode: str) -> list[Waypoint]:
        params = {key: value[env_id] for key, value in runtime.params.items()}
        front_x = float(params["front_x"])
        base_z = float(params["base_z"])
        target_x, target_y, _ = (float(v) for v in params["target_xyz"])
        quat = pitched_quat(0.0, APPROACH_PITCH_RAD)
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
                    (pre_x, gap_y, grasp_z + 0.10), quat, label="above_preinsert", tolerance_m=0.03, tolerance_rad=0.15
                ),
                Waypoint((pre_x, gap_y, grasp_z), quat, label="preinsert", max_speed_mps=0.2),
                Waypoint((behind_row_x, gap_y, grasp_z), quat, label="reach_in", max_speed_mps=0.05, tolerance_m=0.006),
                Waypoint(
                    (behind_row_x, target_y, grasp_z), quat, label="sideways", max_speed_mps=0.03, tolerance_m=0.006
                ),
                Waypoint(
                    (target_x, target_y, grasp_z), quat, label="onto_target", max_speed_mps=0.03, tolerance_m=0.006
                ),
                Waypoint((target_x, target_y, grasp_z), quat, GRIPPER_CLOSE, hold_steps=12, label="close"),
                Waypoint(
                    (target_x, target_y, grasp_z + CARRY_LIFT_M), quat, GRIPPER_CLOSE, label="lift", max_speed_mps=0.05
                ),
                Waypoint(
                    (behind_row_x, target_y, grasp_z + CARRY_LIFT_M),
                    quat,
                    GRIPPER_CLOSE,
                    label="back_off",
                    max_speed_mps=0.05,
                ),
                Waypoint(
                    (behind_row_x, gap_y, grasp_z + CARRY_LIFT_M),
                    quat,
                    GRIPPER_CLOSE,
                    label="back_to_gap",
                    max_speed_mps=0.05,
                ),
                Waypoint(
                    (pre_x, gap_y, grasp_z + CARRY_LIFT_M), quat, GRIPPER_CLOSE, label="pull_out", max_speed_mps=0.08
                ),
            ]
            exit_y = gap_y
        else:
            # Line up with the visible target and reach straight in, ignoring the front items in the way.
            plan = [
                Waypoint(
                    (pre_x, target_y, grasp_z + 0.10),
                    quat,
                    label="above_preinsert",
                    tolerance_m=0.03,
                    tolerance_rad=0.15,
                ),
                Waypoint((pre_x, target_y, grasp_z), quat, label="preinsert", max_speed_mps=0.2),
                Waypoint((target_x, target_y, grasp_z), quat, label="straight", max_speed_mps=0.08, tolerance_m=0.006),
                Waypoint((target_x, target_y, grasp_z), quat, GRIPPER_CLOSE, hold_steps=12, label="close"),
                Waypoint(
                    (target_x, target_y, grasp_z + CARRY_LIFT_M), quat, GRIPPER_CLOSE, label="lift", max_speed_mps=0.05
                ),
                Waypoint(
                    (pre_x, target_y, grasp_z + CARRY_LIFT_M), quat, GRIPPER_CLOSE, label="pull_out", max_speed_mps=0.08
                ),
            ]
            exit_y = target_y
        plan += [
            Waypoint((pre_x - 0.06, exit_y, grasp_z + 0.06), quat, GRIPPER_CLOSE, label="clear", max_speed_mps=0.1),
            Waypoint((pre_x - 0.06, exit_y, grasp_z + 0.06), quat, GRIPPER_CLOSE, label="hold", max_steps=10_000),
        ]
        return plan
