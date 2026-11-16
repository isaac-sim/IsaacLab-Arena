# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task 4: load the dishwasher lower rack.

No dishwasher asset exists, so the stand-in is a low rack below counter height, behind the tub's front lip, with the
open door lying in front of it and dividers that form three slots. The Franka picks a cup or a plate (a thin upright
slab) from a side counter and lowers it into the target slot from above. Once the item is below the lip, the wrist
camera looks down into the rack ahead of the hand: the lip and the door are behind and below the hand, out of view,
yet the item, fingers and hand must clear the lip on the way in and on the way out.

Phases: grasp the item (the rack, lip and door are in view), carry it above the slot, lower it (action phase: the
item's base more than 6 cm below the top of the lip), release and lift out.
Randomised: rack extension and height, target slot, item type (cup or plate), item position.
Difficulty knob: slot clearance in metres, the free width left in a slot around the item and the fingers (smaller is
harder).
Success: the item upright and at rest on the rack floor inside the target slot, gripper open and above the rack, no
collision with the rack, dividers or door.
"""

from __future__ import annotations

import torch

from isaaclab_arena_environments.spatial_memory.controller import GRIPPER_CLOSE, Waypoint, top_down_quat
from isaaclab_arena_environments.spatial_memory.geometry import FACE_NX, FACE_NY, FACE_PX, FACE_PY, FACE_PZ
from isaaclab_arena_environments.spatial_memory.layout_utils import choice, pose, uniform
from isaaclab_arena_environments.spatial_memory.parts import PartSpec, board, box_item, cylinder_item
from isaaclab_arena_environments.spatial_memory.predicates import (
    gripper_open,
    item_at_rest,
    item_pose_e,
    item_upright,
    tool_position_e,
)
from isaaclab_arena_environments.spatial_memory.skills import release_and_hold, top_down_pick
from isaaclab_arena_environments.spatial_memory.task_logic import LayoutSample, PlanMode, SpatialMemoryTaskLogic
from isaaclab_arena_environments.spatial_memory.visibility import VisibilityTarget

RACK_LENGTH_M = 0.40
RACK_WIDTH_M = 0.48
RACK_WALL_M = 0.01
RACK_WALL_HEIGHT_M = 0.10
LIP_HEIGHT_M = 0.13
"""Height of the tub's front lip above the rack floor."""
ACTION_PHASE_DEPTH_M = 0.06
"""The action phase starts once the item's base is this far below the top of the lip."""
DIVIDER_THICKNESS_M = 0.006
DIVIDER_HEIGHT_M = 0.05
DIVIDER_LENGTH_M = 0.34
NUM_SLOTS = 3
FINGER_WIDTH_M = 0.02
"""Width each Franka finger adds next to the item (finger thickness plus tracking margin)."""
DOOR_SIZE_M = (0.30, 0.50, 0.02)
CUP_DIAMETER_M = 0.055
CUP_HEIGHT_M = 0.09
PLATE_SIZE_M = (0.12, 0.03, 0.12)
"""Plate stand-in: a thin upright slab (thickness along y, the fingers' closing direction)."""
GRASP_DEPTH_M = 0.03
"""Depth of the fingertips below the item's top when grasping from above."""
PLACE_DEPTH_M = 0.22
"""Item position behind the rack's front wall."""
SIDE_COUNTER_SIZE_M = (0.36, 0.40, 0.04)
REST_JOINT_POSE = (0.0, 0.6, 0.0, -1.9, 0.0, 2.6, 0.785)
NOMINAL_RACK_Y = -0.07
"""Mean lateral rack position, used by the blind controller."""


class DishwasherRackLogic(SpatialMemoryTaskLogic):
    """Logic of task 4 (see module docstring)."""

    name = "spatial_memory_dishwasher_rack"
    difficulty_name = "slot_clearance_m"
    default_difficulty = 0.03
    episode_length_s = 40.0
    task_description = "Pick up the item from the counter and load it into the indicated slot of the lower rack."

    def build_part_specs(self) -> list[PartSpec]:
        wall = RACK_WALL_M
        parts = [
            board("side_counter", SIDE_COUNTER_SIZE_M, colour="white"),
            board("counter", (0.50, 0.80, 0.04), colour="white"),
            board("door", DOOR_SIZE_M, colour="grey"),
            board("rack_floor", (RACK_LENGTH_M, RACK_WIDTH_M, wall), colour="dark"),
            board("rack_front", (wall, RACK_WIDTH_M, LIP_HEIGHT_M), colour="grey"),
            board("rack_back", (wall, RACK_WIDTH_M, RACK_WALL_HEIGHT_M), colour="dark"),
            board("rack_left", (RACK_LENGTH_M, wall, RACK_WALL_HEIGHT_M), colour="dark"),
            board("rack_right", (RACK_LENGTH_M, wall, RACK_WALL_HEIGHT_M), colour="dark"),
        ]
        for index in range(NUM_SLOTS + 1):
            parts.append(board(f"divider_{index}", (DIVIDER_LENGTH_M, DIVIDER_THICKNESS_M, DIVIDER_HEIGHT_M), "grey"))
        parts.append(cylinder_item("cup", CUP_DIAMETER_M, CUP_HEIGHT_M, colour="blue", mass_kg=0.12))
        parts.append(box_item("plate", PLATE_SIZE_M, colour="yellow", mass_kg=0.12))
        return parts

    def item_width(self, item: str) -> float:
        return CUP_DIAMETER_M if item == "cup" else PLATE_SIZE_M[1]

    def item_height(self, item: str) -> float:
        return CUP_HEIGHT_M if item == "cup" else PLATE_SIZE_M[2]

    def sample_layout(self, generator: torch.Generator) -> LayoutSample:
        table = self.new_pose_table()
        rack_front_x = uniform(generator, 0.44, 0.50)
        rack_y = uniform(generator, -0.10, -0.04)
        floor_z = uniform(generator, -0.22, -0.18)  # top face of the rack floor
        target_slot = choice(generator, NUM_SLOTS)
        item_index = choice(generator, 2)
        item = ("cup", "plate")[item_index]
        slot_width = self.item_width(item) + 2.0 * FINGER_WIDTH_M + self.difficulty
        pitch = slot_width + DIVIDER_THICKNESS_M
        wall = RACK_WALL_M
        center_x = rack_front_x + 0.5 * RACK_LENGTH_M
        wall_z = floor_z + 0.5 * RACK_WALL_HEIGHT_M
        table["side_counter"] = pose(0.36, 0.42, -0.5 * SIDE_COUNTER_SIZE_M[2])
        counter_front_x = rack_front_x + 0.36
        table["counter"] = pose(counter_front_x + 0.25, rack_y, -0.02)
        table["door"] = pose(rack_front_x - 0.5 * DOOR_SIZE_M[0], rack_y, floor_z - 0.5 * DOOR_SIZE_M[2])
        table["rack_floor"] = pose(center_x, rack_y, floor_z - 0.5 * wall)
        table["rack_front"] = pose(rack_front_x + 0.5 * wall, rack_y, floor_z - wall + 0.5 * LIP_HEIGHT_M)
        table["rack_back"] = pose(rack_front_x + RACK_LENGTH_M - 0.5 * wall, rack_y, wall_z)
        table["rack_left"] = pose(center_x, rack_y + 0.5 * RACK_WIDTH_M - 0.5 * wall, wall_z)
        table["rack_right"] = pose(center_x, rack_y - 0.5 * RACK_WIDTH_M + 0.5 * wall, wall_z)
        divider_x = rack_front_x + wall + 0.5 * DIVIDER_LENGTH_M + 0.01
        for index in range(NUM_SLOTS + 1):
            offset = (index - 0.5 * NUM_SLOTS) * pitch
            table[f"divider_{index}"] = pose(divider_x, rack_y + offset, floor_z + 0.5 * DIVIDER_HEIGHT_M)
        slot_y = rack_y + (target_slot - 0.5 * (NUM_SLOTS - 1)) * pitch
        item_x = uniform(generator, 0.30, 0.38)
        item_y = uniform(generator, 0.36, 0.46)
        table[item] = pose(item_x, item_y, 0.5 * self.item_height(item) + 0.001)
        params = {
            "rack_front_x": torch.tensor(rack_front_x),
            "rack_y": torch.tensor(rack_y),
            "floor_z": torch.tensor(floor_z),
            "target_slot": torch.tensor(target_slot),
            "slot_y": torch.tensor(slot_y),
            "slot_width": torch.tensor(slot_width),
            "item_index": torch.tensor(item_index),
            "item_start_xy": torch.tensor([item_x, item_y]),
        }
        return LayoutSample(part_poses_e=self.pose_table_to_tensor(table), params=params)

    # ----------------------------------------------------------------------------------------------------------------

    def visibility_targets(self) -> list[VisibilityTarget]:
        targets = [
            VisibilityTarget("door", "door", (FACE_PZ,), spacing_m=0.03),
            VisibilityTarget("rack_front_wall", "rack_front", (FACE_NX, FACE_PX, FACE_PZ), spacing_m=0.02),
        ]
        for index in range(NUM_SLOTS + 1):
            targets.append(
                VisibilityTarget(f"divider_{index}", f"divider_{index}", (FACE_PY, FACE_NY, FACE_PZ), spacing_m=0.02)
            )
        return targets

    def hidden_target_names(self) -> list[str]:
        return ["door", "rack_front_wall"]

    def manipulated_item_names(self) -> list[str]:
        return ["cup", "plate"]

    def item_support_names(self) -> list[str]:
        return ["side_counter", "rack_floor"]

    def active_item(self, runtime, env_id: int) -> str:
        return ("cup", "plate")[int(runtime.params["item_index"][env_id])]

    def _active_item_pose_e(self, env, runtime) -> torch.Tensor:
        cup = item_pose_e(env, "cup")
        plate = item_pose_e(env, "plate")
        is_cup = (runtime.params["item_index"] == 0).unsqueeze(-1)
        return torch.where(is_cup, cup, plate)

    def action_phase(self, env, runtime) -> torch.Tensor:
        params = runtime.params
        item = self._active_item_pose_e(env, runtime)
        half_height = torch.where(params["item_index"] == 0, 0.5 * CUP_HEIGHT_M, 0.5 * PLATE_SIZE_M[2])
        base_z = item[:, 2] - half_height
        in_rack_x = (item[:, 0] > params["rack_front_x"]) & (item[:, 0] < params["rack_front_x"] + RACK_LENGTH_M)
        # The hand is down in the rack once the item's base is well below the lip.
        in_rack = base_z < params["floor_z"] + LIP_HEIGHT_M - ACTION_PHASE_DEPTH_M
        return in_rack_x & in_rack & self.gripper_closed(env)

    def success_terms(self, env, runtime) -> dict[str, torch.Tensor]:
        params = runtime.params
        item = self._active_item_pose_e(env, runtime)
        is_cup = params["item_index"] == 0
        half_height = torch.where(is_cup, 0.5 * CUP_HEIGHT_M, 0.5 * PLATE_SIZE_M[2])
        on_floor = (item[:, 2] - half_height - params["floor_z"]).abs() < 0.012
        in_slot_y = (item[:, 1] - params["slot_y"]).abs() < 0.5 * params["slot_width"]
        in_rack_x = (item[:, 0] > params["rack_front_x"] + 0.03) & (
            item[:, 0] < params["rack_front_x"] + RACK_LENGTH_M - 0.03
        )
        upright = torch.where(is_cup, item_upright(env, "cup"), item_upright(env, "plate"))
        at_rest = torch.where(is_cup, item_at_rest(env, "cup"), item_at_rest(env, "plate"))
        tool_above = tool_position_e(env, self)[:, 2] > params["floor_z"] + LIP_HEIGHT_M + 0.05
        return {
            "item_in_target_slot": in_slot_y & in_rack_x,
            "item_on_rack_floor": on_floor,
            "item_upright": upright,
            "item_at_rest": at_rest,
            "released_and_lifted_out": gripper_open(env, self) & tool_above,
        }

    def extra_failures(self, env, runtime) -> torch.Tensor:
        return self._active_item_pose_e(env, runtime)[:, 2] < runtime.params["floor_z"] - 0.1

    # ----------------------------------------------------------------------------------------------------------------

    def plan(self, env, runtime, env_id: int, mode: str) -> list[Waypoint]:
        params = {key: value[env_id] for key, value in runtime.params.items()}
        item = self.active_item(runtime, env_id)
        item_x, item_y = (float(v) for v in params["item_start_xy"])
        floor_z = float(params["floor_z"])
        rack_front_x = float(params["rack_front_x"])
        slot_y = float(params["slot_y"])
        quat = top_down_quat(0.0)
        grip_offset = self.item_height(item) - GRASP_DEPTH_M
        """Tool height above the item's base while it is held."""
        place_x = rack_front_x + PLACE_DEPTH_M
        place_z = floor_z + grip_offset + 0.004
        clear_z = floor_z + LIP_HEIGHT_M + grip_offset + 0.04
        plan = top_down_pick((item_x, item_y), grip_offset, camera_yaw_rad=0.0)
        if mode == PlanMode.ORACLE:
            # Lift the item clear of the lip, move above the slot, lower straight down, and leave straight up.
            lift_z = max(clear_z, grip_offset + 0.06)
            plan += [
                Waypoint((item_x, item_y, lift_z), quat, GRIPPER_CLOSE, label="lift", max_speed_mps=0.12),
                Waypoint((place_x, slot_y, lift_z), quat, GRIPPER_CLOSE, label="above_slot", max_speed_mps=0.2),
                Waypoint((place_x, slot_y, place_z), quat, GRIPPER_CLOSE, label="lower", max_speed_mps=0.08),
            ]
            release_y = slot_y
        else:
            # Ignore the hidden rack geometry: aim at the slot's nominal position (mean rack placement and default
            # clearance) and move straight there after a nominal lift.
            nominal_pitch = self.item_width(item) + 2.0 * FINGER_WIDTH_M + self.default_difficulty + DIVIDER_THICKNESS_M
            nominal_y = NOMINAL_RACK_Y + (int(params["target_slot"]) - 0.5 * (NUM_SLOTS - 1)) * nominal_pitch
            plan += [
                Waypoint((item_x, item_y, grip_offset + 0.06), quat, GRIPPER_CLOSE, label="lift", max_speed_mps=0.12),
                Waypoint((place_x, nominal_y, place_z), quat, GRIPPER_CLOSE, label="straight", max_speed_mps=0.15),
            ]
            release_y = nominal_y
        plan += release_and_hold((place_x, release_y, place_z), quat, (place_x, release_y, clear_z + 0.06))
        return plan
