# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Put the item on the high shelf.

A shelf at about the height of the Franka's shoulder has a front rail and five slots, N of them holding low items that
are shorter than the rail. The robot looks down at the shelf once from above, then picks up a tall item from a stand
next to it with the hand level (the wrist camera under the wrist). The instruction to place it is revealed once the
shelf has been out of view for 3.5 s, longer than the 3 s frame history. The robot must then lift the item over the rail into a free
slot. From below the rail, nothing on the shelf is visible: the rail and the shelf board block the view, so the free
slot must be remembered.

Hidden: the shelf surface and the existing items from when the instruction is revealed until the release.
Randomized: shelf position and height, which slots are occupied, the stand's position and height.
Difficulty knob: number of existing items N (1 to 4 of 5 slots; more is harder).
Success: the item stands upright at rest in a slot that was free, the gripper is open and out of the shelf, no
existing item moved by more than 2 cm, no collision, and the shelf was not re-observed for more than 1 s between the
instruction being revealed and the release (else "searched again", a failure).
"""

from __future__ import annotations

import math
import torch

from isaaclab_arena.tasks.predicates.spatial import velocity_below_threshold
from isaaclab_arena_environments.spatial_memory.franka_controller import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    HOLD_FOREVER_STEPS,
    Waypoint,
    pitched_quat,
    tool_quat_pointing,
)
from isaaclab_arena_environments.spatial_memory.geometry import ALL_FACES, FACE_PZ
from isaaclab_arena_environments.spatial_memory.layout_sampling import (
    BOARD_THICKNESS_M,
    counter_pose,
    counter_spec,
    permutation,
    pose,
    uniform,
)
from isaaclab_arena_environments.spatial_memory.parts import ITEM_COLORS, PartSpec, board, box_item
from isaaclab_arena_environments.spatial_memory.predicates import AT_REST_SPEED_MPS, part_upright, tool_position_e
from isaaclab_arena_environments.spatial_memory.task_logic import LayoutSample, PlanMode, SpatialMemoryTaskLogic
from isaaclab_arena_environments.spatial_memory.visibility import VisibilityTarget

NUM_SLOTS = 5
"""Number of slots in the shelf's row."""
SLOT_SPACING_M = 0.10
"""Distance between the centers of adjacent slots."""
SHELF_SIZE_M = (0.25, 0.58, BOARD_THICKNESS_M)
"""Size of the shelf board."""
RAIL_THICKNESS_M = 0.015
"""Thickness of the rails."""
RAIL_HEIGHT_M = 0.06
"""Height of the rails that close the shelf's front and ends above its surface."""
RAIL_OVERHANG_M = 0.005
"""Overhang of the rails past the shelf board's edges and each other, so no line of sight passes between them."""
ROW_DEPTH_M = 0.065
"""Distance of the slot row's center behind the shelf front."""
EXISTING_ITEM_SIZE_M = (0.05, 0.07, 0.04)
"""Size of each existing item: lower than the rail, so it cannot be seen from below the rail."""
ITEM_SIZE_M = (0.05, 0.06, 0.15)
"""Size of the item to place (depth, grasped width, height)."""
GRASP_HEIGHT_M = 0.10
"""Height of the tool center point above the item's base while it is held."""
RAIL_CLEARANCE_M = 0.012
"""Clearance of the item's base above the rail while it is carried over it."""
STAND_SIZE_M = (0.10, 0.10)
"""Footprint of the stand the item starts on."""
APPROACH_SETBACK_M = 0.10
"""Distance along the reach direction at which the hand stays back from the item before moving onto it."""
LOOK_SETBACK_M = 0.12
"""Distance of the tool center point in front of the shelf front in the look pose."""
LOOK_HEIGHT_M = 0.12
"""Height of the tool center point above the shelf surface in the look pose (hand pitched down, camera above it)."""
REST_JOINT_POSE = (0.0, -0.3, 0.0, -2.2, 0.0, 3.47, -2.356)
"""Franka arm posture the scripted controller pulls towards: hand level and pointing forward, camera under the wrist."""
ROW_TOLERANCE_M = 0.03
"""Maximum x distance between the item's center and the slot row for the item to stand in the row."""
SLOT_TOLERANCE_M = 0.03
"""Maximum y distance between the item's center and a slot's center for the item to stand in that slot."""
ON_SURFACE_TOLERANCE_M = 0.012
"""Maximum vertical distance between the item's base and the shelf surface for the item to stand on it."""
EXISTING_ITEM_MAX_SHIFT_M = 0.02
"""Horizontal displacement of an existing item above which it counts as disturbed."""
HAND_OUT_MARGIN_M = 0.02
"""Distance in front of the shelf front the tool center point must be for the hand to count as out of the shelf."""
DROPPED_Z_M = 0.3
"""Environment-frame height below which the item counts as dropped."""


def level_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    """Return a level tool orientation with heading ``yaw_rad`` and the wrist camera (hand x) below the wrist."""
    return tool_quat_pointing((math.cos(yaw_rad), math.sin(yaw_rad), 0.0), (0.0, 0.0, -1.0))


class HighShelfLogic(SpatialMemoryTaskLogic):
    """Logic of the high-shelf task (see the module docstring)."""

    name = "spatial_memory_high_shelf"
    rest_joint_pose = REST_JOINT_POSE
    difficulty_name = "num_existing_items"
    default_difficulty = 4
    episode_length_s = 45.0
    task_description = "Look at the high shelf, pick up the item, and put it on the shelf in a free spot."
    forbids_searching_again = True
    reobservation_budget_s = 1.0

    @property
    def num_existing(self) -> int:
        """Return the number of existing items on the shelf."""
        num = int(round(self.difficulty))
        assert 1 <= num <= NUM_SLOTS - 1, f"num_existing_items must be between 1 and {NUM_SLOTS - 1}, got {num}"
        return num

    def existing_names(self) -> list[str]:
        """Return the names of the existing items."""
        return [f"existing_{index}" for index in range(self.num_existing)]

    def build_part_specs(self) -> list[PartSpec]:
        parts = [
            counter_spec(),
            board("shelf", SHELF_SIZE_M, color="wood"),
            board("rail", (RAIL_THICKNESS_M, SHELF_SIZE_M[1] + 2 * RAIL_OVERHANG_M, RAIL_HEIGHT_M), color="dark"),
            board("rail_left", (SHELF_SIZE_M[0] + RAIL_OVERHANG_M, RAIL_THICKNESS_M, RAIL_HEIGHT_M), color="dark"),
            board("rail_right", (SHELF_SIZE_M[0] + RAIL_OVERHANG_M, RAIL_THICKNESS_M, RAIL_HEIGHT_M), color="dark"),
            board("shelf_back", (BOARD_THICKNESS_M, SHELF_SIZE_M[1], 0.60), color="gray"),
            board("stand", (STAND_SIZE_M[0], STAND_SIZE_M[1], 0.60), color="gray"),
        ]
        for index, name in enumerate(self.existing_names()):
            parts.append(box_item(name, EXISTING_ITEM_SIZE_M, color=ITEM_COLORS[1 + index], mass_kg=0.1))
        parts.append(box_item("item", ITEM_SIZE_M, color="red", mass_kg=0.08))
        return parts

    def sample_layout(self, generator: torch.Generator) -> LayoutSample:
        poses = self.parked_part_poses()
        front_x = uniform(generator, 0.53, 0.57)
        center_y = uniform(generator, -0.03, 0.03)
        surface_z = uniform(generator, 0.57, 0.61)
        poses["counter"] = counter_pose()
        poses["shelf"] = pose(front_x + 0.5 * SHELF_SIZE_M[0], center_y, surface_z - 0.5 * SHELF_SIZE_M[2])
        rail_z = surface_z + 0.5 * RAIL_HEIGHT_M
        poses["rail"] = pose(front_x + 0.5 * RAIL_THICKNESS_M - RAIL_OVERHANG_M, center_y, rail_z)
        side_x = front_x + 0.5 * (SHELF_SIZE_M[0] - RAIL_OVERHANG_M)
        side_y = 0.5 * SHELF_SIZE_M[1] - 0.5 * RAIL_THICKNESS_M + RAIL_OVERHANG_M
        poses["rail_left"] = pose(side_x, center_y + side_y, rail_z)
        poses["rail_right"] = pose(side_x, center_y - side_y, rail_z)
        # The back panel carries the shelf from the counter; its height is fixed, so it is sunk into the counter.
        poses["shelf_back"] = pose(
            front_x + SHELF_SIZE_M[0] - 0.5 * BOARD_THICKNESS_M, center_y, surface_z - SHELF_SIZE_M[2] - 0.30
        )
        row_x = front_x + ROW_DEPTH_M
        slot_ys = [center_y + (index - 0.5 * (NUM_SLOTS - 1)) * SLOT_SPACING_M for index in range(NUM_SLOTS)]
        order = permutation(generator, NUM_SLOTS)
        occupied = sorted(order[: self.num_existing])
        existing_xy = []
        for name, slot in zip(self.existing_names(), occupied):
            poses[name] = pose(row_x, slot_ys[slot], surface_z + 0.5 * EXISTING_ITEM_SIZE_M[2] + 0.001)
            existing_xy.append([row_x, slot_ys[slot]])
        stand_x = uniform(generator, 0.40, 0.45)
        stand_y = uniform(generator, -0.42, -0.38)
        stand_top_z = uniform(generator, 0.53, 0.56)
        poses["stand"] = pose(stand_x, stand_y, stand_top_z - 0.30)
        # The item faces the level hand that reaches for it from the robot.
        poses["item"] = pose(stand_x, stand_y, stand_top_z + 0.5 * ITEM_SIZE_M[2] + 0.001, math.atan2(stand_y, stand_x))
        params = {
            "front_x": torch.tensor(front_x),
            "center_y": torch.tensor(center_y),
            "surface_z": torch.tensor(surface_z),
            "slot_ys": torch.tensor(slot_ys),
            "free_mask": torch.tensor([index not in occupied for index in range(NUM_SLOTS)]),
            "existing_xy": torch.tensor(existing_xy),
            "stand_xyz": torch.tensor([stand_x, stand_y, stand_top_z]),
        }
        return LayoutSample(part_poses_e=self.part_poses_tensor(poses), layout_params=params)

    # ----------------------------------------------------------------------------------------------------------------

    def shelf_target_names(self) -> list[str]:
        """Return the visibility targets on the shelf: its surface and the existing items."""
        return ["shelf_surface"] + self.existing_names()

    def visibility_targets(self) -> list[VisibilityTarget]:
        targets = [VisibilityTarget("shelf_surface", "shelf", (FACE_PZ,))]
        targets += [VisibilityTarget(name, name, ALL_FACES, spacing_m=0.02) for name in self.existing_names()]
        return targets

    def hidden_target_names(self) -> list[str]:
        return self.shelf_target_names()

    def memory_target_names(self) -> list[str]:
        return self.shelf_target_names()

    def manipulated_item_names(self) -> list[str]:
        return ["item"]

    def item_support_names(self) -> list[str]:
        return ["counter", "shelf", "stand"]

    def forbidden_item_contacts(self) -> list[tuple[str, str]]:
        return [("item", name) for name in self.existing_names()]

    def commitment_made(self, env, episode, active: torch.Tensor) -> torch.Tensor:
        """Return whether the gripper has released the item after the instruction: the answer is where it lands."""
        return self.released_after_instruction(env, episode, active)

    def hidden_phase(self, env, episode) -> torch.Tensor:
        return episode.instruction_revealed & ~episode.task_state_buffer(
            "released_after_instruction", False, torch.bool
        )

    def instruction_dim(self) -> int:
        return 1

    def instruction(self, env, episode) -> torch.Tensor:
        return episode.instruction_revealed.float().unsqueeze(-1)

    def _slot_of_item(self, env, episode) -> torch.Tensor:
        """Return the index (N,) of the slot the item stands in, or -1."""
        params = episode.layout_params
        t_E_item = env.arena_world.get_pose_e("item")[:, :3]
        row_x = params["front_x"] + ROW_DEPTH_M
        in_row = (t_E_item[:, 0] - row_x).abs() < ROW_TOLERANCE_M
        on_surface = (t_E_item[:, 2] - 0.5 * ITEM_SIZE_M[2] - params["surface_z"]).abs() < ON_SURFACE_TOLERANCE_M
        near = (t_E_item[:, None, 1] - params["slot_ys"]).abs() < SLOT_TOLERANCE_M
        hit = near & (in_row & on_surface).unsqueeze(-1)
        return torch.where(
            hit.any(dim=-1), hit.float().argmax(dim=-1), torch.full_like(hit[:, 0], -1, dtype=torch.long)
        )

    def success_terms(self, env, episode) -> dict[str, torch.Tensor]:
        params = episode.layout_params
        slot = self._slot_of_item(env, episode)
        in_free = (slot >= 0) & params["free_mask"].gather(1, slot.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        moved = []
        for index, name in enumerate(self.existing_names()):
            t_E_existing_xy = env.arena_world.get_pose_e(name)[:, :2]
            displacement = (t_E_existing_xy - params["existing_xy"][:, index]).norm(dim=-1)
            moved.append(displacement > EXISTING_ITEM_MAX_SHIFT_M)
        hand_out = tool_position_e(env, self)[:, 0] < params["front_x"] - HAND_OUT_MARGIN_M
        return {
            "instruction_revealed": episode.instruction_revealed,
            "item_in_free_slot": in_free,
            "item_upright_at_rest": part_upright(env, "item") & velocity_below_threshold(
                env, "item", AT_REST_SPEED_MPS
            ),
            "existing_items_undisturbed": ~torch.stack(moved, dim=-1).any(dim=-1),
            "released_and_out": ~self.gripper_closed(env) & hand_out,
        }

    def extra_failures(self, env, episode) -> torch.Tensor:
        return env.arena_world.get_pose_e("item")[:, 2] < DROPPED_Z_M

    # ----------------------------------------------------------------------------------------------------------------

    def plan(self, env, episode, env_id: int, mode: str) -> list[Waypoint]:
        params = {key: value[env_id] for key, value in episode.layout_params.items()}
        front_x = float(params["front_x"])
        center_y = float(params["center_y"])
        surface_z = float(params["surface_z"])
        stand_x, stand_y, stand_top_z = (float(v) for v in params["stand_xyz"])
        if mode == PlanMode.ORACLE:
            free_slots = []
            for index, is_free in enumerate(params["free_mask"].tolist()):
                if is_free:
                    free_slots.append(index)
            slot = free_slots[0]
        else:
            # Ignore the remembered shelf: put the item into the middle slot.
            slot = NUM_SLOTS // 2
        slot_y = float(params["slot_ys"][slot])
        row_x = front_x + ROW_DEPTH_M

        q_E_look = pitched_quat(0.0, math.radians(45.0))
        stand_yaw = math.atan2(stand_y, stand_x)
        at_stand = level_quat(stand_yaw)
        at_slot = level_quat(math.atan2(slot_y, row_x))
        pick_z = stand_top_z + GRASP_HEIGHT_M
        back_x = stand_x - APPROACH_SETBACK_M * math.cos(stand_yaw)
        back_y = stand_y - APPROACH_SETBACK_M * math.sin(stand_yaw)
        carry_z = surface_z + RAIL_HEIGHT_M + RAIL_CLEARANCE_M + GRASP_HEIGHT_M
        place_z = surface_z + GRASP_HEIGHT_M + 0.003
        return [
            Waypoint(
                (front_x - LOOK_SETBACK_M, center_y, surface_z + LOOK_HEIGHT_M),
                q_E_look,
                label="look",
                max_speed_mps=0.3,
                tolerance_m=0.03,
                hold_steps=10,
            ),
            # Turn the hand level (camera under the wrist) clear of the stand, then come down in front of the item.
            Waypoint(
                (back_x, back_y, pick_z + 0.10),
                at_stand,
                label="turn",
                max_speed_mps=0.3,
                tolerance_m=0.02,
                tolerance_rad=0.1,
            ),
            Waypoint((back_x, back_y, pick_z), at_stand, label="to_stand", max_speed_mps=0.15, tolerance_m=0.01),
            Waypoint((stand_x, stand_y, pick_z), at_stand, label="reach", max_speed_mps=0.08, tolerance_m=0.006),
            Waypoint((stand_x, stand_y, pick_z), at_stand, GRIPPER_CLOSE, hold_steps=12, label="close"),
            Waypoint((stand_x, stand_y, pick_z + 0.03), at_stand, GRIPPER_CLOSE, label="lift", max_speed_mps=0.08),
            Waypoint((back_x, back_y, pick_z + 0.03), at_stand, GRIPPER_CLOSE, label="back", max_speed_mps=0.12),
            Waypoint(
                (back_x, back_y, pick_z + 0.03),
                at_stand,
                GRIPPER_CLOSE,
                label="wait_for_instruction",
                gate="instruction_revealed",
            ),
            Waypoint((front_x - 0.07, slot_y, carry_z), at_slot, GRIPPER_CLOSE, label="to_shelf", max_speed_mps=0.2),
            Waypoint(
                (row_x, slot_y, carry_z),
                at_slot,
                GRIPPER_CLOSE,
                label="over_rail",
                max_speed_mps=0.05,
                tolerance_m=0.006,
            ),
            Waypoint(
                (row_x, slot_y, place_z),
                at_slot,
                GRIPPER_CLOSE,
                label="lower",
                max_speed_mps=0.03,
                tolerance_m=0.004,
                align_item="item",
            ),
            Waypoint((row_x, slot_y, place_z), at_slot, GRIPPER_OPEN, hold_steps=15, label="release"),
            # Pull the open fingers off the item slowly so that they do not drag it.
            Waypoint((front_x - 0.06, slot_y, place_z), at_slot, label="pull_out", max_speed_mps=0.03),
            Waypoint((front_x - 0.06, slot_y, place_z), at_slot, label="done", max_steps=HOLD_FOREVER_STEPS),
        ]
