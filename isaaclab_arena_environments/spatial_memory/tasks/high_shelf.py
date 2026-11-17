# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task 6: put the item on the high shelf.

A shelf at about the height of the Franka's shoulder has a front rail and five slots, N of them holding low items that
are shorter than the rail. The robot looks down at the shelf once from above, then picks up a tall item from a stand
next to it with the hand level (the wrist camera under the wrist). The place signal is given once the shelf has been
out of view for 3.5 s, longer than the 3 s frame history. The robot must then lift the item over the rail into a free
slot. From below the rail, nothing on the shelf is visible: the rail and the shelf board block the view, so the free
slot must be remembered.

Hidden: the shelf surface and the existing items from the signal until the release.
Randomised: shelf position and height, which slots are occupied, the stand's position and height.
Difficulty knob: number of existing items N (1 to 4 of 5 slots; more is harder).
Success: the item stands upright at rest in a slot that was free, the gripper is open and out of the shelf, no
existing item moved by more than 2 cm, no collision, and the shelf was not re-observed for more than 1 s between the
signal and the release (else "searched again", a failure).
"""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.controller import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    Waypoint,
    pitched_quat,
    tool_quat_pointing,
)
from isaaclab_arena_environments.spatial_memory.geometry import ALL_FACES, FACE_PZ
from isaaclab_arena_environments.spatial_memory.layout_utils import (
    BOARD_T_M,
    counter_pose,
    counter_spec,
    permutation,
    pose,
    uniform,
)
from isaaclab_arena_environments.spatial_memory.parts import ITEM_COLOURS, PartSpec, board, box_item
from isaaclab_arena_environments.spatial_memory.predicates import (
    item_at_rest,
    item_pose_e,
    item_upright,
    tool_position_e,
)
from isaaclab_arena_environments.spatial_memory.task_logic import LayoutSample, PlanMode, SpatialMemoryTaskLogic
from isaaclab_arena_environments.spatial_memory.visibility import VisibilityTarget

NUM_SLOTS = 5
SLOT_SPACING_M = 0.10
SHELF_SIZE_M = (0.25, 0.58, BOARD_T_M)
RAIL_T_M = 0.015
RAIL_HEIGHT_M = 0.06
"""Rails close the shelf's front and ends above its surface."""
RAIL_OVERHANG_M = 0.005
"""The rails overhang the shelf board's edges and each other, so no line of sight passes between them."""
ROW_DEPTH_M = 0.065
"""Distance of the slot row's centre behind the shelf front."""
EXISTING_ITEM_SIZE_M = (0.05, 0.07, 0.04)
"""Existing items are lower than the rail, so they cannot be seen from below it."""
ITEM_SIZE_M = (0.05, 0.06, 0.15)
"""The item to place (depth, grasped width, height)."""
GRASP_HEIGHT_M = 0.10
"""Height of the tool centre point above the item's base while it is held."""
RAIL_CLEARANCE_M = 0.012
"""Clearance of the item's base above the rail while it is carried over it."""
STAND_SIZE_M = (0.10, 0.10)
APPROACH_SETBACK_M = 0.10
LOOK_SETBACK_M = 0.12
LOOK_HEIGHT_M = 0.12
"""Tool position in front of and above the shelf front in the look pose (hand pitched down, camera above it)."""
REST_JOINT_POSE = (0.0, -0.3, 0.0, -2.2, 0.0, 3.47, -2.356)
"""Hand level and pointing forward, with the wrist camera under the wrist."""

__all__ = ["HighShelfLogic", "REST_JOINT_POSE"]


def level_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    """Return a level tool orientation with heading ``yaw_rad`` and the wrist camera (hand x) below the wrist."""
    return tool_quat_pointing((math.cos(yaw_rad), math.sin(yaw_rad), 0.0), (0.0, 0.0, -1.0))


class HighShelfLogic(SpatialMemoryTaskLogic):
    """Logic of task 6 (see module docstring)."""

    name = "spatial_memory_high_shelf"
    difficulty_name = "num_existing_items"
    default_difficulty = 4
    episode_length_s = 45.0
    task_description = "Look at the high shelf, pick up the item, and put it on the shelf in a free spot."
    forbids_searching_again = True
    reobservation_budget_s = 1.0

    @property
    def num_existing(self) -> int:
        num = int(round(self.difficulty))
        assert 1 <= num <= NUM_SLOTS - 1, f"num_existing_items must be between 1 and {NUM_SLOTS - 1}, got {num}"
        return num

    def existing_names(self) -> list[str]:
        return [f"existing_{index}" for index in range(self.num_existing)]

    def build_part_specs(self) -> list[PartSpec]:
        parts = [
            counter_spec(),
            board("shelf", SHELF_SIZE_M, colour="wood"),
            board("rail", (RAIL_T_M, SHELF_SIZE_M[1] + 2 * RAIL_OVERHANG_M, RAIL_HEIGHT_M), colour="dark"),
            board("rail_left", (SHELF_SIZE_M[0] + RAIL_OVERHANG_M, RAIL_T_M, RAIL_HEIGHT_M), colour="dark"),
            board("rail_right", (SHELF_SIZE_M[0] + RAIL_OVERHANG_M, RAIL_T_M, RAIL_HEIGHT_M), colour="dark"),
            board("shelf_back", (BOARD_T_M, SHELF_SIZE_M[1], 0.60), colour="grey"),
            board("stand", (STAND_SIZE_M[0], STAND_SIZE_M[1], 0.60), colour="grey"),
        ]
        for index, name in enumerate(self.existing_names()):
            parts.append(box_item(name, EXISTING_ITEM_SIZE_M, colour=ITEM_COLOURS[1 + index], mass_kg=0.1))
        parts.append(box_item("item", ITEM_SIZE_M, colour="red", mass_kg=0.08))
        return parts

    def sample_layout(self, generator: torch.Generator) -> LayoutSample:
        table = self.new_pose_table()
        front_x = uniform(generator, 0.53, 0.57)
        center_y = uniform(generator, -0.03, 0.03)
        surface_z = uniform(generator, 0.57, 0.61)
        table["counter"] = counter_pose()
        table["shelf"] = pose(front_x + 0.5 * SHELF_SIZE_M[0], center_y, surface_z - 0.5 * SHELF_SIZE_M[2])
        rail_z = surface_z + 0.5 * RAIL_HEIGHT_M
        table["rail"] = pose(front_x + 0.5 * RAIL_T_M - RAIL_OVERHANG_M, center_y, rail_z)
        side_x = front_x + 0.5 * (SHELF_SIZE_M[0] - RAIL_OVERHANG_M)
        side_y = 0.5 * SHELF_SIZE_M[1] - 0.5 * RAIL_T_M + RAIL_OVERHANG_M
        table["rail_left"] = pose(side_x, center_y + side_y, rail_z)
        table["rail_right"] = pose(side_x, center_y - side_y, rail_z)
        # The back panel carries the shelf from the counter; its height is fixed, so it is sunk into the counter.
        table["shelf_back"] = pose(
            front_x + SHELF_SIZE_M[0] - 0.5 * BOARD_T_M, center_y, surface_z - SHELF_SIZE_M[2] - 0.30
        )
        row_x = front_x + ROW_DEPTH_M
        slot_ys = [center_y + (index - 0.5 * (NUM_SLOTS - 1)) * SLOT_SPACING_M for index in range(NUM_SLOTS)]
        order = permutation(generator, NUM_SLOTS)
        occupied = sorted(order[: self.num_existing])
        existing_xy = []
        for name, slot in zip(self.existing_names(), occupied):
            table[name] = pose(row_x, slot_ys[slot], surface_z + 0.5 * EXISTING_ITEM_SIZE_M[2] + 0.001)
            existing_xy.append([row_x, slot_ys[slot]])
        stand_x = uniform(generator, 0.40, 0.45)
        stand_y = uniform(generator, -0.42, -0.38)
        stand_top_z = uniform(generator, 0.53, 0.56)
        table["stand"] = pose(stand_x, stand_y, stand_top_z - 0.30)
        # The item faces the level hand that reaches for it from the robot.
        table["item"] = pose(stand_x, stand_y, stand_top_z + 0.5 * ITEM_SIZE_M[2] + 0.001, math.atan2(stand_y, stand_x))
        params = {
            "front_x": torch.tensor(front_x),
            "center_y": torch.tensor(center_y),
            "surface_z": torch.tensor(surface_z),
            "slot_ys": torch.tensor(slot_ys),
            "free_mask": torch.tensor([index not in occupied for index in range(NUM_SLOTS)]),
            "existing_xy": torch.tensor(existing_xy),
            "stand_xyz": torch.tensor([stand_x, stand_y, stand_top_z]),
        }
        return LayoutSample(part_poses_e=self.pose_table_to_tensor(table), params=params)

    # ----------------------------------------------------------------------------------------------------------------

    def shelf_target_names(self) -> list[str]:
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

    def allowed_contacts(self, env, runtime) -> torch.Tensor:
        """The default contacts, except that the item may not touch the existing items."""
        from isaaclab_arena_environments.spatial_memory.collision import filtered_sensor_labels

        allowed = super().allowed_contacts(env, runtime)[0].clone()
        labels = filtered_sensor_labels(self)
        existing_ids = runtime.part_ids(self.existing_names())
        allowed[labels.index("item"), existing_ids] = False
        for name in self.existing_names():
            allowed[labels.index(name), runtime.part_index["item"]] = False
        return allowed.unsqueeze(0).expand(env.num_envs, -1, -1)

    def on_reset(self, env, runtime, env_ids: torch.Tensor) -> None:
        state = runtime.task_state
        if "released_after_signal" not in state:
            state["released_after_signal"] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            state["was_closed"] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        state["released_after_signal"][env_ids] = False
        state["was_closed"][env_ids] = False

    def on_step(self, env, runtime, visible: torch.Tensor, active: torch.Tensor) -> None:
        # The memory probe counts re-observation until the first gripper close after the signal; here the commitment is
        # the release, so the count runs until the gripper opens after the signal.
        state = runtime.task_state
        indices = [runtime.target_names.index(name) for name in self.memory_target_names()]
        memory_visible = visible[:, indices].any(dim=-1) & active
        runtime.memory_seen |= memory_visible
        runtime.memory_unseen_steps = torch.where(
            memory_visible, torch.zeros_like(runtime.memory_unseen_steps), runtime.memory_unseen_steps + 1
        )
        delay_steps = int(math.ceil(self.reveal_delay_s / runtime.step_dt))
        newly_revealed = ~runtime.revealed & runtime.memory_seen & (runtime.memory_unseen_steps >= delay_steps) & active
        runtime.reveal_step[newly_revealed] = env.episode_length_buf[newly_revealed]
        runtime.revealed |= newly_revealed
        closed = self.gripper_closed(env) & active
        released = state["was_closed"] & ~closed & runtime.revealed
        state["released_after_signal"] |= released
        state["was_closed"] = closed
        counting = runtime.revealed & ~newly_revealed & ~state["released_after_signal"]
        runtime.reobservation_steps += (counting & memory_visible).long()
        budget_steps = int(math.floor(self.reobservation_budget_s / runtime.step_dt))
        runtime.searched_again |= runtime.reobservation_steps > budget_steps

    def action_phase(self, env, runtime) -> torch.Tensor:
        return runtime.revealed & ~runtime.task_state["released_after_signal"]

    def instruction_dim(self) -> int:
        return 1

    def instruction(self, env, runtime) -> torch.Tensor:
        return runtime.revealed.float().unsqueeze(-1)

    def _slot_of_item(self, env, runtime) -> torch.Tensor:
        """Return the index (N,) of the slot the item stands in, or -1."""
        params = runtime.params
        item = item_pose_e(env, "item")
        row_x = params["front_x"] + ROW_DEPTH_M
        in_row = (item[:, 0] - row_x).abs() < 0.03
        on_surface = (item[:, 2] - 0.5 * ITEM_SIZE_M[2] - params["surface_z"]).abs() < 0.012
        near = (item[:, None, 1] - params["slot_ys"]).abs() < 0.03
        hit = near & (in_row & on_surface).unsqueeze(-1)
        return torch.where(
            hit.any(dim=-1), hit.float().argmax(dim=-1), torch.full_like(hit[:, 0], -1, dtype=torch.long)
        )

    def success_terms(self, env, runtime) -> dict[str, torch.Tensor]:
        params = runtime.params
        slot = self._slot_of_item(env, runtime)
        in_free = (slot >= 0) & params["free_mask"].gather(1, slot.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        moved = []
        for index, name in enumerate(self.existing_names()):
            moved.append((item_pose_e(env, name)[:, :2] - params["existing_xy"][:, index]).norm(dim=-1) > 0.02)
        hand_out = tool_position_e(env, self)[:, 0] < params["front_x"] - 0.02
        return {
            "signal_given": runtime.revealed,
            "item_in_free_slot": in_free,
            "item_upright_at_rest": item_upright(env, "item") & item_at_rest(env, "item"),
            "existing_items_undisturbed": ~torch.stack(moved, dim=-1).any(dim=-1),
            "released_and_out": ~self.gripper_closed(env) & hand_out,
        }

    def extra_failures(self, env, runtime) -> torch.Tensor:
        return item_pose_e(env, "item")[:, 2] < 0.3

    # ----------------------------------------------------------------------------------------------------------------

    def plan(self, env, runtime, env_id: int, mode: str) -> list[Waypoint]:
        params = {key: value[env_id] for key, value in runtime.params.items()}
        front_x = float(params["front_x"])
        center_y = float(params["center_y"])
        surface_z = float(params["surface_z"])
        stand_x, stand_y, stand_top_z = (float(v) for v in params["stand_xyz"])
        if mode == PlanMode.ORACLE:
            free = [index for index, is_free in enumerate(params["free_mask"].tolist()) if is_free]
            slot = free[0]
        else:
            # Ignore the remembered shelf: put the item into the middle slot.
            slot = NUM_SLOTS // 2
        slot_y = float(params["slot_ys"][slot])
        row_x = front_x + ROW_DEPTH_M

        look_quat = pitched_quat(0.0, math.radians(45.0))
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
                look_quat,
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
                (back_x, back_y, pick_z + 0.03), at_stand, GRIPPER_CLOSE, label="wait_for_signal", gate="revealed"
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
            Waypoint((front_x - 0.06, slot_y, place_z), at_slot, label="done", max_steps=10_000),
        ]
