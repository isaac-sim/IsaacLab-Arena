# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task 9: restock into the empty slot.

A two-tier rack beside the robot has six bins (the slot grid); all but K hold stock items. The robot looks at the rack
once, turns to the counter and picks up a new item there. The restock signal is given once the rack has been out of
view for 3.5 s, longer than the 3 s frame history. The robot must then carry the item back and put it into a bin that
was empty: the answer is a free region, not an object identity.

Hidden: every bin and stock item at the start of the return (from the signal until the hand is 0.43 m from the rack).
Randomised: which bins are empty, rack tier heights, the new item's position.
Difficulty knob: number of empty bins K (1 to 3; fewer is harder).
Success: the new item rests upright in a bin that was empty, the gripper is open and lifted out, no stock item moved
by more than 2 cm, no collision, and the rack was not re-observed for more than 5 s between the signal and the release
(else "searched again", a failure).
"""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.controller import GRIPPER_CLOSE, GRIPPER_OPEN, Waypoint, top_down_quat
from isaaclab_arena_environments.spatial_memory.geometry import ALL_FACES, FACE_PZ
from isaaclab_arena_environments.spatial_memory.layout_utils import (
    counter_pose,
    counter_spec,
    permutation,
    pose,
    uniform,
)
from isaaclab_arena_environments.spatial_memory.parts import PartSpec, board, box_item, cylinder_item
from isaaclab_arena_environments.spatial_memory.predicates import (
    item_at_rest,
    item_pose_e,
    item_upright,
    tool_position_e,
)
from isaaclab_arena_environments.spatial_memory.task_logic import LayoutSample, PlanMode, SpatialMemoryTaskLogic
from isaaclab_arena_environments.spatial_memory.tasks.fetch_behind import (
    LOOK_JOINT_POSE,
    LOOK_POSITION,
    REST_JOINT_POSE,
    SHELF_CENTER_X,
    SHELF_EDGE_Y,
    SLOT_XS,
    TIER_SIZE_M,
    TIER_YS,
)
from isaaclab_arena_environments.spatial_memory.tasks.kitting import (
    BIN_BASE_M,
    BIN_WALL_HEIGHT_M,
    BIN_WALL_T_M,
    COUNTER_WAIT_Z,
)
from isaaclab_arena_environments.spatial_memory.visibility import VisibilityTarget

NUM_BINS = len(SLOT_XS) * len(TIER_YS)
ITEM_SIZE_M = (0.045, 0.045, 0.08)
"""Stock items are upright cylinders; the new item is an upright box, which does not rock once placed."""
GRASP_DEPTH_M = 0.02
REACH_START_END_Y = 0.05
"""The return (action) phase lasts from the signal until the tool passes this y, 0.43 m from the low tier."""

__all__ = ["RestockSlotLogic", "LOOK_JOINT_POSE", "REST_JOINT_POSE"]


class RestockSlotLogic(SpatialMemoryTaskLogic):
    """Logic of task 9 (see module docstring)."""

    name = "spatial_memory_restock_slot"
    difficulty_name = "num_empty_bins"
    default_difficulty = 1
    episode_length_s = 45.0
    task_description = "Look at the rack, pick up the new item, and put it into an empty bin."
    forbids_searching_again = True
    reobservation_budget_s = 5.0

    @property
    def num_empty(self) -> int:
        num = int(round(self.difficulty))
        assert 1 <= num <= 3, f"num_empty_bins must be between 1 and 3, got {num}"
        return num

    def stock_names(self) -> list[str]:
        return [f"stock_{index}" for index in range(NUM_BINS - 1)]

    def bin_names(self, index: int) -> list[str]:
        return [f"bin_{index}_{side}" for side in ("base", "front", "back", "left", "right")]

    def build_part_specs(self) -> list[PartSpec]:
        parts = [
            counter_spec(),
            board("tier_low", TIER_SIZE_M, colour="wood"),
            board("tier_high", TIER_SIZE_M, colour="wood"),
            board("riser", (TIER_SIZE_M[0], TIER_SIZE_M[1], 0.30), colour="dark"),
        ]
        for index in range(NUM_BINS):
            names = self.bin_names(index)
            parts.append(board(names[0], BIN_BASE_M, colour="grey"))
            for name in names[1:3]:
                parts.append(board(name, (BIN_BASE_M[0], BIN_WALL_T_M, BIN_WALL_HEIGHT_M), colour="grey"))
            for name in names[3:5]:
                parts.append(board(name, (BIN_WALL_T_M, BIN_BASE_M[1], BIN_WALL_HEIGHT_M), colour="grey"))
        # Stock for the most filled case (one empty bin); unused stock is parked away from the scene.
        for name in self.stock_names():
            parts.append(cylinder_item(name, ITEM_SIZE_M[0], ITEM_SIZE_M[2], colour="blue", mass_kg=0.08))
        parts.append(box_item("new_item", ITEM_SIZE_M, colour="red", mass_kg=0.08))
        return parts

    def _place_bin(self, table, index, center, floor_z):
        cx, cy = center
        names = self.bin_names(index)
        table[names[0]] = pose(cx, cy, floor_z - 0.5 * BIN_BASE_M[2])
        wall_z = floor_z + 0.5 * BIN_WALL_HEIGHT_M
        half_x = 0.5 * BIN_BASE_M[0] - 0.5 * BIN_WALL_T_M
        half_y = 0.5 * BIN_BASE_M[1] - 0.5 * BIN_WALL_T_M
        table[names[1]] = pose(cx, cy + half_y, wall_z)
        table[names[2]] = pose(cx, cy - half_y, wall_z)
        table[names[3]] = pose(cx + half_x, cy, wall_z)
        table[names[4]] = pose(cx - half_x, cy, wall_z)

    def sample_layout(self, generator: torch.Generator) -> LayoutSample:
        table = self.new_pose_table()
        low_z = uniform(generator, -0.02, 0.04)
        high_z = low_z + uniform(generator, 0.10, 0.15)
        table["counter"] = counter_pose(center_x=0.60, center_y=-0.30)
        table["tier_low"] = pose(SHELF_CENTER_X, TIER_YS[0], low_z - 0.5 * TIER_SIZE_M[2])
        table["tier_high"] = pose(SHELF_CENTER_X, TIER_YS[1], high_z - 0.5 * TIER_SIZE_M[2])
        table["riser"] = pose(SHELF_CENTER_X, TIER_YS[1], high_z - TIER_SIZE_M[2] - 0.15)
        bin_xyz = []
        for index in range(NUM_BINS):
            tier = index // len(SLOT_XS)
            x = SLOT_XS[index % len(SLOT_XS)]
            floor_z = (low_z, high_z)[tier] + BIN_BASE_M[2]
            self._place_bin(table, index, (x, TIER_YS[tier]), floor_z)
            bin_xyz.append([x, TIER_YS[tier], floor_z])
        order = permutation(generator, NUM_BINS)
        empty = sorted(order[: self.num_empty])
        filled = [index for index in range(NUM_BINS) if index not in empty]
        stock_xy = []
        for stock_index, name in enumerate(self.stock_names()):
            if stock_index < len(filled):
                x, y, floor_z = bin_xyz[filled[stock_index]]
                table[name] = pose(x, y, floor_z + 0.5 * ITEM_SIZE_M[2] + 0.001)
                stock_xy.append([x, y])
            else:
                stock_xy.append([table[name][0], table[name][1]])
        item_x = uniform(generator, 0.40, 0.48)
        item_y = uniform(generator, -0.30, -0.20)
        # Face the wrist's grasp orientation at this spot, so the fingers close on two faces.
        table["new_item"] = pose(item_x, item_y, 0.5 * ITEM_SIZE_M[2] + 0.001, math.atan2(-item_y, -item_x))
        empty_mask = [index in empty for index in range(NUM_BINS)]
        params = {
            "bin_xyz": torch.tensor(bin_xyz),
            "empty_mask": torch.tensor(empty_mask),
            "stock_xy": torch.tensor(stock_xy),
            "item_start_xy": torch.tensor([item_x, item_y]),
            "tier_z": torch.tensor([low_z, high_z]),
        }
        return LayoutSample(part_poses_e=self.pose_table_to_tensor(table), params=params)

    # ----------------------------------------------------------------------------------------------------------------

    def rack_target_names(self) -> list[str]:
        return [f"bin_{index}" for index in range(NUM_BINS)] + self.stock_names()

    def visibility_targets(self) -> list[VisibilityTarget]:
        targets = [VisibilityTarget(f"bin_{index}", f"bin_{index}_base", (FACE_PZ,)) for index in range(NUM_BINS)]
        targets += [VisibilityTarget(name, name, ALL_FACES, spacing_m=0.02) for name in self.stock_names()]
        return targets

    def hidden_target_names(self) -> list[str]:
        return self.rack_target_names()

    def memory_target_names(self) -> list[str]:
        return self.rack_target_names()

    def manipulated_item_names(self) -> list[str]:
        return ["new_item"]

    def item_support_names(self) -> list[str]:
        names = ["counter", "tier_low", "tier_high"]
        for index in range(NUM_BINS):
            names += self.bin_names(index)
        return names

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
        tool_y = tool_position_e(env, self)[:, 1]
        return runtime.revealed & ~runtime.task_state["released_after_signal"] & (tool_y < REACH_START_END_Y)

    def instruction_dim(self) -> int:
        return 1

    def instruction(self, env, runtime) -> torch.Tensor:
        return runtime.revealed.float().unsqueeze(-1)

    def _bin_of_item(self, env, runtime) -> torch.Tensor:
        """Return the index (N,) of the bin the new item rests in, or -1."""
        item = item_pose_e(env, "new_item")
        bins = runtime.params["bin_xyz"]
        dx = (item[:, None, 0] - bins[..., 0]).abs()
        dy = (item[:, None, 1] - bins[..., 1]).abs()
        inside = (dx < 0.5 * BIN_BASE_M[0] - 0.01) & (dy < 0.5 * BIN_BASE_M[1] - 0.01)
        on_floor = (item[:, None, 2] - 0.5 * ITEM_SIZE_M[2] - bins[..., 2]).abs() < 0.012
        hit = inside & on_floor
        return torch.where(
            hit.any(dim=-1), hit.float().argmax(dim=-1), torch.full_like(hit[:, 0], -1, dtype=torch.long)
        )

    def success_terms(self, env, runtime) -> dict[str, torch.Tensor]:
        bin_index = self._bin_of_item(env, runtime)
        in_empty = (bin_index >= 0) & runtime.params["empty_mask"].gather(
            1, bin_index.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)
        stock_moved = []
        for index, name in enumerate(self.stock_names()):
            moved = (item_pose_e(env, name)[:, :2] - runtime.params["stock_xy"][:, index]).norm(dim=-1) > 0.02
            stock_moved.append(moved)
        tool_z = tool_position_e(env, self)[:, 2]
        lifted_out = tool_z > runtime.params["bin_xyz"][..., 2].amax(dim=-1) + ITEM_SIZE_M[2] + 0.05
        return {
            "signal_given": runtime.revealed,
            "item_in_empty_bin": in_empty,
            "item_upright_at_rest": item_upright(env, "new_item") & item_at_rest(env, "new_item"),
            "stock_undisturbed": ~torch.stack(stock_moved, dim=-1).any(dim=-1),
            "released_and_lifted_out": ~self.gripper_closed(env) & lifted_out,
        }

    def extra_failures(self, env, runtime) -> torch.Tensor:
        return item_pose_e(env, "new_item")[:, 2] < -0.3

    # ----------------------------------------------------------------------------------------------------------------

    def plan(self, env, runtime, env_id: int, mode: str) -> list[Waypoint]:
        params = {key: value[env_id] for key, value in runtime.params.items()}
        item_x, item_y = (float(v) for v in params["item_start_xy"])
        low_z = float(params["tier_z"][0])
        high_z = float(params["tier_z"][1])
        if mode == PlanMode.ORACLE:
            empty = [index for index, is_empty in enumerate(params["empty_mask"].tolist()) if is_empty]
            target = empty[0]
        else:
            # Ignore the remembered layout: put the item into the middle bin of the low tier.
            target = 1
        bin_x, bin_y, floor_z = (float(v) for v in params["bin_xyz"][target])

        def camera_to_base(x: float, y: float) -> tuple[float, float, float, float]:
            return top_down_quat(math.atan2(-y, -x))

        grip_offset = ITEM_SIZE_M[2] - GRASP_DEPTH_M
        return_z = low_z + BIN_BASE_M[2] + 2.0 * ITEM_SIZE_M[2] + 0.06
        over_z = (
            return_z if bin_y < TIER_YS[1] - 0.05 else max(return_z, high_z + BIN_BASE_M[2] + ITEM_SIZE_M[2] + 0.13)
        )
        look = camera_to_base(LOOK_POSITION[0], LOOK_POSITION[1])
        at_item = camera_to_base(item_x, item_y)
        at_edge = camera_to_base(bin_x, SHELF_EDGE_Y)
        at_bin = camera_to_base(bin_x, bin_y)
        # Release just above the bin floor.
        place_z = floor_z + grip_offset + 0.004
        return [
            Waypoint(LOOK_POSITION, look, label="look", max_speed_mps=0.3, tolerance_m=0.03, hold_steps=10),
            Waypoint((item_x, item_y, COUNTER_WAIT_Z), at_item, label="to_item", max_speed_mps=0.35, tolerance_m=0.02),
            Waypoint((item_x, item_y, grip_offset), at_item, label="descend", max_speed_mps=0.2, tolerance_m=0.006),
            Waypoint((item_x, item_y, grip_offset), at_item, GRIPPER_CLOSE, hold_steps=12, label="close"),
            Waypoint((item_x, item_y, COUNTER_WAIT_Z), at_item, GRIPPER_CLOSE, label="lift", max_speed_mps=0.2),
            Waypoint(
                (item_x, item_y, COUNTER_WAIT_Z), at_item, GRIPPER_CLOSE, label="wait_for_signal", gate="revealed"
            ),
            Waypoint((bin_x, SHELF_EDGE_Y, return_z), at_edge, GRIPPER_CLOSE, label="return", max_speed_mps=0.3),
            Waypoint((bin_x, bin_y, over_z), at_bin, GRIPPER_CLOSE, label="over_bin", max_speed_mps=0.4),
            Waypoint(
                (bin_x, bin_y, place_z), at_bin, GRIPPER_CLOSE, label="lower", max_speed_mps=0.2, align_item="new_item"
            ),
            Waypoint((bin_x, bin_y, place_z), at_bin, GRIPPER_OPEN, hold_steps=15, label="release"),
            # Clear the fingers from the item slowly so that they do not drag it.
            Waypoint((bin_x, bin_y, place_z + 0.04), at_bin, label="clear", max_speed_mps=0.03),
            Waypoint((bin_x, bin_y, over_z + 0.05), at_bin, label="lift_out", max_speed_mps=0.2),
            Waypoint((bin_x, bin_y, over_z + 0.05), at_bin, label="done", max_steps=10_000),
        ]
