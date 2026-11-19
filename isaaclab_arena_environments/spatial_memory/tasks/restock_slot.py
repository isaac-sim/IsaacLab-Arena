# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Restock into the empty slot.

A two-tier rack beside the robot has six bins (the slot grid); all but K hold stock items. The robot looks at the rack
once, turns to the counter and picks up a new item there. The instruction to restock is revealed once the rack has
been out of view for 3.5 s, longer than the 3 s frame history. The robot must then carry the item back and put it into a bin that
was empty: the answer is a free region, not an object identity.

Hidden: every bin and stock item at the start of the return (from when the instruction is revealed until the hand is
0.43 m from the rack).
Randomized: which bins are empty, rack tier heights, the new item's position.
Difficulty knob: number of empty bins K (1 to 3; fewer is harder).
Success: the new item rests upright in a bin that was empty, the gripper is open and lifted out, no stock item moved
by more than 2 cm, no collision, and the rack was not re-observed for more than 5 s between the instruction being
revealed and the release (else "searched again", a failure).
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
)
from isaaclab_arena_environments.spatial_memory.geometry import ALL_FACES, FACE_PZ
from isaaclab_arena_environments.spatial_memory.layout_sampling import (
    counter_pose,
    counter_spec,
    permutation,
    pose,
    uniform,
)
from isaaclab_arena_environments.spatial_memory.parts import PartSpec, box_item, cylinder_item
from isaaclab_arena_environments.spatial_memory.predicates import AT_REST_SPEED_MPS, part_upright, tool_position_e
from isaaclab_arena_environments.spatial_memory.task_logic import LayoutSample, PlanMode, SpatialMemoryTaskLogic
from isaaclab_arena_environments.spatial_memory.tasks.side_rack import (
    BIN_BASE_M,
    BIN_WALL_HEIGHT_M,
    COUNTER_WAIT_Z_M,
    LOOK_JOINT_POSE,
    LOOK_POSITION_E,
    RACK_EDGE_Y_M,
    REST_JOINT_POSE,
    SLOT_XS_M,
    TIER_YS_M,
    open_box_names,
    open_box_part_specs,
    place_open_box,
    place_rack,
    rack_part_specs,
    rack_slots,
    top_down_quat_camera_towards_base,
)
from isaaclab_arena_environments.spatial_memory.visibility import VisibilityTarget

NUM_BINS = len(SLOT_XS_M) * len(TIER_YS_M)
"""Number of bins, one per rack slot."""
ITEM_SIZE_M = (0.045, 0.045, 0.08)
"""Size of the stock items (upright cylinders) and the new item (an upright box, which does not rock once placed)."""
GRASP_DEPTH_M = 0.02
"""Depth of the fingertips below the new item's top when grasping from above."""
HIDDEN_PHASE_END_Y_M = 0.05
"""Environment-frame y of the tool center point at which the hidden phase ends, 0.43 m from the low tier."""
BIN_EDGE_MARGIN_M = 0.01
"""Minimum horizontal distance between the new item's center and a bin's edge for the item to be inside the bin."""
ON_FLOOR_TOLERANCE_M = 0.012
"""Maximum vertical distance between the new item's base and a bin floor for the item to rest on it."""
STOCK_MAX_SHIFT_M = 0.02
"""Horizontal displacement of a stock item above which it counts as disturbed."""
LIFTED_OUT_CLEARANCE_M = 0.05
"""Height of the tool center point above the top of an item in the highest bin above which the hand is lifted out."""
DROPPED_Z_M = -0.3
"""Environment-frame height below which the new item counts as dropped."""


class RestockSlotLogic(SpatialMemoryTaskLogic):
    """Logic of the restock-slot task (see the module docstring)."""

    name = "spatial_memory_restock_slot"
    rest_joint_pose = REST_JOINT_POSE
    initial_joint_pose = LOOK_JOINT_POSE
    difficulty_name = "num_empty_bins"
    default_difficulty = 1
    episode_length_s = 45.0
    task_description = "Look at the rack, pick up the new item, and put it into an empty bin."
    forbids_searching_again = True
    reobservation_budget_s = 5.0

    @property
    def num_empty(self) -> int:
        """Return the number of empty bins."""
        num = int(round(self.difficulty))
        assert 1 <= num <= 3, f"num_empty_bins must be between 1 and 3, got {num}"
        return num

    def stock_names(self) -> list[str]:
        """Return the names of the stock items, enough to fill all bins but one."""
        return [f"stock_{index}" for index in range(NUM_BINS - 1)]

    def bin_names(self, index: int) -> list[str]:
        """Return the names of the boards of bin ``index``."""
        return open_box_names(f"bin_{index}")

    def build_part_specs(self) -> list[PartSpec]:
        parts = [counter_spec(), *rack_part_specs()]
        for index in range(NUM_BINS):
            parts += open_box_part_specs(f"bin_{index}", BIN_BASE_M, BIN_WALL_HEIGHT_M, "gray", "gray")
        # Stock for the most filled case (one empty bin); unused stock is parked away from the scene.
        for name in self.stock_names():
            parts.append(cylinder_item(name, ITEM_SIZE_M[0], ITEM_SIZE_M[2], color="blue", mass_kg=0.08))
        parts.append(box_item("new_item", ITEM_SIZE_M, color="red", mass_kg=0.08))
        return parts

    def sample_layout(self, generator: torch.Generator) -> LayoutSample:
        poses = self.parked_part_poses()
        low_z = uniform(generator, -0.02, 0.04)
        high_z = low_z + uniform(generator, 0.10, 0.15)
        poses["counter"] = counter_pose(center_x=0.60, center_y=-0.30)
        place_rack(poses, low_z, high_z)
        bin_xyz = []
        for index, (slot_x, slot_y, tier_z) in enumerate(rack_slots(low_z, high_z)):
            floor_z = tier_z + BIN_BASE_M[2]
            place_open_box(poses, f"bin_{index}", (slot_x, slot_y), BIN_BASE_M, BIN_WALL_HEIGHT_M, floor_z)
            bin_xyz.append([slot_x, slot_y, floor_z])
        order = permutation(generator, NUM_BINS)
        empty = sorted(order[: self.num_empty])
        filled = []
        for index in range(NUM_BINS):
            if index not in empty:
                filled.append(index)
        stock_xy = []
        for stock_index, name in enumerate(self.stock_names()):
            if stock_index < len(filled):
                x, y, floor_z = bin_xyz[filled[stock_index]]
                poses[name] = pose(x, y, floor_z + 0.5 * ITEM_SIZE_M[2] + 0.001)
                stock_xy.append([x, y])
            else:
                stock_xy.append([poses[name][0], poses[name][1]])
        item_x = uniform(generator, 0.40, 0.48)
        item_y = uniform(generator, -0.30, -0.20)
        # Face the wrist's grasp orientation at this spot, so the fingers close on two faces.
        poses["new_item"] = pose(item_x, item_y, 0.5 * ITEM_SIZE_M[2] + 0.001, math.atan2(-item_y, -item_x))
        empty_mask = [index in empty for index in range(NUM_BINS)]
        params = {
            "bin_xyz": torch.tensor(bin_xyz),
            "empty_mask": torch.tensor(empty_mask),
            "stock_xy": torch.tensor(stock_xy),
            "item_start_xy": torch.tensor([item_x, item_y]),
            "tier_z": torch.tensor([low_z, high_z]),
        }
        return LayoutSample(part_poses_e=self.part_poses_tensor(poses), layout_params=params)

    # ----------------------------------------------------------------------------------------------------------------

    def rack_target_names(self) -> list[str]:
        """Return the visibility targets on the rack: the bins and the stock items."""
        return [f"bin_{index}" for index in range(NUM_BINS)] + self.stock_names()

    def visibility_targets(self) -> list[VisibilityTarget]:
        targets = [VisibilityTarget(f"bin_{index}", f"bin_{index}_base", (FACE_PZ,)) for index in range(NUM_BINS)]
        targets += [VisibilityTarget(name, name, ALL_FACES, spacing_m=0.02) for name in self.stock_names()]
        return targets

    def hidden_target_names(self) -> list[str]:
        return self.rack_target_names()

    def remembered_target_names(self) -> list[str]:
        """Return the bins, whose fill state is what the robot must remember."""
        return [f"bin_{index}" for index in range(NUM_BINS)]

    def memory_target_names(self) -> list[str]:
        return self.rack_target_names()

    def manipulated_item_names(self) -> list[str]:
        return ["new_item"]

    def item_support_names(self) -> list[str]:
        names = ["counter", "tier_low", "tier_high"]
        for index in range(NUM_BINS):
            names += self.bin_names(index)
        return names

    def commitment_made(self, env, episode, active: torch.Tensor) -> torch.Tensor:
        """Return whether the gripper has released the item after the instruction: the answer is where it lands."""
        return self.released_after_instruction(env, episode, active)

    def hidden_phase(self, env, episode) -> torch.Tensor:
        t_E_tcp_y = tool_position_e(env, self)[:, 1]
        return (
            episode.instruction_revealed
            & ~episode.task_state_buffer("released_after_instruction", False, torch.bool)
            & (t_E_tcp_y < HIDDEN_PHASE_END_Y_M)
        )

    def instruction_dim(self) -> int:
        return 1

    def instruction(self, env, episode) -> torch.Tensor:
        return episode.instruction_revealed.float().unsqueeze(-1)

    def _bin_of_item(self, env, episode) -> torch.Tensor:
        """Return the index (N,) of the bin the new item rests in, or -1."""
        t_E_item = env.arena_world.get_pose_e("new_item")[:, :3]
        t_E_bins = episode.layout_params["bin_xyz"]
        dx = (t_E_item[:, None, 0] - t_E_bins[..., 0]).abs()
        dy = (t_E_item[:, None, 1] - t_E_bins[..., 1]).abs()
        inside = (dx < 0.5 * BIN_BASE_M[0] - BIN_EDGE_MARGIN_M) & (dy < 0.5 * BIN_BASE_M[1] - BIN_EDGE_MARGIN_M)
        on_floor = (t_E_item[:, None, 2] - 0.5 * ITEM_SIZE_M[2] - t_E_bins[..., 2]).abs() < ON_FLOOR_TOLERANCE_M
        hit = inside & on_floor
        return torch.where(
            hit.any(dim=-1), hit.float().argmax(dim=-1), torch.full_like(hit[:, 0], -1, dtype=torch.long)
        )

    def success_terms(self, env, episode) -> dict[str, torch.Tensor]:
        bin_index = self._bin_of_item(env, episode)
        in_empty = (bin_index >= 0) & episode.layout_params["empty_mask"].gather(
            1, bin_index.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)
        stock_moved = []
        for index, name in enumerate(self.stock_names()):
            t_E_stock_xy = env.arena_world.get_pose_e(name)[:, :2]
            displacement = (t_E_stock_xy - episode.layout_params["stock_xy"][:, index]).norm(dim=-1)
            stock_moved.append(displacement > STOCK_MAX_SHIFT_M)
        t_E_tcp_z = tool_position_e(env, self)[:, 2]
        highest_floor_z = episode.layout_params["bin_xyz"][..., 2].amax(dim=-1)
        lifted_out = t_E_tcp_z > highest_floor_z + ITEM_SIZE_M[2] + LIFTED_OUT_CLEARANCE_M
        return {
            "instruction_revealed": episode.instruction_revealed,
            "item_in_empty_bin": in_empty,
            "item_upright_at_rest": part_upright(env, "new_item") & velocity_below_threshold(
                env, "new_item", AT_REST_SPEED_MPS
            ),
            "stock_undisturbed": ~torch.stack(stock_moved, dim=-1).any(dim=-1),
            "released_and_lifted_out": ~self.gripper_closed(env) & lifted_out,
        }

    def extra_failures(self, env, episode) -> torch.Tensor:
        return env.arena_world.get_pose_e("new_item")[:, 2] < DROPPED_Z_M

    # ----------------------------------------------------------------------------------------------------------------

    def plan(self, env, episode, env_id: int, mode: str) -> list[Waypoint]:
        params = {key: value[env_id] for key, value in episode.layout_params.items()}
        item_x, item_y = (float(v) for v in params["item_start_xy"])
        low_z = float(params["tier_z"][0])
        high_z = float(params["tier_z"][1])
        if mode == PlanMode.ORACLE:
            empty_bins = []
            for index, is_empty in enumerate(params["empty_mask"].tolist()):
                if is_empty:
                    empty_bins.append(index)
            target = empty_bins[0]
        else:
            # Ignore the remembered layout: put the item into the middle bin of the low tier.
            target = 1
        bin_x, bin_y, floor_z = (float(v) for v in params["bin_xyz"][target])
        camera_to_base = top_down_quat_camera_towards_base
        grip_offset = ITEM_SIZE_M[2] - GRASP_DEPTH_M
        return_z = low_z + BIN_BASE_M[2] + 2.0 * ITEM_SIZE_M[2] + 0.06
        over_z = (
            return_z if bin_y < TIER_YS_M[1] - 0.05 else max(return_z, high_z + BIN_BASE_M[2] + ITEM_SIZE_M[2] + 0.13)
        )
        look = camera_to_base(LOOK_POSITION_E[0], LOOK_POSITION_E[1])
        at_item = camera_to_base(item_x, item_y)
        at_edge = camera_to_base(bin_x, RACK_EDGE_Y_M)
        at_bin = camera_to_base(bin_x, bin_y)
        # Release just above the bin floor.
        place_z = floor_z + grip_offset + 0.004
        return [
            Waypoint(LOOK_POSITION_E, look, label="look", max_speed_mps=0.3, tolerance_m=0.03, hold_steps=10),
            Waypoint(
                (item_x, item_y, COUNTER_WAIT_Z_M), at_item, label="to_item", max_speed_mps=0.35, tolerance_m=0.02
            ),
            Waypoint((item_x, item_y, grip_offset), at_item, label="descend", max_speed_mps=0.2, tolerance_m=0.006),
            Waypoint((item_x, item_y, grip_offset), at_item, GRIPPER_CLOSE, hold_steps=12, label="close"),
            Waypoint((item_x, item_y, COUNTER_WAIT_Z_M), at_item, GRIPPER_CLOSE, label="lift", max_speed_mps=0.2),
            Waypoint(
                (item_x, item_y, COUNTER_WAIT_Z_M),
                at_item,
                GRIPPER_CLOSE,
                label="wait_for_instruction",
                gate="instruction_revealed",
            ),
            Waypoint((bin_x, RACK_EDGE_Y_M, return_z), at_edge, GRIPPER_CLOSE, label="return", max_speed_mps=0.3),
            Waypoint((bin_x, bin_y, over_z), at_bin, GRIPPER_CLOSE, label="over_bin", max_speed_mps=0.4),
            Waypoint(
                (bin_x, bin_y, place_z), at_bin, GRIPPER_CLOSE, label="lower", max_speed_mps=0.2, align_item="new_item"
            ),
            Waypoint((bin_x, bin_y, place_z), at_bin, GRIPPER_OPEN, hold_steps=15, label="release"),
            # Clear the fingers from the item slowly so that they do not drag it.
            Waypoint((bin_x, bin_y, place_z + 0.04), at_bin, label="clear", max_speed_mps=0.03),
            Waypoint((bin_x, bin_y, over_z + 0.05), at_bin, label="lift_out", max_speed_mps=0.2),
            Waypoint((bin_x, bin_y, over_z + 0.05), at_bin, label="done", max_steps=HOLD_FOREVER_STEPS),
        ]
