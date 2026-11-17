# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task 8: kitting from bins.

A two-tier rack beside the robot holds N open bins, each with a part of a distinct colour; a kit tray lies on the
counter in front of the robot. The robot looks at the rack once, turns to the tray, and then receives M = 2 requests
one after the other. Each request (a one-hot colour) is revealed only once the rack has been out of view for 3.5 s,
longer than the 3 s frame history. For each request the robot must reach back, pick the requested part from its bin on
the first reach, carry it to the tray and place it there.

Hidden: the rack and its parts at the start of each reach (from the request until the hand is 0.36 m from the rack).
The tray's visibility while the hand is over the bins is logged but not claimed: the tilted wrist camera can catch it at
the edge of its view, and the robot sees the tray again whenever it turns back to it, so it is not a memory demand.
Randomised: which part is in which bin, rack tier heights, the tray position, the requested parts.
Difficulty knob: number of bins N (2 to 6).
Success: both requested parts lie in the tray, each was picked on the first reach (first gripper close after its
request on the requested part), no other part or fixture was touched, and the rack was not re-observed for more than
5 s between a request and its grasp (else "searched again", a failure).
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
from isaaclab_arena_environments.spatial_memory.parts import ITEM_COLOURS, PartSpec, board, cylinder_item
from isaaclab_arena_environments.spatial_memory.predicates import item_pose_e, tool_position_e
from isaaclab_arena_environments.spatial_memory.runtime import NOT_SET
from isaaclab_arena_environments.spatial_memory.task_logic import LayoutSample, PlanMode, SpatialMemoryTaskLogic
from isaaclab_arena_environments.spatial_memory.tasks.fetch_behind import (
    LOOK_JOINT_POSE,
    LOOK_POSITION,
    REACH_START_END_Y,
    REST_JOINT_POSE,
    SHELF_CENTER_X,
    SHELF_EDGE_Y,
    SLOT_XS,
    TIER_SIZE_M,
    TIER_YS,
)
from isaaclab_arena_environments.spatial_memory.visibility import VisibilityTarget

NUM_REQUESTS = 2
PART_SIZE_M = (0.045, 0.045, 0.08)
"""Parts are upright cylinders, so a grasp works from any wrist yaw."""
BIN_BASE_M = (0.105, 0.13, 0.01)
BIN_WALL_HEIGHT_M = 0.03
BIN_WALL_T_M = 0.006
TRAY_BASE_M = (0.22, 0.30, 0.01)
TRAY_WALL_HEIGHT_M = 0.025
TRAY_SPOT_OFFSETS_Y = (0.075, -0.075)
"""Where the first and the second part go in the tray, relative to its centre."""
AT_RACK_Y = 0.44
"""The hand is at the rack (and the tray must be out of view) beyond this y."""
FIRST_REACH_TOLERANCE_M = 0.04
COUNTER_WAIT_Z = 0.25

__all__ = ["KittingLogic", "LOOK_JOINT_POSE", "REST_JOINT_POSE"]


class KittingLogic(SpatialMemoryTaskLogic):
    """Logic of task 8 (see module docstring)."""

    name = "spatial_memory_kitting"
    difficulty_name = "num_bins"
    default_difficulty = 5
    episode_length_s = 70.0
    task_description = "Look at the bin rack, then put each requested part into the kit tray."
    forbids_searching_again = True
    reobservation_budget_s = 5.0

    @property
    def num_bins(self) -> int:
        num = int(round(self.difficulty))
        assert NUM_REQUESTS <= num <= len(SLOT_XS) * len(TIER_YS), f"num_bins must be between 2 and 6, got {num}"
        return num

    def part_names_(self) -> list[str]:
        return [f"part_{index}" for index in range(self.num_bins)]

    def bin_names(self, index: int) -> list[str]:
        return [f"bin_{index}_{side}" for side in ("base", "front", "back", "left", "right")]

    def tray_names(self) -> list[str]:
        return [f"tray_{side}" for side in ("base", "front", "back", "left", "right")]

    def build_part_specs(self) -> list[PartSpec]:
        parts = [
            counter_spec(),
            board("tier_low", TIER_SIZE_M, colour="wood"),
            board("tier_high", TIER_SIZE_M, colour="wood"),
            board("riser", (TIER_SIZE_M[0], TIER_SIZE_M[1], 0.30), colour="dark"),
        ]
        for index in range(self.num_bins):
            names = self.bin_names(index)
            parts.append(board(names[0], BIN_BASE_M, colour="grey"))
            for name in names[1:3]:
                parts.append(board(name, (BIN_BASE_M[0], BIN_WALL_T_M, BIN_WALL_HEIGHT_M), colour="grey"))
            for name in names[3:5]:
                parts.append(board(name, (BIN_WALL_T_M, BIN_BASE_M[1], BIN_WALL_HEIGHT_M), colour="grey"))
        tray = self.tray_names()
        parts.append(board(tray[0], TRAY_BASE_M, colour="white"))
        for name in tray[1:3]:
            parts.append(board(name, (TRAY_BASE_M[0], BIN_WALL_T_M, TRAY_WALL_HEIGHT_M), colour="dark"))
        for name in tray[3:5]:
            parts.append(board(name, (BIN_WALL_T_M, TRAY_BASE_M[1], TRAY_WALL_HEIGHT_M), colour="dark"))
        for index, name in enumerate(self.part_names_()):
            parts.append(cylinder_item(name, PART_SIZE_M[0], PART_SIZE_M[2], colour=ITEM_COLOURS[index], mass_kg=0.08))
        return parts

    def _place_box_walls(self, table, names, center, base_size, wall_height, base_top_z):
        cx, cy = center
        table[names[0]] = pose(cx, cy, base_top_z - 0.5 * base_size[2])
        wall_z = base_top_z + 0.5 * wall_height
        half_x = 0.5 * base_size[0] - 0.5 * BIN_WALL_T_M
        half_y = 0.5 * base_size[1] - 0.5 * BIN_WALL_T_M
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
        slots = [(x, TIER_YS[tier], (low_z, high_z)[tier]) for tier in range(2) for x in SLOT_XS]
        order = permutation(generator, len(slots))
        part_xyz = []
        for index, name in enumerate(self.part_names_()):
            slot_x, slot_y, slot_z = slots[order[index]]
            floor_z = slot_z + BIN_BASE_M[2]
            self._place_box_walls(
                table, self.bin_names(index), (slot_x, slot_y), BIN_BASE_M, BIN_WALL_HEIGHT_M, floor_z
            )
            table[name] = pose(slot_x, slot_y, floor_z + 0.5 * PART_SIZE_M[2] + 0.001)
            part_xyz.append([slot_x, slot_y, floor_z])
        tray_x = uniform(generator, 0.40, 0.48)
        tray_y = uniform(generator, -0.30, -0.20)
        self._place_box_walls(
            table, self.tray_names(), (tray_x, tray_y), TRAY_BASE_M, TRAY_WALL_HEIGHT_M, TRAY_BASE_M[2]
        )
        requests = permutation(generator, self.num_bins)[:NUM_REQUESTS]
        params = {
            "requests": torch.tensor(requests),
            "part_xyz": torch.tensor(part_xyz),
            "tier_z": torch.tensor([low_z, high_z]),
            "tray_xy": torch.tensor([tray_x, tray_y]),
        }
        return LayoutSample(part_poses_e=self.pose_table_to_tensor(table), params=params)

    # ----------------------------------------------------------------------------------------------------------------

    def rack_target_names(self) -> list[str]:
        return self.part_names_() + [f"bin_{index}" for index in range(self.num_bins)]

    def visibility_targets(self) -> list[VisibilityTarget]:
        targets = [VisibilityTarget(name, name, ALL_FACES, spacing_m=0.02) for name in self.part_names_()]
        targets += [VisibilityTarget(f"bin_{index}", f"bin_{index}_base", (FACE_PZ,)) for index in range(self.num_bins)]
        targets.append(VisibilityTarget("tray", "tray_base", (FACE_PZ,)))
        return targets

    def hidden_target_names(self) -> list[str]:
        return self.rack_target_names()

    def memory_target_names(self) -> list[str]:
        return self.rack_target_names()

    def item_support_names(self) -> list[str]:
        names = ["counter", "tier_low", "tier_high"] + self.tray_names()
        for index in range(self.num_bins):
            names += self.bin_names(index)
        return names

    def _state(self, runtime, key):
        return runtime.task_state[key]

    def on_reset(self, env, runtime, env_ids: torch.Tensor) -> None:
        state = runtime.task_state
        if "request_index" not in state:
            num = env.num_envs
            state["request_index"] = torch.zeros(num, dtype=torch.long, device=env.device)
            state["request_revealed"] = torch.zeros(num, dtype=torch.bool, device=env.device)
            state["unseen_steps"] = torch.zeros(num, dtype=torch.long, device=env.device)
            state["reobs_steps"] = torch.zeros(num, NUM_REQUESTS, dtype=torch.long, device=env.device)
            state["close_xy"] = torch.zeros(num, NUM_REQUESTS, 2, device=env.device)
            state["closed_for"] = torch.zeros(num, NUM_REQUESTS, dtype=torch.bool, device=env.device)
            state["was_closed"] = torch.zeros(num, dtype=torch.bool, device=env.device)
            state["delivered"] = torch.zeros(num, NUM_REQUESTS, dtype=torch.bool, device=env.device)
        for key in ("request_index", "unseen_steps", "reobs_steps", "close_xy"):
            state[key][env_ids] = 0
        for key in ("request_revealed", "closed_for", "was_closed", "delivered"):
            state[key][env_ids] = False

    def requested_part(self, runtime) -> torch.Tensor:
        """Return the index (N,) of the currently requested part."""
        state = runtime.task_state
        index = state["request_index"].clamp(max=NUM_REQUESTS - 1)
        return runtime.params["requests"].long().gather(1, index.unsqueeze(-1)).squeeze(-1)

    def _rack_target_mask(self, env, runtime) -> torch.Tensor:
        """Return (N, T): rack targets, excluding parts already delivered to the tray."""
        mask = torch.zeros(env.num_envs, len(runtime.target_names), dtype=torch.bool, device=env.device)
        for index in range(self.num_bins):
            mask[:, runtime.target_names.index(f"bin_{index}")] = True
        part_target_ids = [runtime.target_names.index(name) for name in self.part_names_()]
        delivered_parts = torch.zeros(env.num_envs, self.num_bins, dtype=torch.bool, device=env.device)
        delivered_parts.scatter_(1, runtime.params["requests"].long(), runtime.task_state["delivered"])
        mask[:, part_target_ids] = ~delivered_parts
        return mask

    def on_step(self, env, runtime, visible: torch.Tensor, active: torch.Tensor) -> None:
        state = runtime.task_state
        rack_visible = (visible & self._rack_target_mask(env, runtime)).any(dim=-1) & active
        runtime.memory_seen |= rack_visible
        state["unseen_steps"] = torch.where(
            rack_visible, torch.zeros_like(state["unseen_steps"]), state["unseen_steps"] + 1
        )
        delay_steps = int(math.ceil(self.reveal_delay_s / runtime.step_dt))
        pending = state["request_index"] < NUM_REQUESTS
        # A request is revealed when the previous one was delivered (or none was made) and the rack has been out of
        # view for the reveal delay.
        ready = pending & ~state["request_revealed"] & runtime.memory_seen & (state["unseen_steps"] >= delay_steps)
        if ready.any():
            state["request_revealed"] |= ready
            runtime.revealed |= ready
            first = ready & (runtime.reveal_step == NOT_SET)
            runtime.reveal_step[first] = env.episode_length_buf[first]
        # Gripper closes after a reveal belong to the current request.
        closed = self.gripper_closed(env) & active
        rising = closed & ~state["was_closed"]
        state["was_closed"] = closed
        env_index = torch.arange(env.num_envs, device=env.device)
        current = state["request_index"].clamp(max=NUM_REQUESTS - 1)
        open_request = pending & state["request_revealed"] & ~state["closed_for"][env_index, current]
        counting = open_request & rack_visible
        state["reobs_steps"][env_index[counting], current[counting]] += 1
        runtime.reobservation_steps = state["reobs_steps"].amax(dim=-1)
        budget_steps = int(math.floor(self.reobservation_budget_s / runtime.step_dt))
        runtime.searched_again |= runtime.reobservation_steps > budget_steps
        first_close = rising & open_request
        if first_close.any():
            tool_xy = tool_position_e(env, self)[:, :2]
            state["close_xy"][env_index[first_close], current[first_close]] = tool_xy[first_close]
            state["closed_for"][env_index[first_close], current[first_close]] = True
        # A request is delivered when its part rests in the tray; the next request then waits for its reveal.
        in_tray = self._parts_in_tray(env, runtime)
        requested = runtime.params["requests"].long()
        delivered_now = in_tray.gather(1, requested) & ~closed.unsqueeze(-1)
        state["delivered"] |= delivered_now & state["closed_for"]
        advance = pending & state["request_revealed"] & state["delivered"][env_index, current]
        if advance.any():
            state["request_index"][advance] += 1
            state["request_revealed"][advance] = False

    def _parts_in_tray(self, env, runtime) -> torch.Tensor:
        """Return (N, B): whether each part rests inside the tray."""
        tray_xy = runtime.params["tray_xy"]
        flags = []
        for name in self.part_names_():
            part = item_pose_e(env, name)
            inside = ((part[:, 0] - tray_xy[:, 0]).abs() < 0.5 * TRAY_BASE_M[0] - 0.02) & (
                (part[:, 1] - tray_xy[:, 1]).abs() < 0.5 * TRAY_BASE_M[1] - 0.02
            )
            low = part[:, 2] < TRAY_BASE_M[2] + PART_SIZE_M[2]
            flags.append(inside & low)
        return torch.stack(flags, dim=-1)

    def instruction_dim(self) -> int:
        return self.num_bins

    def instruction(self, env, runtime) -> torch.Tensor:
        state = runtime.task_state
        one_hot = torch.nn.functional.one_hot(self.requested_part(runtime), self.num_bins).float()
        shown = state["request_revealed"] & (state["request_index"] < NUM_REQUESTS)
        return one_hot * shown.unsqueeze(-1).float()

    def action_phase(self, env, runtime) -> torch.Tensor:
        state = runtime.task_state
        tool_y = tool_position_e(env, self)[:, 1]
        env_index = torch.arange(env.num_envs, device=env.device)
        current = state["request_index"].clamp(max=NUM_REQUESTS - 1)
        before_grasp = state["request_revealed"] & ~state["closed_for"][env_index, current]
        reaching = before_grasp & (tool_y < REACH_START_END_Y)
        # Facing the rack: the hand is over the bins and has not yet closed on the requested part.
        at_rack = before_grasp & (tool_y > AT_RACK_Y)
        return reaching | at_rack

    def hidden_phase_mask(self, env, runtime, in_action: torch.Tensor) -> torch.Tensor:
        tool_y = tool_position_e(env, self)[:, 1]
        at_rack = (tool_y > AT_RACK_Y) & in_action
        reaching = in_action & (tool_y < REACH_START_END_Y)
        mask = self._rack_target_mask(env, runtime) & reaching.unsqueeze(-1)
        mask[:, runtime.target_names.index("tray")] = at_rack
        return mask

    def static_part_mask(self, env, runtime) -> torch.Tensor:
        mask = torch.ones(env.num_envs, len(runtime.part_names), dtype=torch.bool, device=env.device)
        part_ids = torch.tensor(runtime.part_ids(self.part_names_()), device=env.device)
        requested = torch.zeros(env.num_envs, self.num_bins, dtype=torch.bool, device=env.device)
        requested.scatter_(1, runtime.params["requests"].long(), True)
        mask[:, part_ids] = ~requested
        return mask

    def allowed_contacts(self, env, runtime) -> torch.Tensor:
        """The robot may touch only the requested parts; parts may rest on supports and touch each other."""
        from isaaclab_arena_environments.spatial_memory.collision import filtered_sensor_labels

        labels = filtered_sensor_labels(self)
        allowed = torch.zeros(env.num_envs, len(labels), len(runtime.part_names), dtype=torch.bool, device=env.device)
        part_ids = runtime.part_ids(self.part_names_())
        support_ids = runtime.part_ids(self.item_support_names())
        requested = torch.zeros(env.num_envs, self.num_bins, dtype=torch.bool, device=env.device)
        requested.scatter_(1, runtime.params["requests"].long(), True)
        for sensor_index, label in enumerate(labels):
            if label in self.embodiment.arm_links or label in self.embodiment.gripper_bodies:
                allowed[:, sensor_index, part_ids] = requested
            else:
                allowed[:, sensor_index, part_ids] = True
                allowed[:, sensor_index, support_ids] = True
        return allowed

    def _first_reach_ok(self, runtime) -> torch.Tensor:
        state = runtime.task_state
        requested = runtime.params["requests"].long()
        targets = runtime.params["part_xyz"].gather(1, requested.unsqueeze(-1).expand(-1, -1, 3))[..., :2]
        near = (state["close_xy"] - targets).norm(dim=-1) < FIRST_REACH_TOLERANCE_M
        return near | ~state["closed_for"]

    def success_terms(self, env, runtime) -> dict[str, torch.Tensor]:
        state = runtime.task_state
        in_tray = self._parts_in_tray(env, runtime).gather(1, runtime.params["requests"].long())
        return {
            "all_requests_given": state["request_index"] >= NUM_REQUESTS - 1,
            "requested_parts_in_tray": in_tray.all(dim=-1) & ~self.gripper_closed(env),
            "each_picked_on_first_reach": self._first_reach_ok(runtime).all(dim=-1) & state["closed_for"].all(dim=-1),
        }

    def extra_failures(self, env, runtime) -> torch.Tensor:
        dropped = torch.stack([item_pose_e(env, name)[:, 2] < -0.3 for name in self.part_names_()], dim=-1).any(dim=-1)
        return ~self._first_reach_ok(runtime).all(dim=-1) | dropped

    def plan_gates(self, env, runtime) -> dict[str, torch.Tensor]:
        state = runtime.task_state
        return {
            "request_0": state["request_revealed"] & (state["request_index"] == 0),
            "request_1": state["request_revealed"] & (state["request_index"] == 1),
        }

    # ----------------------------------------------------------------------------------------------------------------

    def plan(self, env, runtime, env_id: int, mode: str) -> list[Waypoint]:
        params = {key: value[env_id] for key, value in runtime.params.items()}
        tray_x, tray_y = (float(v) for v in params["tray_xy"])
        low_z = float(params["tier_z"][0])
        high_z = float(params["tier_z"][1])

        def camera_to_base(x: float, y: float) -> tuple[float, float, float, float]:
            return top_down_quat(math.atan2(-y, -x))

        bin_floor_low = low_z + BIN_BASE_M[2]
        return_z = bin_floor_low + PART_SIZE_M[2] + 0.10
        look = camera_to_base(LOOK_POSITION[0], LOOK_POSITION[1])
        plan = [Waypoint(LOOK_POSITION, look, label="look", max_speed_mps=0.3, tolerance_m=0.03, hold_steps=10)]
        wait_xy = (tray_x, tray_y)
        for request in range(NUM_REQUESTS):
            if mode == PlanMode.ORACLE:
                part = int(params["requests"][request])
                part_x, part_y, floor_z = (float(v) for v in params["part_xyz"][part])
                drop_x, drop_y = tray_x, tray_y + TRAY_SPOT_OFFSETS_Y[request]
            else:
                # Ignore the remembered layout: reach for a middle bin of the low tier and drop at the nominal tray spot.
                part_x, part_y, floor_z = SLOT_XS[1], TIER_YS[0], low_z + BIN_BASE_M[2]
                drop_x, drop_y = 0.44, -0.25 + TRAY_SPOT_OFFSETS_Y[request]
            grasp_z = floor_z + PART_SIZE_M[2] - 0.02
            over_z = return_z if part_y < TIER_YS[1] - 0.05 else max(return_z, high_z + BIN_BASE_M[2] + 0.13)
            at_wait = camera_to_base(*wait_xy)
            at_edge = camera_to_base(part_x, SHELF_EDGE_Y)
            at_part = camera_to_base(part_x, part_y)
            at_drop = camera_to_base(drop_x, drop_y)
            plan += [
                Waypoint(
                    (*wait_xy, COUNTER_WAIT_Z), at_wait, label=f"wait_{request}", max_speed_mps=0.35, tolerance_m=0.03
                ),
                Waypoint(
                    (*wait_xy, COUNTER_WAIT_Z),
                    at_wait,
                    label=f"request_{request}",
                    gate=f"request_{request}",
                    tolerance_m=0.03,
                ),
                Waypoint(
                    (part_x, SHELF_EDGE_Y, return_z), at_edge, label="return", max_speed_mps=0.3, tolerance_m=0.02
                ),
                Waypoint((part_x, part_y, over_z), at_part, label="over_part", max_speed_mps=0.5),
                Waypoint((part_x, part_y, grasp_z), at_part, label="descend", max_speed_mps=0.25, tolerance_m=0.006),
                Waypoint((part_x, part_y, grasp_z), at_part, GRIPPER_CLOSE, hold_steps=12, label="close"),
                Waypoint((part_x, part_y, over_z + 0.04), at_part, GRIPPER_CLOSE, label="lift", max_speed_mps=0.2),
                Waypoint((drop_x, drop_y, 0.20), at_drop, GRIPPER_CLOSE, label="to_tray", max_speed_mps=0.35),
                Waypoint(
                    (drop_x, drop_y, TRAY_BASE_M[2] + PART_SIZE_M[2] - 0.02 + 0.01),
                    at_drop,
                    GRIPPER_CLOSE,
                    label="lower_into_tray",
                    max_speed_mps=0.15,
                    tolerance_m=0.008,
                ),
                Waypoint(
                    (drop_x, drop_y, TRAY_BASE_M[2] + PART_SIZE_M[2] - 0.02 + 0.01),
                    at_drop,
                    GRIPPER_OPEN,
                    hold_steps=8,
                    label="release",
                ),
                Waypoint((drop_x, drop_y, COUNTER_WAIT_Z), at_drop, label="up", max_speed_mps=0.2),
            ]
            wait_xy = (drop_x, drop_y)
        plan.append(Waypoint((*wait_xy, COUNTER_WAIT_Z), camera_to_base(*wait_xy), label="done", max_steps=10_000))
        return plan
