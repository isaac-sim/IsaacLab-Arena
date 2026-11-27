# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Kitting from bins.

A two-tier rack beside the robot holds N open bins, each with a kit item of a distinct color; a kit tray lies on the
counter in front of the robot. The robot looks at the rack once, turns to the tray, and then receives M = 2 requests
one after the other. Each request (a one-hot color) is revealed only once the rack has been out of view for 3.5 s,
longer than the 3 s frame history. For each request the robot must reach back, pick the requested kit item from its bin
on the first reach, carry it to the tray and place it there.

Hidden: the rack and its kit items at the start of each reach (from the request until the hand is 0.42 m from the rack).
The tray's visibility while the hand is over the bins is logged but not claimed: the tilted wrist camera can catch it at
the edge of its view, and the robot sees the tray again whenever it turns back to it, so it is not a memory demand.
Randomized: which kit item is in which bin, rack tier heights, the tray position, the requested kit items.
Difficulty knob: number of bins N (2 to 6).
Success: both requested kit items lie in the tray, each was picked on the first reach (first gripper close after its
request on the requested kit item), no other kit item or fixture was touched, and the rack was not re-observed for more than
6 s between a request and its grasp (else "searched again", a failure).
"""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.episode_state import NOT_SET
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
from isaaclab_arena_environments.spatial_memory.parts import ITEM_COLORS, PartSpec, cylinder_item
from isaaclab_arena_environments.spatial_memory.predicates import tool_position_e
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

NUM_REQUESTS = 2
"""Number of kit items requested per episode, one after the other."""
HIDDEN_PHASE_END_Y_M = 0.06
"""Environment-frame y of the tool center point at which each request's hidden phase ends, 0.42 m from the low tier."""
KIT_ITEM_SIZE_M = (0.045, 0.045, 0.08)
"""Size of each kit item, an upright cylinder, so a grasp works from any wrist yaw."""
TRAY_BASE_M = (0.22, 0.30, 0.01)
"""Extents of the kit tray's base."""
TRAY_WALL_HEIGHT_M = 0.025
"""Height of the kit tray's walls above its base."""
TRAY_SPOT_OFFSETS_Y_M = (0.075, -0.075)
"""Lateral offsets from the tray's center of the spots for the first and the second kit item."""
TRAY_EDGE_MARGIN_M = 0.02
"""Minimum horizontal distance between a kit item's center and the tray's edge for the item to be inside the tray."""
AT_RACK_Y_M = 0.44
"""Environment-frame y of the tool center point beyond which the hand is at the rack (and the tray must be out of
view)."""
FIRST_REACH_TOLERANCE_M = 0.04
"""Maximum horizontal distance between each request's first gripper close and the requested kit item's start."""
DROPPED_Z_M = -0.3
"""Environment-frame height below which a kit item counts as dropped."""


class KittingLogic(SpatialMemoryTaskLogic):
    """Logic of the kitting task (see the module docstring)."""

    name = "spatial_memory_kitting"
    rest_joint_pose = REST_JOINT_POSE
    initial_joint_pose = LOOK_JOINT_POSE
    difficulty_name = "num_bins"
    default_difficulty = 5
    episode_length_s = 70.0
    task_description = "Look at the bin rack, then put each requested item into the kit tray."
    forbids_searching_again = True
    reobservation_budget_s = 6.0

    @property
    def num_bins(self) -> int:
        """Return the number of bins on the rack."""
        num = int(round(self.difficulty))
        assert NUM_REQUESTS <= num <= len(SLOT_XS_M) * len(TIER_YS_M), f"num_bins must be between 2 and 6, got {num}"
        return num

    def kit_item_names(self) -> list[str]:
        """Return the names of the kit items, one per bin."""
        return [f"kit_item_{index}" for index in range(self.num_bins)]

    def bin_names(self, index: int) -> list[str]:
        """Return the names of the boards of bin ``index``."""
        return open_box_names(f"bin_{index}")

    def tray_names(self) -> list[str]:
        """Return the names of the tray's boards."""
        return open_box_names("tray")

    def build_part_specs(self) -> list[PartSpec]:
        parts = [counter_spec(), *rack_part_specs()]
        for index in range(self.num_bins):
            parts += open_box_part_specs(f"bin_{index}", BIN_BASE_M, BIN_WALL_HEIGHT_M, "gray", "gray")
        parts += open_box_part_specs("tray", TRAY_BASE_M, TRAY_WALL_HEIGHT_M, "white", "dark")
        for index, name in enumerate(self.kit_item_names()):
            parts.append(
                cylinder_item(name, KIT_ITEM_SIZE_M[0], KIT_ITEM_SIZE_M[2], color=ITEM_COLORS[index], mass_kg=0.08)
            )
        return parts

    def sample_layout(self, generator: torch.Generator) -> LayoutSample:
        poses = self.parked_part_poses()
        low_z = uniform(generator, -0.02, 0.04)
        high_z = low_z + uniform(generator, 0.10, 0.15)
        poses["counter"] = counter_pose(center_x=0.60, center_y=-0.30)
        place_rack(poses, low_z, high_z)
        slots = rack_slots(low_z, high_z)
        order = permutation(generator, len(slots))
        kit_item_xyz = []
        for index, name in enumerate(self.kit_item_names()):
            slot_x, slot_y, slot_z = slots[order[index]]
            floor_z = slot_z + BIN_BASE_M[2]
            place_open_box(poses, f"bin_{index}", (slot_x, slot_y), BIN_BASE_M, BIN_WALL_HEIGHT_M, floor_z)
            poses[name] = pose(slot_x, slot_y, floor_z + 0.5 * KIT_ITEM_SIZE_M[2] + 0.001)
            kit_item_xyz.append([slot_x, slot_y, floor_z])
        tray_x = uniform(generator, 0.40, 0.48)
        tray_y = uniform(generator, -0.30, -0.20)
        place_open_box(poses, "tray", (tray_x, tray_y), TRAY_BASE_M, TRAY_WALL_HEIGHT_M, TRAY_BASE_M[2])
        requests = permutation(generator, self.num_bins)[:NUM_REQUESTS]
        params = {
            "requests": torch.tensor(requests),
            "kit_item_xyz": torch.tensor(kit_item_xyz),
            "tier_z": torch.tensor([low_z, high_z]),
            "tray_xy": torch.tensor([tray_x, tray_y]),
        }
        return LayoutSample(part_poses_e=self.part_poses_tensor(poses), layout_params=params)

    # ----------------------------------------------------------------------------------------------------------------

    def rack_target_names(self) -> list[str]:
        """Return the visibility targets on the rack: the kit items and the bins."""
        return self.kit_item_names() + [f"bin_{index}" for index in range(self.num_bins)]

    def visibility_targets(self) -> list[VisibilityTarget]:
        targets = [VisibilityTarget(name, name, ALL_FACES, spacing_m=0.02) for name in self.kit_item_names()]
        targets += [VisibilityTarget(f"bin_{index}", f"bin_{index}_base", (FACE_PZ,)) for index in range(self.num_bins)]
        targets.append(VisibilityTarget("tray", "tray_base", (FACE_PZ,)))
        return targets

    def hidden_target_names(self) -> list[str]:
        return self.rack_target_names()

    def remembered_target_names(self) -> list[str]:
        """Return the kit items in the bins."""
        return self.kit_item_names()

    def memory_target_names(self) -> list[str]:
        return self.rack_target_names()

    def item_support_names(self) -> list[str]:
        names = ["counter", "tier_low", "tier_high"] + self.tray_names()
        for index in range(self.num_bins):
            names += self.bin_names(index)
        return names

    def on_reset(self, env, episode, env_ids: torch.Tensor) -> None:
        episode.task_state_buffer("request_index", 0, torch.long)
        episode.task_state_buffer("request_revealed", False, torch.bool)
        episode.task_state_buffer("unseen_steps", 0, torch.long)
        episode.task_state_buffer("reobservation_steps", 0, torch.long, (NUM_REQUESTS,))
        episode.task_state_buffer("close_xy", 0.0, torch.float32, (NUM_REQUESTS, 2))
        episode.task_state_buffer("closed_for", False, torch.bool, (NUM_REQUESTS,))
        episode.task_state_buffer("was_closed", False, torch.bool)
        episode.task_state_buffer("delivered", False, torch.bool, (NUM_REQUESTS,))

    def requested_kit_item(self, episode) -> torch.Tensor:
        """Return the index (N,) of the currently requested kit item."""
        state = episode.task_state
        index = state["request_index"].clamp(max=NUM_REQUESTS - 1)
        return episode.layout_params["requests"].long().gather(1, index.unsqueeze(-1)).squeeze(-1)

    def _rack_target_mask(self, env, episode) -> torch.Tensor:
        """Return (N, T): rack targets, excluding kit items already delivered to the tray."""
        mask = torch.zeros(env.num_envs, len(episode.target_names), dtype=torch.bool, device=env.device)
        for index in range(self.num_bins):
            mask[:, episode.target_names.index(f"bin_{index}")] = True
        kit_item_target_ids = [episode.target_names.index(name) for name in self.kit_item_names()]
        delivered_kit_items = torch.zeros(env.num_envs, self.num_bins, dtype=torch.bool, device=env.device)
        delivered_kit_items.scatter_(1, episode.layout_params["requests"].long(), episode.task_state["delivered"])
        mask[:, kit_item_target_ids] = ~delivered_kit_items
        return mask

    def on_step(self, env, episode, visible: torch.Tensor, active: torch.Tensor) -> None:
        state = episode.task_state
        rack_visible = (visible & self._rack_target_mask(env, episode)).any(dim=-1) & active
        episode.memory_seen |= rack_visible
        state["unseen_steps"] = torch.where(
            rack_visible, torch.zeros_like(state["unseen_steps"]), state["unseen_steps"] + 1
        )
        delay_steps = int(math.ceil(self.instruction_delay_s / episode.step_dt))
        pending = state["request_index"] < NUM_REQUESTS
        # A request is revealed when the previous one was delivered (or none was made) and the rack has been out of
        # view for the reveal delay.
        ready = pending & ~state["request_revealed"] & episode.memory_seen & (state["unseen_steps"] >= delay_steps)
        if ready.any():
            state["request_revealed"] |= ready
            episode.instruction_revealed |= ready
            first = ready & (episode.instruction_reveal_step == NOT_SET)
            episode.instruction_reveal_step[first] = env.episode_length_buf[first]
        # Gripper closes after a reveal belong to the current request.
        closed = self.gripper_closed(env) & active
        rising = closed & ~state["was_closed"]
        state["was_closed"] = closed
        env_index = torch.arange(env.num_envs, device=env.device)
        current = state["request_index"].clamp(max=NUM_REQUESTS - 1)
        open_request = pending & state["request_revealed"] & ~state["closed_for"][env_index, current]
        counting = open_request & rack_visible
        state["reobservation_steps"][env_index[counting], current[counting]] += 1
        episode.reobservation_steps = state["reobservation_steps"].amax(dim=-1)
        budget_steps = int(math.floor(self.reobservation_budget_s / episode.step_dt))
        episode.searched_again |= episode.reobservation_steps > budget_steps
        first_close = rising & open_request
        if first_close.any():
            t_E_tcp_xy = tool_position_e(env, self)[:, :2]
            state["close_xy"][env_index[first_close], current[first_close]] = t_E_tcp_xy[first_close]
            state["closed_for"][env_index[first_close], current[first_close]] = True
        # A request is delivered when its kit item rests in the tray; the next request then waits for its reveal.
        in_tray = self._kit_items_in_tray(env, episode)
        requested = episode.layout_params["requests"].long()
        delivered_now = in_tray.gather(1, requested) & ~closed.unsqueeze(-1)
        state["delivered"] |= delivered_now & state["closed_for"]
        advance = pending & state["request_revealed"] & state["delivered"][env_index, current]
        if advance.any():
            state["request_index"][advance] += 1
            state["request_revealed"][advance] = False

    def _kit_items_in_tray(self, env, episode) -> torch.Tensor:
        """Return (N, B): whether each kit item rests inside the tray."""
        t_E_tray_xy = episode.layout_params["tray_xy"]
        flags = []
        for name in self.kit_item_names():
            t_E_item = env.arena_world.get_pose_e(name)[:, :3]
            inside = ((t_E_item[:, 0] - t_E_tray_xy[:, 0]).abs() < 0.5 * TRAY_BASE_M[0] - TRAY_EDGE_MARGIN_M) & (
                (t_E_item[:, 1] - t_E_tray_xy[:, 1]).abs() < 0.5 * TRAY_BASE_M[1] - TRAY_EDGE_MARGIN_M
            )
            low = t_E_item[:, 2] < TRAY_BASE_M[2] + KIT_ITEM_SIZE_M[2]
            flags.append(inside & low)
        return torch.stack(flags, dim=-1)

    def instruction_dim(self) -> int:
        return self.num_bins

    def instruction(self, env, episode) -> torch.Tensor:
        state = episode.task_state
        one_hot = torch.nn.functional.one_hot(self.requested_kit_item(episode), self.num_bins).float()
        shown = state["request_revealed"] & (state["request_index"] < NUM_REQUESTS)
        return one_hot * shown.unsqueeze(-1).float()

    def hidden_phase(self, env, episode) -> torch.Tensor:
        state = episode.task_state
        t_E_tcp_y = tool_position_e(env, self)[:, 1]
        env_index = torch.arange(env.num_envs, device=env.device)
        current = state["request_index"].clamp(max=NUM_REQUESTS - 1)
        before_grasp = state["request_revealed"] & ~state["closed_for"][env_index, current]
        reaching = before_grasp & (t_E_tcp_y < HIDDEN_PHASE_END_Y_M)
        # Facing the rack: the hand is over the bins and has not yet closed on the requested kit item.
        at_rack = before_grasp & (t_E_tcp_y > AT_RACK_Y_M)
        return reaching | at_rack

    def hidden_target_mask(self, env, episode, in_hidden_phase: torch.Tensor) -> torch.Tensor:
        t_E_tcp_y = tool_position_e(env, self)[:, 1]
        at_rack = (t_E_tcp_y > AT_RACK_Y_M) & in_hidden_phase
        reaching = in_hidden_phase & (t_E_tcp_y < HIDDEN_PHASE_END_Y_M)
        mask = self._rack_target_mask(env, episode) & reaching.unsqueeze(-1)
        mask[:, episode.target_names.index("tray")] = at_rack
        return mask

    def static_part_mask(self, env, episode) -> torch.Tensor:
        mask = torch.ones(env.num_envs, len(episode.part_names), dtype=torch.bool, device=env.device)
        kit_item_ids = torch.tensor(episode.part_ids(self.kit_item_names()), device=env.device)
        requested = torch.zeros(env.num_envs, self.num_bins, dtype=torch.bool, device=env.device)
        requested.scatter_(1, episode.layout_params["requests"].long(), True)
        mask[:, kit_item_ids] = ~requested
        return mask

    def robot_touchable_part_mask(self, env, episode) -> torch.Tensor:
        """Return the requested kit items: the robot may touch no other part."""
        requested = torch.zeros(env.num_envs, self.num_bins, dtype=torch.bool, device=env.device)
        requested.scatter_(1, episode.layout_params["requests"].long(), True)
        mask = torch.zeros(env.num_envs, len(episode.part_names), dtype=torch.bool, device=env.device)
        mask[:, episode.part_ids(self.kit_item_names())] = requested
        return mask

    def _first_reach_ok(self, episode) -> torch.Tensor:
        """Return (N, R): whether each request's first gripper close, if any, was at the requested kit item."""
        state = episode.task_state
        requested = episode.layout_params["requests"].long()
        t_E_requested_xyz = episode.layout_params["kit_item_xyz"].gather(1, requested.unsqueeze(-1).expand(-1, -1, 3))
        near = (state["close_xy"] - t_E_requested_xyz[..., :2]).norm(dim=-1) < FIRST_REACH_TOLERANCE_M
        return near | ~state["closed_for"]

    def success_terms(self, env, episode) -> dict[str, torch.Tensor]:
        state = episode.task_state
        in_tray = self._kit_items_in_tray(env, episode).gather(1, episode.layout_params["requests"].long())
        return {
            "all_requests_given": state["request_index"] >= NUM_REQUESTS - 1,
            "requested_kit_items_in_tray": in_tray.all(dim=-1) & ~self.gripper_closed(env),
            "each_picked_on_first_reach": self._first_reach_ok(episode).all(dim=-1) & state["closed_for"].all(dim=-1),
        }

    def extra_failures(self, env, episode) -> torch.Tensor:
        dropped_flags = []
        for name in self.kit_item_names():
            dropped_flags.append(env.arena_world.get_pose_e(name)[:, 2] < DROPPED_Z_M)
        dropped = torch.stack(dropped_flags, dim=-1).any(dim=-1)
        return ~self._first_reach_ok(episode).all(dim=-1) | dropped

    def plan_gates(self, env, episode) -> dict[str, torch.Tensor]:
        state = episode.task_state
        return {
            "request_0": state["request_revealed"] & (state["request_index"] == 0),
            "request_1": state["request_revealed"] & (state["request_index"] == 1),
        }

    # ----------------------------------------------------------------------------------------------------------------

    def plan(self, env, episode, env_id: int, mode: str) -> list[Waypoint]:
        params = {key: value[env_id] for key, value in episode.layout_params.items()}
        tray_x, tray_y = (float(v) for v in params["tray_xy"])
        low_z = float(params["tier_z"][0])
        high_z = float(params["tier_z"][1])
        camera_to_base = top_down_quat_camera_towards_base
        bin_floor_low = low_z + BIN_BASE_M[2]
        return_z = bin_floor_low + KIT_ITEM_SIZE_M[2] + 0.10
        look = camera_to_base(LOOK_POSITION_E[0], LOOK_POSITION_E[1])
        plan = [Waypoint(LOOK_POSITION_E, look, label="look", max_speed_mps=0.3, tolerance_m=0.03, hold_steps=10)]
        wait_xy = (tray_x, tray_y)
        for request in range(NUM_REQUESTS):
            if mode == PlanMode.ORACLE:
                kit_item = int(params["requests"][request])
                item_x, item_y, floor_z = (float(v) for v in params["kit_item_xyz"][kit_item])
                drop_x, drop_y = tray_x, tray_y + TRAY_SPOT_OFFSETS_Y_M[request]
            else:
                # Ignore the remembered layout: reach for a middle bin of the low tier and drop at the nominal tray spot.
                item_x, item_y, floor_z = SLOT_XS_M[1], TIER_YS_M[0], low_z + BIN_BASE_M[2]
                drop_x, drop_y = 0.44, -0.25 + TRAY_SPOT_OFFSETS_Y_M[request]
            grasp_z = floor_z + KIT_ITEM_SIZE_M[2] - 0.02
            over_z = return_z if item_y < TIER_YS_M[1] - 0.05 else max(return_z, high_z + BIN_BASE_M[2] + 0.13)
            at_wait = camera_to_base(*wait_xy)
            at_edge = camera_to_base(item_x, RACK_EDGE_Y_M)
            at_item = camera_to_base(item_x, item_y)
            at_drop = camera_to_base(drop_x, drop_y)
            plan += [
                Waypoint(
                    (*wait_xy, COUNTER_WAIT_Z_M), at_wait, label=f"wait_{request}", max_speed_mps=0.35, tolerance_m=0.03
                ),
                Waypoint(
                    (*wait_xy, COUNTER_WAIT_Z_M),
                    at_wait,
                    label=f"request_{request}",
                    gate=f"request_{request}",
                    tolerance_m=0.03,
                ),
                # The rack edge is a pass-through point: settling there would only extend the view of the rack.
                Waypoint(
                    (item_x, RACK_EDGE_Y_M, return_z), at_edge, label="return", max_speed_mps=0.3, tolerance_m=0.05
                ),
                Waypoint((item_x, item_y, over_z), at_item, label="over_kit_item", max_speed_mps=0.5),
                Waypoint((item_x, item_y, grasp_z), at_item, label="descend", max_speed_mps=0.25, tolerance_m=0.006),
                Waypoint((item_x, item_y, grasp_z), at_item, GRIPPER_CLOSE, hold_steps=12, label="close"),
                Waypoint((item_x, item_y, over_z + 0.04), at_item, GRIPPER_CLOSE, label="lift", max_speed_mps=0.2),
                Waypoint((drop_x, drop_y, 0.20), at_drop, GRIPPER_CLOSE, label="to_tray", max_speed_mps=0.35),
                Waypoint(
                    (drop_x, drop_y, TRAY_BASE_M[2] + KIT_ITEM_SIZE_M[2] - 0.02 + 0.01),
                    at_drop,
                    GRIPPER_CLOSE,
                    label="lower_into_tray",
                    max_speed_mps=0.15,
                    tolerance_m=0.008,
                ),
                Waypoint(
                    (drop_x, drop_y, TRAY_BASE_M[2] + KIT_ITEM_SIZE_M[2] - 0.02 + 0.01),
                    at_drop,
                    GRIPPER_OPEN,
                    hold_steps=8,
                    label="release",
                ),
                Waypoint((drop_x, drop_y, COUNTER_WAIT_Z_M), at_drop, label="up", max_speed_mps=0.2),
            ]
            wait_xy = (drop_x, drop_y)
        plan.append(
            Waypoint((*wait_xy, COUNTER_WAIT_Z_M), camera_to_base(*wait_xy), label="done", max_steps=HOLD_FOREVER_STEPS)
        )
        return plan
