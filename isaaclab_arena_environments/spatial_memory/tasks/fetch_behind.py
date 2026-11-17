# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task 2: fetch the mug from the shelf behind you.

N mugs of different colours stand on a two-tier open shelf beside the robot (the stand-in for a cabinet: an open shelf
lets the Franka grasp mugs from above). The episode starts with the wrist camera looking at the shelf. The robot turns
to the counter in front; once the shelf has been out of view for 3.5 s, longer than the 3 s frame history, the
instruction (a one-hot colour) is revealed. The robot must then reach back and grasp that mug on its first reach.

Phases: look (shelf in view), turn (shelf out of view), instruction, reach (action phase: from the instruction until
the hand is 0.36 m from the low tier), grasp and lift.
Randomised: which mug stands in which shelf slot, tier heights, the robot's turn target on the counter.
Difficulty knob: number of mugs N (2 to 6; the shelf has six slots the scripted Franka reaches reliably).
Success: the instructed mug is lifted, the first gripper close was on it, no other mug was touched, and the mugs were
not re-observed for more than 4 s between the instruction and the first close (else "searched again", a failure).
"""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.controller import GRIPPER_CLOSE, Waypoint, top_down_quat
from isaaclab_arena_environments.spatial_memory.geometry import ALL_FACES
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
from isaaclab_arena_environments.spatial_memory.visibility import VisibilityTarget

MUG_DIAMETER_M = 0.05
MUG_HEIGHT_M = 0.08
TIER_SIZE_M = (0.44, 0.16, 0.04)
RISER_HEIGHT_M = 0.30
SLOT_XS = (0.02, 0.145, 0.27)
"""Three slots per tier: the part of the Franka's top-down workspace beside it where the scripted reach is fast and
keeps the shelf out of view, at a spacing that clears the hand."""
TIER_YS = (0.48, 0.64)
SHELF_CENTER_X = 0.145
LOOK_POSITION = (0.145, 0.40, 0.62)
"""The oracle looks at the whole shelf from here before turning to the counter."""
SHELF_EDGE_Y = 0.24
"""The final approach to the instructed mug starts from this y, just in front of the low tier."""
REACH_START_END_Y = 0.12
"""The reach-start (action) phase lasts from the instruction until the tool passes this y, 0.36 m from the low tier."""
COUNTER_WAIT_Z = 0.25
FIRST_REACH_TOLERANCE_M = 0.04
"""The first gripper close must be within this horizontal distance of the instructed mug."""
LOOK_JOINT_POSE = (1.45, -0.6, 0.0, -2.0, 0.0, 1.45, -2.356)
"""Spawn posture: turned towards the shelf with the wrist camera looking down at it."""
REST_JOINT_POSE = (0.6, 0.2, 0.0, -2.0, 0.0, 2.2, -2.356)
"""Joint 7 is turned half a turn from Arena's ready posture so the wrist camera trails on the robot side."""


class FetchBehindLogic(SpatialMemoryTaskLogic):
    """Logic of task 2 (see module docstring)."""

    name = "spatial_memory_fetch_behind"
    difficulty_name = "num_mugs"
    default_difficulty = 5
    episode_length_s = 40.0
    task_description = "Look at the shelf, turn to the counter, then fetch the mug of the colour you are given."
    forbids_searching_again = True
    reobservation_budget_s = 4.0

    @property
    def num_mugs(self) -> int:
        num = int(round(self.difficulty))
        assert 2 <= num <= len(SLOT_XS) * len(TIER_YS), f"num_mugs must be between 2 and 6, got {num}"
        return num

    def mug_names(self) -> list[str]:
        return [f"mug_{index}" for index in range(self.num_mugs)]

    def build_part_specs(self) -> list[PartSpec]:
        parts = [
            counter_spec(),
            board("tier_low", TIER_SIZE_M, colour="wood"),
            board("tier_high", TIER_SIZE_M, colour="wood"),
            board("riser", (TIER_SIZE_M[0], TIER_SIZE_M[1], RISER_HEIGHT_M), colour="dark"),
        ]
        for index, name in enumerate(self.mug_names()):
            parts.append(cylinder_item(name, MUG_DIAMETER_M, MUG_HEIGHT_M, colour=ITEM_COLOURS[index], mass_kg=0.12))
        return parts

    def sample_layout(self, generator: torch.Generator) -> LayoutSample:
        table = self.new_pose_table()
        low_z = uniform(generator, -0.02, 0.04)  # top face of the low tier
        high_z = low_z + uniform(generator, 0.10, 0.15)
        table["counter"] = counter_pose(center_x=0.60, center_y=-0.30)
        table["tier_low"] = pose(SHELF_CENTER_X, TIER_YS[0], low_z - 0.5 * TIER_SIZE_M[2])
        table["tier_high"] = pose(SHELF_CENTER_X, TIER_YS[1], high_z - 0.5 * TIER_SIZE_M[2])
        table["riser"] = pose(SHELF_CENTER_X, TIER_YS[1], high_z - TIER_SIZE_M[2] - 0.5 * RISER_HEIGHT_M)
        slots = [(x, TIER_YS[tier], (low_z, high_z)[tier]) for tier in range(2) for x in SLOT_XS]
        order = permutation(generator, len(slots))
        mug_xyz = []
        for index, name in enumerate(self.mug_names()):
            slot_x, slot_y, slot_z = slots[order[index]]
            x = slot_x + uniform(generator, -0.01, 0.01)
            y = slot_y + uniform(generator, -0.01, 0.01)
            table[name] = pose(x, y, slot_z + 0.5 * MUG_HEIGHT_M + 0.001)
            mug_xyz.append([x, y, slot_z])
        target = int(torch.randint(self.num_mugs, (1,), generator=generator))
        turn_xy = [uniform(generator, 0.42, 0.50), uniform(generator, -0.30, -0.20)]
        params = {
            "target_index": torch.tensor(target),
            "mug_xyz": torch.tensor(mug_xyz),
            "tier_z": torch.tensor([low_z, high_z]),
            "turn_xy": torch.tensor(turn_xy),
        }
        return LayoutSample(part_poses_e=self.pose_table_to_tensor(table), params=params)

    # ----------------------------------------------------------------------------------------------------------------

    def visibility_targets(self) -> list[VisibilityTarget]:
        return [VisibilityTarget(name, name, ALL_FACES, spacing_m=0.02) for name in self.mug_names()]

    def hidden_target_names(self) -> list[str]:
        return self.mug_names()

    def memory_target_names(self) -> list[str]:
        return self.mug_names()

    def item_support_names(self) -> list[str]:
        return ["tier_low", "tier_high"]

    def target_mask(self, runtime) -> torch.Tensor:
        """Return a (N, M) one-hot mask of the instructed mug."""
        return torch.nn.functional.one_hot(runtime.params["target_index"].long(), self.num_mugs).bool()

    def _target_position_e(self, env, runtime) -> torch.Tensor:
        poses = torch.stack([item_pose_e(env, name)[:, :3] for name in self.mug_names()], dim=1)
        return poses[torch.arange(env.num_envs, device=env.device), runtime.params["target_index"].long()]

    def action_phase(self, env, runtime) -> torch.Tensor:
        before_shelf = tool_position_e(env, self)[:, 1] < REACH_START_END_Y
        return runtime.revealed & (runtime.first_close_step == NOT_SET) & before_shelf

    def static_part_mask(self, env, runtime) -> torch.Tensor:
        mask = torch.ones(env.num_envs, len(runtime.part_names), dtype=torch.bool, device=env.device)
        mug_ids = runtime.part_ids(self.mug_names())
        mask[:, mug_ids] = ~self.target_mask(runtime)
        return mask

    def allowed_contacts(self, env, runtime) -> torch.Tensor:
        """The robot may touch only the instructed mug; mugs may touch the tiers and each other."""
        from isaaclab_arena_environments.spatial_memory.collision import filtered_sensor_labels

        labels = filtered_sensor_labels(self)
        num_parts = len(runtime.part_names)
        allowed = torch.zeros(env.num_envs, len(labels), num_parts, dtype=torch.bool, device=env.device)
        mug_ids = runtime.part_ids(self.mug_names())
        support_ids = runtime.part_ids(self.item_support_names())
        target_mask = self.target_mask(runtime)
        for sensor_index, label in enumerate(labels):
            if label in self.embodiment.arm_links or label in self.embodiment.gripper_bodies:
                allowed[:, sensor_index, mug_ids] = target_mask
            else:
                allowed[:, sensor_index, mug_ids] = True
                allowed[:, sensor_index, support_ids] = True
        return allowed

    def instruction_dim(self) -> int:
        return self.num_mugs

    def instruction(self, env, runtime) -> torch.Tensor:
        one_hot = self.target_mask(runtime).float()
        return one_hot * runtime.revealed.unsqueeze(-1).float()

    def on_reset(self, env, runtime, env_ids: torch.Tensor) -> None:
        if "first_close_xy" not in runtime.task_state:
            runtime.task_state["first_close_xy"] = torch.zeros(env.num_envs, 2, device=env.device)
        runtime.task_state["first_close_xy"][env_ids] = 0.0

    def on_step(self, env, runtime, visible: torch.Tensor, active: torch.Tensor) -> None:
        super().on_step(env, runtime, visible, active)
        just_closed = runtime.first_close_step == env.episode_length_buf
        if just_closed.any():
            runtime.task_state["first_close_xy"][just_closed] = tool_position_e(env, self)[just_closed, :2]

    def success_terms(self, env, runtime) -> dict[str, torch.Tensor]:
        env_index = torch.arange(env.num_envs, device=env.device)
        target_start = runtime.params["mug_xyz"][env_index, runtime.params["target_index"].long()]
        target_now = self._target_position_e(env, runtime)
        lifted = target_now[:, 2] - 0.5 * MUG_HEIGHT_M > target_start[:, 2] + 0.05
        first_close_xy = runtime.task_state["first_close_xy"]
        first_reach = (runtime.first_close_step != NOT_SET) & (
            (first_close_xy - target_start[:, :2]).norm(dim=-1) < FIRST_REACH_TOLERANCE_M
        )
        others = []
        for index, name in enumerate(self.mug_names()):
            moved = (item_pose_e(env, name)[:, :2] - runtime.params["mug_xyz"][:, index, :2]).norm(dim=-1) > 0.02
            others.append(moved & ~self.target_mask(runtime)[:, index])
        others_disturbed = torch.stack(others, dim=-1).any(dim=-1)
        return {
            "instruction_given": runtime.revealed,
            "target_lifted": lifted & self.gripper_closed(env),
            "grasped_on_first_reach": first_reach,
            "other_mugs_undisturbed": ~others_disturbed,
        }

    def extra_failures(self, env, runtime) -> torch.Tensor:
        # Closing the gripper anywhere but at the instructed mug ends the first reach unsuccessfully.
        first_close_xy = runtime.task_state["first_close_xy"]
        env_index = torch.arange(env.num_envs, device=env.device)
        target_start = runtime.params["mug_xyz"][env_index, runtime.params["target_index"].long()]
        closed_elsewhere = (runtime.first_close_step != NOT_SET) & (
            (first_close_xy - target_start[:, :2]).norm(dim=-1) > FIRST_REACH_TOLERANCE_M
        )
        return closed_elsewhere | (self._target_position_e(env, runtime)[:, 2] < -0.3)

    # ----------------------------------------------------------------------------------------------------------------

    def plan(self, env, runtime, env_id: int, mode: str) -> list[Waypoint]:
        params = {key: value[env_id] for key, value in runtime.params.items()}
        turn_x, turn_y = (float(v) for v in params["turn_xy"])
        low_z = float(params["tier_z"][0])
        high_z = float(params["tier_z"][1])
        if mode == PlanMode.ORACLE:
            target_x, target_y, target_z = (float(v) for v in params["mug_xyz"][int(params["target_index"])])
        else:
            # Ignore the remembered layout: reach for a middle slot of the low tier.
            target_x, target_y, target_z = SLOT_XS[1], TIER_YS[0], float(params["tier_z"][0])

        # The wrist camera points back towards the robot base at every waypoint. The shelf is radially outward, so
        # the camera trails it, and this orientation family is the one the arm keeps naturally as it turns.
        def camera_to_base(x: float, y: float) -> tuple[float, float, float, float]:
            return top_down_quat(math.atan2(-y, -x))

        # The return stays low, so the camera's footprint stops short of the low tier; the hand rises over the
        # high tier only at the shelf edge, after the reach-start phase.
        return_z = low_z + MUG_HEIGHT_M + 0.10
        # A low-tier target is reached at the return height; for a high-tier target the hand climbs diagonally on its
        # way over the low tier, which clears the riser.
        over_z = return_z if target_y < TIER_YS[1] - 0.05 else max(return_z, high_z + 0.13)
        grasp_z = target_z + MUG_HEIGHT_M - 0.02
        approach_y = SHELF_EDGE_Y
        at_counter = camera_to_base(turn_x, turn_y)
        at_edge = camera_to_base(target_x, approach_y)
        at_target = camera_to_base(target_x, target_y)
        look = camera_to_base(LOOK_POSITION[0], LOOK_POSITION[1])
        return [
            Waypoint(LOOK_POSITION, look, label="look", max_speed_mps=0.3, tolerance_m=0.03, hold_steps=10),
            Waypoint(
                (turn_x, turn_y, COUNTER_WAIT_Z),
                at_counter,
                label="turn_to_counter",
                max_speed_mps=0.35,
                tolerance_m=0.03,
            ),
            Waypoint(
                (turn_x, turn_y, COUNTER_WAIT_Z),
                at_counter,
                label="wait_for_instruction",
                gate="revealed",
                tolerance_m=0.03,
            ),
            Waypoint((target_x, approach_y, return_z), at_edge, label="return", max_speed_mps=0.3, tolerance_m=0.02),
            Waypoint((target_x, target_y, over_z), at_target, label="over_target", max_speed_mps=0.5),
            Waypoint((target_x, target_y, grasp_z), at_target, label="descend", max_speed_mps=0.25, tolerance_m=0.006),
            Waypoint((target_x, target_y, grasp_z), at_target, GRIPPER_CLOSE, hold_steps=12, label="close"),
            Waypoint((target_x, target_y, grasp_z + 0.12), at_target, GRIPPER_CLOSE, label="lift", max_speed_mps=0.15),
            Waypoint((target_x, target_y, grasp_z + 0.12), at_target, GRIPPER_CLOSE, label="hold", max_steps=10_000),
        ]
