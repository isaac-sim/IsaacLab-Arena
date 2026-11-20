# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fetch the mug from the shelf behind you.

N mugs of different colors stand on a two-tier open shelf beside the robot (the stand-in for a cabinet: an open shelf
lets the Franka grasp mugs from above). The episode starts with the wrist camera looking at the shelf. The robot turns
to the counter in front; once the shelf has been out of view for 3.5 s, longer than the 3 s frame history, the
instruction (a one-hot color) is revealed. The robot must then reach back and grasp that mug on its first reach.

Phases: look (shelf in view), turn (shelf out of view), instruction, reach (hidden phase: from the instruction until
the hand is 0.36 m from the low tier), grasp and lift.
Randomized: which mug stands in which shelf slot, tier heights, the robot's turn target on the counter.
Difficulty knob: number of mugs N (2 to 6; the shelf has six slots the scripted Franka reaches reliably).
Success: the instructed mug is lifted, the first gripper close was on it, no other mug was touched, and the mugs were
not re-observed for more than 5 s between the instruction and the first close (else "searched again", a failure).
"""

from __future__ import annotations

import torch

from isaaclab_arena_environments.spatial_memory.episode_state import NOT_SET
from isaaclab_arena_environments.spatial_memory.franka_controller import GRIPPER_CLOSE, HOLD_FOREVER_STEPS, Waypoint
from isaaclab_arena_environments.spatial_memory.geometry import ALL_FACES
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
    COUNTER_WAIT_Z_M,
    LOOK_JOINT_POSE,
    LOOK_POSITION_E,
    RACK_EDGE_Y_M,
    REST_JOINT_POSE,
    SLOT_XS_M,
    TIER_YS_M,
    place_rack,
    rack_part_specs,
    rack_slots,
    top_down_quat_camera_towards_base,
)
from isaaclab_arena_environments.spatial_memory.visibility import VisibilityTarget

MUG_DIAMETER_M = 0.05
"""Diameter of each mug."""
MUG_HEIGHT_M = 0.08
"""Height of each mug."""
HIDDEN_PHASE_END_Y_M = 0.12
"""Environment-frame y of the tool center point at which the hidden phase ends, 0.36 m from the low tier."""
FIRST_REACH_TOLERANCE_M = 0.04
"""Maximum horizontal distance between the first gripper close and the instructed mug's start position."""
LIFTED_M = 0.05
"""Height of the instructed mug's base above its tier above which it counts as lifted."""
MUG_MAX_SHIFT_M = 0.02
"""Horizontal displacement of a mug above which it counts as disturbed."""
DROPPED_Z_M = -0.3
"""Environment-frame height below which the instructed mug counts as dropped."""


class FetchBehindLogic(SpatialMemoryTaskLogic):
    """Logic of the fetch-behind task (see the module docstring)."""

    name = "spatial_memory_fetch_behind"
    rest_joint_pose = REST_JOINT_POSE
    initial_joint_pose = LOOK_JOINT_POSE
    difficulty_name = "num_mugs"
    default_difficulty = 5
    episode_length_s = 40.0
    task_description = "Look at the shelf, turn to the counter, then fetch the mug of the color you are given."
    forbids_searching_again = True
    reobservation_budget_s = 5.0

    @property
    def num_mugs(self) -> int:
        """Return the number of mugs on the shelf."""
        num = int(round(self.difficulty))
        assert 2 <= num <= len(SLOT_XS_M) * len(TIER_YS_M), f"num_mugs must be between 2 and 6, got {num}"
        return num

    def mug_names(self) -> list[str]:
        """Return the names of the mugs."""
        return [f"mug_{index}" for index in range(self.num_mugs)]

    def build_part_specs(self) -> list[PartSpec]:
        parts = [counter_spec(), *rack_part_specs()]
        for index, name in enumerate(self.mug_names()):
            parts.append(cylinder_item(name, MUG_DIAMETER_M, MUG_HEIGHT_M, color=ITEM_COLORS[index], mass_kg=0.12))
        return parts

    def sample_layout(self, generator: torch.Generator) -> LayoutSample:
        poses = self.parked_part_poses()
        low_z = uniform(generator, -0.02, 0.04)  # top face of the low tier
        high_z = low_z + uniform(generator, 0.10, 0.15)
        poses["counter"] = counter_pose(center_x=0.60, center_y=-0.30)
        place_rack(poses, low_z, high_z)
        slots = rack_slots(low_z, high_z)
        order = permutation(generator, len(slots))
        mug_xyz = []
        for index, name in enumerate(self.mug_names()):
            slot_x, slot_y, slot_z = slots[order[index]]
            x = slot_x + uniform(generator, -0.01, 0.01)
            y = slot_y + uniform(generator, -0.01, 0.01)
            poses[name] = pose(x, y, slot_z + 0.5 * MUG_HEIGHT_M + 0.001)
            mug_xyz.append([x, y, slot_z])
        target = int(torch.randint(self.num_mugs, (1,), generator=generator))
        turn_xy = [uniform(generator, 0.42, 0.50), uniform(generator, -0.30, -0.20)]
        params = {
            "target_index": torch.tensor(target),
            "mug_xyz": torch.tensor(mug_xyz),
            "tier_z": torch.tensor([low_z, high_z]),
            "turn_xy": torch.tensor(turn_xy),
        }
        return LayoutSample(part_poses_e=self.part_poses_tensor(poses), layout_params=params)

    # ----------------------------------------------------------------------------------------------------------------

    def visibility_targets(self) -> list[VisibilityTarget]:
        return [VisibilityTarget(name, name, ALL_FACES, spacing_m=0.02) for name in self.mug_names()]

    def hidden_target_names(self) -> list[str]:
        return self.mug_names()

    def memory_target_names(self) -> list[str]:
        return self.mug_names()

    def item_support_names(self) -> list[str]:
        return ["tier_low", "tier_high"]

    def instructed_mug_mask(self, episode) -> torch.Tensor:
        """Return a one-hot (N, M) mask of the instructed mug."""
        return torch.nn.functional.one_hot(episode.layout_params["target_index"].long(), self.num_mugs).bool()

    def _instructed_mug_position_e(self, env, episode) -> torch.Tensor:
        """Return the environment-frame position (N, 3) of the instructed mug."""
        t_E_mugs = torch.stack([env.arena_world.get_pose_e(name)[:, :3] for name in self.mug_names()], dim=1)
        return t_E_mugs[torch.arange(env.num_envs, device=env.device), episode.layout_params["target_index"].long()]

    def hidden_phase(self, env, episode) -> torch.Tensor:
        before_shelf = tool_position_e(env, self)[:, 1] < HIDDEN_PHASE_END_Y_M
        return episode.instruction_revealed & (episode.first_close_step == NOT_SET) & before_shelf

    def static_part_mask(self, env, episode) -> torch.Tensor:
        mask = torch.ones(env.num_envs, len(episode.part_names), dtype=torch.bool, device=env.device)
        mug_ids = episode.part_ids(self.mug_names())
        mask[:, mug_ids] = ~self.instructed_mug_mask(episode)
        return mask

    def robot_touchable_part_mask(self, env, episode) -> torch.Tensor:
        """Return the instructed mug: the robot may touch no other part."""
        mask = torch.zeros(env.num_envs, len(episode.part_names), dtype=torch.bool, device=env.device)
        mask[:, episode.part_ids(self.mug_names())] = self.instructed_mug_mask(episode)
        return mask

    def instruction_dim(self) -> int:
        return self.num_mugs

    def instruction(self, env, episode) -> torch.Tensor:
        one_hot = self.instructed_mug_mask(episode).float()
        return one_hot * episode.instruction_revealed.unsqueeze(-1).float()

    def on_reset(self, env, episode, env_ids: torch.Tensor) -> None:
        episode.task_state_buffer("first_close_xy", 0.0, torch.float32, (2,))

    def on_step(self, env, episode, visible: torch.Tensor, active: torch.Tensor) -> None:
        super().on_step(env, episode, visible, active)
        just_closed = episode.first_close_step == env.episode_length_buf
        if just_closed.any():
            episode.task_state["first_close_xy"][just_closed] = tool_position_e(env, self)[just_closed, :2]

    def success_terms(self, env, episode) -> dict[str, torch.Tensor]:
        env_index = torch.arange(env.num_envs, device=env.device)
        t_E_target_start = episode.layout_params["mug_xyz"][env_index, episode.layout_params["target_index"].long()]
        t_E_target = self._instructed_mug_position_e(env, episode)
        lifted = t_E_target[:, 2] - 0.5 * MUG_HEIGHT_M > t_E_target_start[:, 2] + LIFTED_M
        t_E_first_close_xy = episode.task_state["first_close_xy"]
        first_reach = (episode.first_close_step != NOT_SET) & (
            (t_E_first_close_xy - t_E_target_start[:, :2]).norm(dim=-1) < FIRST_REACH_TOLERANCE_M
        )
        others = []
        for index, name in enumerate(self.mug_names()):
            t_E_mug_xy = env.arena_world.get_pose_e(name)[:, :2]
            moved = (t_E_mug_xy - episode.layout_params["mug_xyz"][:, index, :2]).norm(dim=-1) > MUG_MAX_SHIFT_M
            others.append(moved & ~self.instructed_mug_mask(episode)[:, index])
        others_disturbed = torch.stack(others, dim=-1).any(dim=-1)
        return {
            "instruction_revealed": episode.instruction_revealed,
            "target_lifted": lifted & self.gripper_closed(env),
            "grasped_on_first_reach": first_reach,
            "other_mugs_undisturbed": ~others_disturbed,
        }

    def extra_failures(self, env, episode) -> torch.Tensor:
        # Closing the gripper anywhere but at the instructed mug ends the first reach unsuccessfully.
        t_E_first_close_xy = episode.task_state["first_close_xy"]
        env_index = torch.arange(env.num_envs, device=env.device)
        t_E_target_start = episode.layout_params["mug_xyz"][env_index, episode.layout_params["target_index"].long()]
        closed_elsewhere = (episode.first_close_step != NOT_SET) & (
            (t_E_first_close_xy - t_E_target_start[:, :2]).norm(dim=-1) > FIRST_REACH_TOLERANCE_M
        )
        return closed_elsewhere | (self._instructed_mug_position_e(env, episode)[:, 2] < DROPPED_Z_M)

    # ----------------------------------------------------------------------------------------------------------------

    def plan(self, env, episode, env_id: int, mode: str) -> list[Waypoint]:
        params = {key: value[env_id] for key, value in episode.layout_params.items()}
        turn_x, turn_y = (float(v) for v in params["turn_xy"])
        low_z = float(params["tier_z"][0])
        high_z = float(params["tier_z"][1])
        if mode == PlanMode.ORACLE:
            target_x, target_y, target_z = (float(v) for v in params["mug_xyz"][int(params["target_index"])])
        else:
            # Ignore the remembered layout: reach for a middle slot of the low tier.
            target_x, target_y, target_z = SLOT_XS_M[1], TIER_YS_M[0], float(params["tier_z"][0])
        camera_to_base = top_down_quat_camera_towards_base
        # The return stays low, so the camera's footprint stops short of the low tier; the hand rises over the
        # high tier only at the shelf edge, after the hidden phase.
        return_z = low_z + MUG_HEIGHT_M + 0.10
        # A low-tier target is reached at the return height; for a high-tier target the hand climbs diagonally on its
        # way over the low tier, which clears the riser.
        over_z = return_z if target_y < TIER_YS_M[1] - 0.05 else max(return_z, high_z + 0.13)
        grasp_z = target_z + MUG_HEIGHT_M - 0.02
        approach_y = RACK_EDGE_Y_M
        at_counter = camera_to_base(turn_x, turn_y)
        at_edge = camera_to_base(target_x, approach_y)
        at_target = camera_to_base(target_x, target_y)
        look = camera_to_base(LOOK_POSITION_E[0], LOOK_POSITION_E[1])
        return [
            Waypoint(LOOK_POSITION_E, look, label="look", max_speed_mps=0.3, tolerance_m=0.03, hold_steps=10),
            Waypoint(
                (turn_x, turn_y, COUNTER_WAIT_Z_M),
                at_counter,
                label="turn_to_counter",
                max_speed_mps=0.35,
                tolerance_m=0.03,
            ),
            Waypoint(
                (turn_x, turn_y, COUNTER_WAIT_Z_M),
                at_counter,
                label="wait_for_instruction",
                gate="instruction_revealed",
                tolerance_m=0.03,
            ),
            Waypoint((target_x, approach_y, return_z), at_edge, label="return", max_speed_mps=0.3, tolerance_m=0.02),
            Waypoint((target_x, target_y, over_z), at_target, label="over_target", max_speed_mps=0.5),
            Waypoint((target_x, target_y, grasp_z), at_target, label="descend", max_speed_mps=0.25, tolerance_m=0.006),
            Waypoint((target_x, target_y, grasp_z), at_target, GRIPPER_CLOSE, hold_steps=12, label="close"),
            Waypoint((target_x, target_y, grasp_z + 0.12), at_target, GRIPPER_CLOSE, label="lift", max_speed_mps=0.15),
            Waypoint(
                (target_x, target_y, grasp_z + 0.12),
                at_target,
                GRIPPER_CLOSE,
                label="hold",
                max_steps=HOLD_FOREVER_STEPS,
            ),
        ]
