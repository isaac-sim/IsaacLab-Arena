# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Remove the free dot-marked blocks of a fused block tower onto a side pad, leaving the tower intact."""

from __future__ import annotations

import math
import numpy as np
import torch
from functools import partial
from typing import TYPE_CHECKING

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.tasks.task_transition import TaskTransition
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


def _initial_poses_e(env: IsaacLabArenaManagerBasedRLEnv, block_poses_e: list[tuple]) -> torch.Tensor:
    """Return the blocks' initial poses in the environment frame as a (num_blocks, 7) tensor."""
    return torch.tensor(block_poses_e, device=env.device, dtype=torch.float32)


def blocks_on_pad(
    env: IsaacLabArenaManagerBasedRLEnv,
    block_names: list[str],
    pad_name: str,
    pad_half_size_m: float,
    pad_half_thickness_m: float,
    max_height_above_pad_m: float,
) -> torch.Tensor:
    """Return a (num_envs, num_blocks) mask of blocks whose center lies over the pad, just above its top face."""
    T_W_P = env.arena_world.get_pose_w(pad_name)
    on_pad = []
    for block_name in block_names:
        t_W_B = env.arena_world.get_pose_w(block_name)[:, :3]
        t_P_B = quat_apply_inverse(T_W_P[:, 3:], t_W_B - T_W_P[:, :3])
        height_m = t_P_B[:, 2] - pad_half_thickness_m
        on_pad.append(
            (t_P_B[:, 0].abs() < pad_half_size_m)
            & (t_P_B[:, 1].abs() < pad_half_size_m)
            & (height_m > 0.0)
            & (height_m < max_height_above_pad_m)
        )
    return torch.stack(on_pad, dim=-1)


def blocks_in_place(
    env: IsaacLabArenaManagerBasedRLEnv,
    block_names: list[str],
    block_poses_e: list[tuple],
    position_tolerance_m: float,
    rotation_tolerance_rad: float,
) -> torch.Tensor:
    """Return a (num_envs, num_blocks) mask of blocks within tolerance of their initial pose in the tower."""
    T_E_B0 = _initial_poses_e(env, block_poses_e)
    in_place = []
    for index, block_name in enumerate(block_names):
        T_E_B = env.arena_world.get_pose_e(block_name)
        position_error_m = torch.linalg.vector_norm(T_E_B[:, :3] - T_E_B0[index, :3], dim=-1)
        rotation_error_rad = quat_error_magnitude(T_E_B[:, 3:], T_E_B0[index, 3:].expand_as(T_E_B[:, 3:]))
        in_place.append((position_error_m < position_tolerance_m) & (rotation_error_rad < rotation_tolerance_rad))
    return torch.stack(in_place, dim=-1)


def blocks_slow(env: IsaacLabArenaManagerBasedRLEnv, block_names: list[str], speed_threshold_m_s: float):
    """Return a (num_envs,) mask that is true when every block moves slower than the threshold."""
    slow = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for block_name in block_names:
        speed = torch.linalg.vector_norm(env.arena_world.get_root_linear_velocity_w(block_name), dim=-1)
        slow &= speed < speed_threshold_m_s
    return slow


def candidates_removed_with_tower_intact(
    env: IsaacLabArenaManagerBasedRLEnv,
    num_removed: int,
    block_names: list[str],
    candidate_mask: list[bool],
    block_poses_e: list[tuple],
    pad_name: str,
    pad_half_size_m: float,
    pad_half_thickness_m: float,
    max_height_above_pad_m: float,
    position_tolerance_m: float,
    rotation_tolerance_rad: float,
    speed_threshold_m_s: float | None = None,
) -> torch.Tensor:
    """Check that exactly ``num_removed`` dot-marked blocks lie on the pad and every other block is in place.

    No unmarked block may lie on the pad. When a speed threshold is given, every block must also be slower than it.
    """
    on_pad = blocks_on_pad(env, block_names, pad_name, pad_half_size_m, pad_half_thickness_m, max_height_above_pad_m)
    is_candidate = torch.tensor(candidate_mask, device=env.device, dtype=torch.bool)
    in_place = blocks_in_place(env, block_names, block_poses_e, position_tolerance_m, rotation_tolerance_rad)
    num_candidates_on_pad = (on_pad & is_candidate).sum(dim=-1)
    no_unmarked_on_pad = ~(on_pad & ~is_candidate).any(dim=-1)
    rest_in_place = (on_pad | in_place).all(dim=-1)
    result = (num_candidates_on_pad == num_removed) & no_unmarked_on_pad & rest_in_place
    if speed_threshold_m_s is not None:
        result &= blocks_slow(env, block_names, speed_threshold_m_s)
    return result


def tower_damaged(
    env: IsaacLabArenaManagerBasedRLEnv,
    structural_block_names: list[str],
    structural_block_poses_e: list[tuple],
    displacement_m: float,
) -> torch.Tensor:
    """Return whether any unmarked block moved farther than ``displacement_m`` from its place in the tower."""
    in_place = blocks_in_place(
        env, structural_block_names, structural_block_poses_e, displacement_m, rotation_tolerance_rad=math.pi
    )
    return ~in_place.all(dim=-1)


@register_task
class FusedBlockTowerTask(TaskBase):
    """Remove the free dot-marked blocks of a fused block tower onto a side pad without disturbing the rest.

    Most dot-marked blocks are fused to the layer above; only a hidden free pair can come out cleanly. Success
    requires exactly ``num_blocks_to_remove`` dot-marked blocks resting on the side pad, no unmarked block on it, and
    every other block within tolerance of its initial pose, held for the required consecutive steps. Failure occurs
    when an unmarked block moves farther than ``failure_displacement_m`` from its place, for example when a fused block
    drags the block above it out of the tower.

    Args:
        tower: The fused block tower (static parent of the block references).
        blocks: Every block of the tower, as rigid object references.
        candidate_blocks: The dot-marked blocks; a subset of ``blocks``.
        side_pad: Pad the removed blocks must rest on.
        num_blocks_to_remove: Number of dot-marked blocks that must end on the pad.
        episode_length_s: Maximum episode duration in seconds.
        task_description: Natural-language task instruction.
        position_tolerance_m: Maximum distance of a remaining block from its initial position.
        rotation_tolerance_deg: Maximum rotation of a remaining block from its initial orientation, in degrees.
        failure_displacement_m: Distance an unmarked block may move before the episode fails.
        max_height_above_pad_m: Maximum height of a block's center above the pad's top face to count as on the pad.
        speed_threshold_m_s: Every block must move slower than this for the final state to count.
        hold_steps: Consecutive control steps the final state must hold.
    """

    def __init__(
        self,
        tower: Asset,
        blocks: list[Asset],
        candidate_blocks: list[Asset],
        side_pad: Asset,
        num_blocks_to_remove: int = 2,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        position_tolerance_m: float = 0.012,
        rotation_tolerance_deg: float = 10.0,
        failure_displacement_m: float = 0.04,
        max_height_above_pad_m: float = 0.04,
        speed_threshold_m_s: float = 0.02,
        hold_steps: int = 30,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        block_names = [block.name for block in blocks]
        candidate_names = [block.name for block in candidate_blocks]
        assert set(candidate_names) <= set(block_names), "candidate_blocks must be a subset of blocks."
        assert 0 < num_blocks_to_remove <= len(candidate_names), "num_blocks_to_remove must be 1..len(candidates)."
        assert hold_steps > 0, "hold_steps must be positive."
        self.tower = tower
        self.blocks = list(blocks)
        self.side_pad = side_pad
        self.block_names = block_names
        self.candidate_mask = [name in candidate_names for name in block_names]
        # Object references take their initial pose from the tower USD, so it is known at construction.
        self.block_poses_e = []
        for block in self.blocks:
            T_E_B = block.get_initial_pose()
            self.block_poses_e.append((*T_E_B.position_xyz, *T_E_B.rotation_xyzw))
        self.num_blocks_to_remove = num_blocks_to_remove
        self.position_tolerance_m = position_tolerance_m
        self.rotation_tolerance_rad = math.radians(rotation_tolerance_deg)
        self.failure_displacement_m = failure_displacement_m
        self.max_height_above_pad_m = max_height_above_pad_m
        self.speed_threshold_m_s = speed_threshold_m_s
        self.hold_steps = hold_steps
        if self.task_description is None:
            self.task_description = (
                f"Remove exactly {num_blocks_to_remove} of the dot-marked blocks from the tower and leave them on the"
                " side pad. The tower must still be standing with all its other blocks in place."
            )

    def apply_reachability_constraints(self) -> None:
        """The robot must reach the side pad; the tower blocks are object references at a fixed pose."""
        self._apply_reachability_constraints([self.side_pad])

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("FusedBlockTowerTask has no Mimic configuration.")

    def get_termination_cfg(self) -> TaskTerminationCfg:
        pad_size_m = getattr(self.side_pad, "size_m", 0.13)
        pad_thickness_m = getattr(self.side_pad, "thickness_m", 0.008)
        common = dict(
            block_names=self.block_names,
            candidate_mask=self.candidate_mask,
            block_poses_e=self.block_poses_e,
            pad_name=self.side_pad.name,
            pad_half_size_m=pad_size_m / 2,
            pad_half_thickness_m=pad_thickness_m / 2,
            max_height_above_pad_m=self.max_height_above_pad_m,
            position_tolerance_m=self.position_tolerance_m,
            rotation_tolerance_rad=self.rotation_tolerance_rad,
        )
        first_removed = partial(candidates_removed_with_tower_intact, num_removed=1, **common)
        all_removed = partial(
            candidates_removed_with_tower_intact,
            num_removed=self.num_blocks_to_remove,
            speed_threshold_m_s=self.speed_threshold_m_s,
            **common,
        )
        structural = [
            (name, pose)
            for name, pose, is_candidate in zip(self.block_names, self.block_poses_e, self.candidate_mask)
            if not is_candidate
        ]
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="remove_free_blocks",
                    predicate_sequence=[
                        (first_removed, 0.4),
                        (TrueForConsecutiveStepsCfg(predicate=all_removed, required_steps=self.hold_steps), 0.6),
                    ],
                ),
            ],
            failures={
                "tower_damaged": TerminationTermCfg(
                    func=tower_damaged,
                    params={
                        "structural_block_names": [name for name, _ in structural],
                        "structural_block_poses_e": [pose for _, pose in structural],
                        "displacement_m": self.failure_displacement_m,
                    },
                )
            },
        )

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(lookat_object=self.tower, offset=np.array([-0.45, -0.45, 0.35]))

    @classmethod
    def success_state_transition(cls, side_pad: str, **_) -> TaskTransition:
        """Success moves blocks onto the side pad; the env graph has no node state for the tower's integrity."""
        return TaskTransition(subject=side_pad)
