# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Match look-alike load blocks to spring platforms of hidden spring rate."""

from __future__ import annotations

import numpy as np
import torch
from functools import partial
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.assets.spring_platform_ballast_match_usd import (
    DEFAULT_GEOMETRY,
    NUM_PLATFORMS,
    platform_joint_name,
    platform_top_body_name,
)
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
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.embodiments.gripper import Gripper
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv

_TOP_HALF_SIZE_M = DEFAULT_GEOMETRY.top_size_m / 2
_LOAD_HEIGHT_M = 0.1
"""Height above a top surface up to which a block's center counts as resting on that platform."""


def _joint_columns(env: IsaacLabArenaManagerBasedRLEnv, rack_name: str, values: torch.Tensor) -> torch.Tensor:
    """Select the platforms' spring joints, in platform order, from a (num_envs, num_joints) tensor."""
    joint_names = env.scene.articulations[rack_name].data.joint_names
    joint_ids = [joint_names.index(platform_joint_name(index)) for index in range(NUM_PLATFORMS)]
    return values[:, joint_ids]


def platform_deflections_m(env: IsaacLabArenaManagerBasedRLEnv, rack_name: str) -> torch.Tensor:
    """Return each platform's deflection below level in meters, shape (num_envs, NUM_PLATFORMS)."""
    data = env.scene.articulations[rack_name].data
    return -_joint_columns(env, rack_name, data.joint_pos.torch)


def platform_speeds_m_s(env: IsaacLabArenaManagerBasedRLEnv, rack_name: str) -> torch.Tensor:
    """Return each platform top's vertical speed in m/s, shape (num_envs, NUM_PLATFORMS)."""
    data = env.scene.articulations[rack_name].data
    return _joint_columns(env, rack_name, data.joint_vel.torch).abs()


def _top_centers_w(env: IsaacLabArenaManagerBasedRLEnv, rack_name: str) -> torch.Tensor:
    """Return each platform's top-surface center in the world frame, shape (num_envs, NUM_PLATFORMS, 3)."""
    centers = []
    for index in range(NUM_PLATFORMS):
        T_W_top = env.arena_world.get_body_pose_w(rack_name, platform_top_body_name(index))
        centers.append(T_W_top[:, :3])
    return torch.stack(centers, dim=1)


def blocks_per_platform(env: IsaacLabArenaManagerBasedRLEnv, rack_name: str, block_names: list[str]) -> torch.Tensor:
    """Count the blocks whose center lies over each platform's top and within the load height above it.

    Returns:
        Block counts, shape (num_envs, NUM_PLATFORMS).
    """
    t_W_tops = _top_centers_w(env, rack_name)
    counts = torch.zeros(t_W_tops.shape[:2], device=t_W_tops.device, dtype=torch.long)
    for block_name in block_names:
        t_W_block = env.arena_world.get_pose_w(block_name)[:, :3]
        offset_W = t_W_block[:, None, :] - t_W_tops
        over_top = (offset_W[..., 0].abs() < _TOP_HALF_SIZE_M) & (offset_W[..., 1].abs() < _TOP_HALF_SIZE_M)
        at_height = (offset_W[..., 2] > 0.0) & (offset_W[..., 2] < _LOAD_HEIGHT_M)
        counts += (over_top & at_height).long()
    return counts


def every_platform_loaded(env: IsaacLabArenaManagerBasedRLEnv, rack_name: str, block_names: list[str]) -> torch.Tensor:
    """Check that each platform carries exactly one block."""
    return (blocks_per_platform(env, rack_name, block_names) == 1).all(dim=-1)


def every_pointer_in_band(
    env: IsaacLabArenaManagerBasedRLEnv,
    rack_name: str,
    block_names: list[str],
    target_deflection_m: float,
    band_half_width_m: float,
    platform_speed_threshold_m_s: float,
    block_speed_threshold_m_s: float,
    gripper: Gripper | None,
    gripper_clearance_m: float,
) -> torch.Tensor:
    """Check that every platform carries one block, deflects into its band and is still, with the gripper away.

    The band is target_deflection_m plus or minus band_half_width_m. Every platform top and every block must move
    slower than its speed threshold, and the gripper, when given, must be farther than the clearance from each top.
    """
    loaded = every_platform_loaded(env, rack_name, block_names)
    deflection_error_m = (platform_deflections_m(env, rack_name) - target_deflection_m).abs()
    in_band = (deflection_error_m <= band_half_width_m).all(dim=-1)
    platforms_still = (platform_speeds_m_s(env, rack_name) < platform_speed_threshold_m_s).all(dim=-1)
    blocks_still = torch.ones_like(loaded)
    for block_name in block_names:
        speed = torch.linalg.vector_norm(env.arena_world.get_root_linear_velocity_w(block_name), dim=-1)
        blocks_still &= speed < block_speed_threshold_m_s
    result = loaded & in_band & platforms_still & blocks_still
    if gripper is not None:
        t_W_gripper = gripper.get_position_w(env.arena_world)
        distances_m = torch.linalg.vector_norm(t_W_gripper[:, None, :] - _top_centers_w(env, rack_name), dim=-1)
        result &= (distances_m > gripper_clearance_m).all(dim=-1)
    return result


@register_task
class SpringPlatformBallastTask(TaskBase):
    """Put one look-alike load block on each of three look-alike spring platforms so every pointer reads in band.

    Each platform's spring rate and each block's mass are hidden; only the matching load deflects a platform into its
    green band. Success requires every platform to carry exactly one block, every deflection within the band, the
    platforms and blocks still and the gripper clear of the platforms, for the required consecutive steps. Failure
    occurs when a block falls below the minimum height, for example off the table.

    Args:
        spring_rack: The spring-platform rack articulation.
        load_blocks: The load blocks, one per platform.
        episode_length_s: Maximum episode duration in seconds.
        task_description: Natural-language task instruction.
        target_deflection_m: Center of the green band, as a deflection below level in meters.
        band_half_width_m: Half height of the green band in meters.
        platform_speed_threshold_m_s: Every platform top must move slower than this, in m/s.
        block_speed_threshold_m_s: Every block must move slower than this, in m/s.
        gripper_clearance_m: Minimum gripper distance from each top's center, in meters.
        hold_steps: Consecutive control steps the in-band state must hold.
        block_min_height_m: A block below this world height counts as dropped.
    """

    def __init__(
        self,
        spring_rack: Asset,
        load_blocks: list[Asset],
        episode_length_s: float | None = None,
        task_description: str | None = None,
        target_deflection_m: float = DEFAULT_GEOMETRY.target_deflection_m,
        band_half_width_m: float = DEFAULT_GEOMETRY.band_half_width_m,
        platform_speed_threshold_m_s: float = 0.005,
        block_speed_threshold_m_s: float = 0.01,
        gripper_clearance_m: float = 0.08,
        hold_steps: int = 30,
        block_min_height_m: float = -0.05,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        assert len(load_blocks) == NUM_PLATFORMS, f"SpringPlatformBallastTask needs {NUM_PLATFORMS} load blocks."
        assert band_half_width_m > 0.0, "band_half_width_m must be positive."
        assert hold_steps > 0, "hold_steps must be positive."
        self.spring_rack = spring_rack
        self.load_blocks = list(load_blocks)
        self.target_deflection_m = target_deflection_m
        self.band_half_width_m = band_half_width_m
        self.platform_speed_threshold_m_s = platform_speed_threshold_m_s
        self.block_speed_threshold_m_s = block_speed_threshold_m_s
        self.gripper_clearance_m = gripper_clearance_m
        self.hold_steps = hold_steps
        self.block_min_height_m = block_min_height_m
        self.gripper: Gripper | None = None
        if self.task_description is None:
            self.task_description = (
                "Put one load block on each of the three spring platforms so that every platform's pointer sits in"
                " its green band, and take your hands away."
            )

    def configure_for_embodiment(self, embodiment: EmbodimentBase) -> None:
        """Use the embodiment's gripper, when it has one, for the gripper-clearance check."""
        self.gripper = embodiment.gripper

    def apply_reachability_constraints(self) -> None:
        """The robot must reach every load block."""
        self._apply_reachability_constraints(self.load_blocks)

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("SpringPlatformBallastTask has no Mimic configuration.")

    def get_termination_cfg(self) -> TaskTerminationCfg:
        rack_name = self.spring_rack.name
        block_names = [block.name for block in self.load_blocks]
        # Bind the gripper lazily: configure_for_embodiment runs after the task is constructed.
        task = self

        def in_band(env: IsaacLabArenaManagerBasedRLEnv) -> torch.Tensor:
            return every_pointer_in_band(
                env,
                rack_name=rack_name,
                block_names=block_names,
                target_deflection_m=task.target_deflection_m,
                band_half_width_m=task.band_half_width_m,
                platform_speed_threshold_m_s=task.platform_speed_threshold_m_s,
                block_speed_threshold_m_s=task.block_speed_threshold_m_s,
                gripper=task.gripper,
                gripper_clearance_m=task.gripper_clearance_m,
            )

        failures = {}
        for block_name in block_names:
            failures[f"{block_name}_dropped"] = TerminationTermCfg(
                func=mdp_isaac_lab.root_height_below_minimum,
                params={"minimum_height": self.block_min_height_m, "asset_cfg": SceneEntityCfg(block_name)},
            )
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="match_loads_to_springs",
                    predicate_sequence=[
                        (partial(every_platform_loaded, rack_name=rack_name, block_names=block_names), 0.3),
                        (TrueForConsecutiveStepsCfg(predicate=in_band, required_steps=self.hold_steps), 0.7),
                    ],
                ),
            ],
            failures=failures,
        )

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(lookat_object=self.spring_rack, offset=np.array([-0.8, -0.5, 0.45]))

    @classmethod
    def success_state_transition(cls, spring_rack: str, **_) -> TaskTransition:
        """Success changes the rack's state (loaded and in band); the env graph has no node state for it yet."""
        return TaskTransition(subject=spring_rack)
