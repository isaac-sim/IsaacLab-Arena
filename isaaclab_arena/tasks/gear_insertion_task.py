# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task definition for gear insertion."""

from __future__ import annotations

import math
from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectBase
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.predicates.spatial import (
    depth_in_range,
    lateral_in_proximity,
    tilt_axis_aligned,
    velocity_below_threshold,
)
from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.tasks.task_transition import Relocate, TaskTransition
from isaaclab_arena.tasks.terminations import check_success


@dataclass(frozen=True)
class GearInsertionCriteria:
    """Thresholds that define a stable gear insertion."""

    gear_insertion_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Insertion point in the gear's local frame."""

    xy_threshold: float = 0.015
    """Maximum lateral insertion-point error in meters."""

    z_threshold: float = 0.010
    """Maximum absolute insertion-depth error in meters."""

    upright_axis_threshold_deg: float = 15.0
    """Maximum angle between the gear and target up axes in degrees."""

    linear_velocity_threshold: float = 0.05
    """Maximum gear linear speed in meters per second."""

    angular_velocity_threshold: float = 0.5
    """Maximum gear angular speed in radians per second."""

    support_z_threshold: float = 0.005
    """Maximum distance that the gear may remain above the target frame."""

    consecutive_success_steps: int = 10
    """Control steps for which the gear must remain seated and settled."""

    def __post_init__(self) -> None:
        assert len(self.gear_insertion_offset_xyz) == 3, "gear_insertion_offset_xyz must contain three values"
        assert self.xy_threshold >= 0.0, "xy_threshold must be non-negative"
        assert self.z_threshold >= 0.0, "z_threshold must be non-negative"
        assert 0.0 <= self.upright_axis_threshold_deg < 90.0, "upright_axis_threshold_deg must be in [0, 90)"
        assert self.linear_velocity_threshold >= 0.0, "linear_velocity_threshold must be non-negative"
        assert self.angular_velocity_threshold >= 0.0, "angular_velocity_threshold must be non-negative"
        assert self.support_z_threshold >= 0.0, "support_z_threshold must be non-negative"
        assert self.consecutive_success_steps > 0, "consecutive_success_steps must be positive"


@register_task
class GearInsertionTask(TaskBase):
    """Pick up a gear and insert it onto a designated peg.

    Args:
        fixed_asset: Gear base containing the destination peg.
        held_asset: Gear manipulated by the robot.
        insertion_target: Scene frame defining the successful seated gear pose.
        background_scene: Background whose minimum height defines a dropped gear.
        episode_length_s: Maximum episode duration in seconds.
        success_criteria: Geometric and motion thresholds for insertion.
        task_description: Natural-language instruction. A description is generated when omitted.
    """

    def __init__(
        self,
        fixed_asset: Object,
        held_asset: Object,
        insertion_target: ObjectBase,
        background_scene: Asset,
        episode_length_s: float | None = None,
        success_criteria: GearInsertionCriteria | None = None,
        task_description: str | None = None,
    ) -> None:
        super().__init__(episode_length_s=episode_length_s)
        self.fixed_asset = fixed_asset
        self.held_asset = held_asset
        self.insertion_target = insertion_target
        self.background_scene = background_scene
        self.success_criteria = success_criteria or GearInsertionCriteria()

        self.events_cfg = GearInsertionEventsCfg()
        self.task_description = task_description or (
            f"Pick up the {held_asset.name} and insert it onto the matching peg in the {fixed_asset.name}."
        )

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return self._make_termination_cfg()

    def get_events_cfg(self):
        return self.events_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric(), ObjectMovedRateMetric(self.held_asset)]

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(1.45, 1.10, 0.90), lookat=(0.48, 0.02, 0.08))

    def _make_termination_cfg(self) -> TaskTerminationCfg:
        criteria = self.success_criteria
        mating_params = {
            "subject_name": self.held_asset.name,
            "receiver_name": self.insertion_target.name,
            "target_offset_xyz": (0.0, 0.0, 0.0),
            "subject_offset_xyz": criteria.gear_insertion_offset_xyz,
        }
        predicates = [
            TerminationTermCfg(
                func=lateral_in_proximity,
                params={**mating_params, "tolerance_lateral": criteria.xy_threshold},
            ),
            TerminationTermCfg(
                func=depth_in_range,
                params={
                    **mating_params,
                    "depth_min": -criteria.z_threshold,
                    "depth_max": min(criteria.z_threshold, criteria.support_z_threshold),
                },
            ),
            TerminationTermCfg(
                func=tilt_axis_aligned,
                params={
                    "subject_name": self.held_asset.name,
                    "receiver_name": self.insertion_target.name,
                    "max_tilt_rad": math.radians(criteria.upright_axis_threshold_deg),
                },
            ),
            TerminationTermCfg(
                func=velocity_below_threshold,
                params={
                    "subject_name": self.held_asset.name,
                    "linear_velocity_threshold": criteria.linear_velocity_threshold,
                    "angular_velocity_threshold": criteria.angular_velocity_threshold,
                },
            ),
        ]
        success = TerminationTermCfg(
            func=check_success,
            params={"predicates": predicates},
        )
        gear_dropped = TerminationTermCfg(
            func=mdp.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.held_asset.name),
            },
        )
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="insert_gear",
                    predicate_sequence=[
                        TrueForConsecutiveStepsCfg(
                            predicate=success,
                            required_steps=criteria.consecutive_success_steps,
                        )
                    ],
                )
            ],
            failures={"gear_dropped": gear_dropped},
        )

    @classmethod
    def success_state_transition(cls, held_asset: str, fixed_asset: str, **_) -> TaskTransition:
        """Relate the inserted gear to its base after success."""
        return TaskTransition(
            subject=held_asset,
            effects=(Relocate(subject=held_asset, relation="on", target=fixed_asset),),
        )


@configclass
class GearInsertionEventsCfg:
    """Reset terms for gear insertion."""

    reset_scene: EventTermCfg = EventTermCfg(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
