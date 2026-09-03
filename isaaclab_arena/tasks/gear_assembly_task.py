# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task definition for picking up and inserting a gear."""

from __future__ import annotations

from dataclasses import MISSING, dataclass

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
from isaaclab_arena.tasks.predicates.gear_insertion import gear_is_inserted
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_transition import Relocate, TaskTransition


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

    def __post_init__(self) -> None:
        assert len(self.gear_insertion_offset_xyz) == 3, "gear_insertion_offset_xyz must contain three values"
        assert self.xy_threshold >= 0.0, "xy_threshold must be non-negative"
        assert self.z_threshold >= 0.0, "z_threshold must be non-negative"
        assert 0.0 <= self.upright_axis_threshold_deg < 90.0, "upright_axis_threshold_deg must be in [0, 90)"
        assert self.linear_velocity_threshold >= 0.0, "linear_velocity_threshold must be non-negative"
        assert self.angular_velocity_threshold >= 0.0, "angular_velocity_threshold must be non-negative"
        assert self.support_z_threshold >= 0.0, "support_z_threshold must be non-negative"


@register_task
class GearAssemblyTask(TaskBase):
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

        # A single full-scene reset owns all state restoration for this task.
        self.fixed_asset.disable_reset_pose()
        self.held_asset.disable_reset_pose()
        self.events_cfg = GearAssemblyEventsCfg()
        self.termination_cfg = self._make_termination_cfg()
        self.task_description = task_description or (
            f"Pick up the {held_asset.name} and insert it onto the matching peg in the {fixed_asset.name}."
        )

    def apply_reachability_constraints(self) -> None:
        """The robot must reach both the loose gear and its fixed assembly base."""
        self._apply_reachability_constraints([self.held_asset, self.fixed_asset])

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_events_cfg(self):
        return self.events_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric(), ObjectMovedRateMetric(self.held_asset)]

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(1.45, 1.10, 0.90), lookat=(0.48, 0.02, 0.08))

    def _make_termination_cfg(self) -> GearAssemblyTerminationsCfg:
        criteria = self.success_criteria
        success = TerminationTermCfg(
            func=gear_is_inserted,
            params={
                "gear_cfg": SceneEntityCfg(self.held_asset.name),
                "insertion_target": self.insertion_target,
                "gear_insertion_offset_xyz": criteria.gear_insertion_offset_xyz,
                "xy_threshold": criteria.xy_threshold,
                "z_threshold": criteria.z_threshold,
                "upright_axis_threshold_deg": criteria.upright_axis_threshold_deg,
                "linear_velocity_threshold": criteria.linear_velocity_threshold,
                "angular_velocity_threshold": criteria.angular_velocity_threshold,
                "support_z_threshold": criteria.support_z_threshold,
            },
        )
        gear_dropped = TerminationTermCfg(
            func=mdp.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.held_asset.name),
            },
        )
        return GearAssemblyTerminationsCfg(success=success, gear_dropped=gear_dropped)

    @classmethod
    def success_state_transition(cls, held_asset: str, fixed_asset: str, **_) -> TaskTransition:
        """Relate the inserted gear to its base after success."""
        return TaskTransition(
            subject=held_asset,
            effects=(Relocate(subject=held_asset, relation="on", target=fixed_asset),),
        )


@configclass
class GearAssemblyEventsCfg:
    """Reset terms for gear assembly."""

    reset_scene: EventTermCfg = EventTermCfg(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


@configclass
class GearAssemblyTerminationsCfg:
    """Termination terms for gear assembly."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.time_out, time_out=True)
    success: TerminationTermCfg = MISSING
    gear_dropped: TerminationTermCfg = MISSING
