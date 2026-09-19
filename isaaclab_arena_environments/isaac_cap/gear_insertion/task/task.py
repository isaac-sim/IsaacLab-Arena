# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena task configuration for gear insertion."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.predicates.composite import CompositePredicate
from isaaclab_arena.tasks.predicates.spatial import (
    depth_in_range,
    lateral_in_proximity,
    tilt_axis_aligned,
    velocity_below_threshold,
)
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.tasks.terminations import SuccessMode

from .metrics import GearInsertionFractionMetric
from .predicates import GearIsSupported


@configclass
class EventsCfg:
    """Standard scene reset only."""

    reset_all: EventTermCfg = EventTermCfg(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


def _make_gear_success_composite_predicate_cfg(
    plate: Asset,
    gear: Asset,
    target_offset_xyz: tuple[float, float, float],
    *,
    xy_threshold: float,
    z_threshold: float,
    upright_axis_threshold_deg: float,
    linear_velocity_threshold: float,
    angular_velocity_threshold: float,
    support_z_threshold: float,
) -> TerminationTermCfg:
    """Build the current geometric gates and stateful settling gate for one gear."""
    relative_position_params = {
        "subject_name": gear.name,
        "receiver_name": plate.name,
        "target_offset_xyz": target_offset_xyz,
    }
    predicates = [
        TerminationTermCfg(
            func=lateral_in_proximity,
            params={**relative_position_params, "tolerance_lateral": xy_threshold},
        ),
        TerminationTermCfg(
            func=depth_in_range,
            params={
                **relative_position_params,
                "depth_min": -z_threshold,
                "depth_max": z_threshold,
            },
        ),
        TerminationTermCfg(
            func=tilt_axis_aligned,
            params={
                "subject_name": gear.name,
                "receiver_name": plate.name,
                "max_tilt_rad": math.radians(upright_axis_threshold_deg),
            },
        ),
        TerminationTermCfg(
            func=GearIsSupported,
            params={
                "plate_asset_cfg": SceneEntityCfg(plate.name),
                "gear_asset_cfg": SceneEntityCfg(gear.name),
                "support_z_threshold": support_z_threshold,
            },
        ),
        TerminationTermCfg(
            func=velocity_below_threshold,
            params={
                "subject_name": gear.name,
                "linear_velocity_threshold": linear_velocity_threshold,
                "angular_velocity_threshold": angular_velocity_threshold,
            },
        ),
    ]
    return TerminationTermCfg(
        func=CompositePredicate,
        params={"predicates": predicates, "mode": SuccessMode.ALL},
    )


class GearInsertionTask(TaskBase):
    """Require every configured gear to be seated and settled on the plate."""

    def __init__(
        self,
        plate: Asset,
        gears: list[Asset],
        target_offsets_xyz: Sequence[Sequence[float]],
        xy_threshold: float = 0.015,
        z_threshold: float = 0.01,
        upright_axis_threshold_deg: float = 15.0,
        linear_velocity_threshold: float = 0.05,
        angular_velocity_threshold: float = 0.5,
        support_z_threshold: float = 0.005,
        consecutive_success_steps: int = 10,
        episode_length_s: float = 120.0,
        task_description: str | None = None,
    ) -> None:
        gears = tuple(gears)
        if not gears:
            raise ValueError("gear insertion requires at least one gear asset")
        if len({gear.name for gear in gears}) != len(gears):
            raise ValueError("gear insertion requires unique gear asset names")

        offsets = tuple(tuple(offset) for offset in target_offsets_xyz)
        if len(offsets) != len(gears) or any(len(offset) != 3 for offset in offsets):
            raise ValueError("gear insertion requires one 3D target offset per gear")

        thresholds = {
            "xy_threshold": xy_threshold,
            "z_threshold": z_threshold,
            "linear_velocity_threshold": linear_velocity_threshold,
            "angular_velocity_threshold": angular_velocity_threshold,
            "support_z_threshold": support_z_threshold,
            "episode_length_s": episode_length_s,
        }
        for name, value in thresholds.items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if (
            isinstance(upright_axis_threshold_deg, bool)
            or not math.isfinite(upright_axis_threshold_deg)
            or not 0 < upright_axis_threshold_deg <= 180
        ):
            raise ValueError("upright_axis_threshold_deg must be in (0, 180]")
        if (
            isinstance(consecutive_success_steps, bool)
            or not isinstance(consecutive_success_steps, int)
            or consecutive_success_steps <= 0
        ):
            raise ValueError("consecutive_success_steps must be a positive integer")

        super().__init__(
            episode_length_s=episode_length_s,
            task_description=task_description or "Place all gears correctly on the assembly plate.",
        )
        self.plate = plate
        self.gears = gears
        self.target_offsets_xyz = offsets
        self.events_cfg = EventsCfg()
        gear_success_predicates = [
            _make_gear_success_composite_predicate_cfg(
                plate,
                gear,
                target_offset_xyz,
                xy_threshold=xy_threshold,
                z_threshold=z_threshold,
                upright_axis_threshold_deg=upright_axis_threshold_deg,
                linear_velocity_threshold=linear_velocity_threshold,
                angular_velocity_threshold=angular_velocity_threshold,
                support_z_threshold=support_z_threshold,
            )
            for gear, target_offset_xyz in zip(gears, offsets, strict=True)
        ]
        success = TerminationTermCfg(
            func=CompositePredicate,
            params={
                "predicates": gear_success_predicates,
                "mode": SuccessMode.ALL,
                "consecutive_steps": consecutive_success_steps,
            },
        )
        self.termination_cfg = TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[ProgressObjective(name="gear_insertion", predicate_sequence=[success])],
        )

    def get_scene_cfg(self) -> Any:
        return None

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return self.termination_cfg

    def get_events_cfg(self) -> Any:
        return self.events_cfg

    def get_mimic_env_cfg(self, arm_mode) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric(), GearInsertionFractionMetric(tuple(gear.name for gear in self.gears))]
