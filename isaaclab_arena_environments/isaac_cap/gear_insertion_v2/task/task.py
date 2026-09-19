# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena representation of the AUTOLab gear-mesh task families."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
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
from .terminations import gear_mesh_success


def _gear_outer_diameter_m(gear_teeth: int) -> float:
    """Return gear outer diameter for the task's 2.5 mm module."""
    return 0.0025 * (gear_teeth + 2)


@configclass
class EventsCfg:
    reset_all: EventTermCfg = EventTermCfg(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


class GearMeshTask(TaskBase):
    """Seat every sampled gear, press the button, and prove the train works."""

    def __init__(
        self,
        board: Asset,
        gear: Asset | None = None,
        gears: list[Asset] | None = None,
        gear_teeth: int = 20,
        target_offsets_xyz: Sequence[Sequence[float]] | None = None,
        grasp_width_m: float | None = None,
        release_clearance_m: float = 0.005,
        episode_length_s: float = 100.0,
        task_description: str | None = None,
    ) -> None:
        gear_assets = tuple(gears or (() if gear is None else (gear,)))
        if not gear_assets:
            raise ValueError("gear mesh requires at least one loose gear")
        if len({asset.name for asset in gear_assets}) != len(gear_assets):
            raise ValueError("gear mesh requires unique loose gear names")
        offsets = tuple(
            tuple(float(value) for value in offset) for offset in (target_offsets_xyz or ((0.032, 0.0, 0.008),))
        )
        if len(offsets) != len(gear_assets) or any(len(offset) != 3 for offset in offsets):
            raise ValueError("gear mesh requires one 3D station offset per gear")
        if grasp_width_m is not None and (not math.isfinite(grasp_width_m) or grasp_width_m <= 0.0):
            raise ValueError("grasp_width_m must be a positive finite number")
        if not math.isfinite(release_clearance_m) or release_clearance_m < 0.0:
            raise ValueError("release_clearance_m must be a non-negative finite number")
        teeth = tuple(int(gear_teeth) for _ in gear_assets)

        super().__init__(
            episode_length_s=episode_length_s,
            task_description=task_description
            or "put the gear on its peg so it meshes with the pinion, then press the red button at the near end of the board to start it",
        )
        self.board = board
        self.gears = gear_assets
        self.gear = gear_assets[0]
        self._grasp_width_override_m = grasp_width_m
        self.events_cfg = EventsCfg()
        self._success_cfg = TerminationTermCfg(
            func=gear_mesh_success,
            params={
                "board_asset_cfg": SceneEntityCfg(board.name),
                "gear_asset_cfgs": [SceneEntityCfg(asset.name) for asset in gear_assets],
                "grasp_width_m": grasp_width_m if grasp_width_m is not None else _gear_outer_diameter_m(gear_teeth),
                "release_clearance_m": release_clearance_m,
                "target_offsets_xyz": offsets,
                "button_latch_m": 0.005,
                "drive_speed_rad_s": 4.0,
                "spin_fraction": 0.3,
                "gear_teeth": teeth,
                "spin_window_s": 0.5,
                "xy_threshold_m": 0.005348669,
                "z_threshold_m": 0.008,
                "hold_time_s": 1.0,
            },
        )

    def configure_for_embodiment(self, embodiment: EmbodimentBase) -> None:
        """Configure release checks to use the embodiment's gripper."""
        self._success_cfg.params["gripper"] = embodiment.get_gripper()

    def set_gear_teeth(self, gear_teeth: int) -> None:
        """Update the expected driven speed for the selected asset family."""
        self.configure_layout(
            (int(gear_teeth),),
            self._success_cfg.params["target_offsets_xyz"],
        )

    def configure_layout(
        self,
        gear_teeth: Sequence[int],
        target_offsets_xyz: Sequence[Sequence[float]],
    ) -> None:
        """Configure the selected board's station classes and local targets."""
        teeth = tuple(int(value) for value in gear_teeth)
        offsets = tuple(tuple(float(value) for value in offset) for offset in target_offsets_xyz)
        if len(teeth) != len(self.gears) or len(offsets) != len(self.gears):
            raise ValueError("selected gear-mesh layout does not match the scene gear count")
        if any(value not in (16, 20, 24) for value in teeth):
            raise ValueError("gear-mesh station teeth must be 16, 20, or 24")
        if any(len(offset) != 3 for offset in offsets):
            raise ValueError("gear-mesh station offsets must be 3D")
        self._success_cfg.params["gear_teeth"] = teeth
        self._success_cfg.params["target_offsets_xyz"] = offsets
        if self._grasp_width_override_m is None:
            self._success_cfg.params["grasp_width_m"] = max(_gear_outer_diameter_m(value) for value in teeth)

    def get_scene_cfg(self) -> Any:
        return None

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[ProgressObjective(name="gear_mesh", predicate_sequence=[self._success_cfg])],
        )

    def get_events_cfg(self) -> Any:
        return self.events_cfg

    def get_mimic_env_cfg(self, arm_mode) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]


def _make_legacy_gear_success_cfg(
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
    """Build the retired Factory-gear predicate used by compatibility tests."""
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
    """Retain the retired Factory-gear task API for downstream compatibility."""

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
        """Configure the legacy plate-and-settled-gears success criteria."""
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
            _make_legacy_gear_success_cfg(
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
        self.termination_cfg = TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                ProgressObjective(
                    name="gear_insertion",
                    predicate_sequence=[
                        TerminationTermCfg(
                            func=CompositePredicate,
                            params={
                                "predicates": gear_success_predicates,
                                "mode": SuccessMode.ALL,
                                "consecutive_steps": consecutive_success_steps,
                            },
                        )
                    ],
                )
            ],
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
