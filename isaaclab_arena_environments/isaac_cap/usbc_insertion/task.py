# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reusable receiver-frame success definition for USB-C insertion."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from isaaclab.managers import TerminationTermCfg

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.predicates.composite import CompositePredicate
from isaaclab_arena.tasks.predicates.gripper import gripper_released
from isaaclab_arena.tasks.predicates.spatial import (
    depth_in_range,
    gripper_distance_from_object_exceeds_threshold,
    lateral_in_proximity,
    tilt_axis_aligned,
    velocity_below_threshold,
)
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.tasks.terminations import SuccessMode

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase


class UsbcInsertionTask(TaskBase):
    """Require a seated, slow plug with optional release and hand withdrawal."""

    def __init__(
        self,
        plug: Asset,
        receiver: Asset,
        *,
        receiver_mouth_offset_xyz: tuple[float, float, float],
        receiver_axis: tuple[float, float, float],
        subject_tip_offset_xyz: tuple[float, float, float],
        depth_min: float,
        lateral_max: float,
        speed_max: float,
        depth_max: float | None = None,
        subject_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
        tilt_max: float | None = None,
        allow_antiparallel_axes: bool = False,
        grasp_width_m: float | None = None,
        release_clearance_m: float = 1.5e-3,
        withdrawal_distance_min: float | None = None,
        require_released: bool = False,
        consecutive_success_steps: int = 1,
        episode_length_s: float = 120.0,
        task_description: str | None = None,
    ) -> None:
        """Configure geometry and thresholds supplied by one connector variant."""
        for name, vector in (
            ("receiver_mouth_offset_xyz", receiver_mouth_offset_xyz),
            ("receiver_axis", receiver_axis),
            ("subject_tip_offset_xyz", subject_tip_offset_xyz),
            ("subject_axis", subject_axis),
        ):
            assert len(vector) == 3 and all(
                math.isfinite(value) for value in vector
            ), f"{name} must contain three finite values."
        assert math.dist(receiver_axis, (0.0, 0.0, 0.0)) > 0.0, "receiver_axis must be non-zero."
        assert math.dist(subject_axis, (0.0, 0.0, 0.0)) > 0.0, "subject_axis must be non-zero."
        assert depth_min > 0.0 and math.isfinite(depth_min), "depth_min must be positive and finite."
        assert depth_max is None or depth_max >= depth_min, "depth_max must not be less than depth_min."
        assert lateral_max > 0.0 and math.isfinite(lateral_max), "lateral_max must be positive and finite."
        assert speed_max > 0.0 and math.isfinite(speed_max), "speed_max must be positive and finite."
        assert grasp_width_m is None or (
            math.isfinite(grasp_width_m) and grasp_width_m > 0.0
        ), "grasp_width_m must be positive and finite."
        assert (
            math.isfinite(release_clearance_m) and release_clearance_m >= 0.0
        ), "release_clearance_m must be non-negative and finite."
        assert not require_released or grasp_width_m is not None, "Release checking requires grasp_width_m."
        assert withdrawal_distance_min is None or (
            math.isfinite(withdrawal_distance_min) and withdrawal_distance_min > 0.0
        ), "withdrawal_distance_min must be positive and finite."
        assert tilt_max is None or 0.0 <= tilt_max <= math.pi, "tilt_max must be in [0, pi]."
        assert isinstance(consecutive_success_steps, int) and not isinstance(
            consecutive_success_steps, bool
        ), "consecutive_success_steps must be an integer."
        assert consecutive_success_steps > 0, "consecutive_success_steps must be positive."
        assert episode_length_s > 0.0 and math.isfinite(
            episode_length_s
        ), "episode_length_s must be positive and finite."

        super().__init__(
            episode_length_s=episode_length_s,
            task_description=task_description or "Insert the USB-C plug into the receiver.",
        )
        self.plug = plug
        self.receiver = receiver

        mating_params = {
            "subject_name": plug.name,
            "receiver_name": receiver.name,
            "subject_offset_xyz": tuple(subject_tip_offset_xyz),
            "target_offset_xyz": tuple(receiver_mouth_offset_xyz),
            "receiver_axis": tuple(receiver_axis),
        }
        predicates = [
            TerminationTermCfg(
                func=depth_in_range,
                params={
                    **mating_params,
                    "depth_min": depth_min,
                    "depth_max": depth_max,
                },
            ),
            TerminationTermCfg(
                func=lateral_in_proximity,
                params={**mating_params, "tolerance_lateral": lateral_max},
            ),
        ]
        if tilt_max is not None:
            predicates.append(
                TerminationTermCfg(
                    func=tilt_axis_aligned,
                    params={
                        "subject_name": plug.name,
                        "receiver_name": receiver.name,
                        "subject_axis": tuple(subject_axis),
                        "receiver_axis": tuple(receiver_axis),
                        "max_tilt_rad": tilt_max,
                        "allow_antiparallel": allow_antiparallel_axes,
                    },
                )
            )
        predicates.append(
            TerminationTermCfg(
                func=velocity_below_threshold,
                params={
                    "subject_name": plug.name,
                    "linear_velocity_threshold": speed_max,
                },
            )
        )
        if require_released:
            release_cfg = TerminationTermCfg(
                func=gripper_released,
                params={
                    "grasp_width_m": grasp_width_m,
                    "release_clearance_m": release_clearance_m,
                },
            )
            predicates.append(release_cfg)
        if withdrawal_distance_min is not None:
            withdrawal_cfg = TerminationTermCfg(
                func=gripper_distance_from_object_exceeds_threshold,
                params={
                    "subject_name": plug.name,
                    "distance_threshold_m": withdrawal_distance_min,
                },
            )
            predicates.append(withdrawal_cfg)
        self._success_cfg = TerminationTermCfg(
            func=CompositePredicate,
            params={
                "predicates": predicates,
                "mode": SuccessMode.ALL,
                "consecutive_steps": consecutive_success_steps,
            },
        )
        self._gripper_predicates = [
            predicate
            for predicate in self._success_cfg.params["predicates"]
            if predicate.func in (gripper_released, gripper_distance_from_object_exceeds_threshold)
        ]
        self.termination_cfg = TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                ProgressObjective(
                    name="usbc_insertion",
                    predicate_sequence=[self._success_cfg],
                )
            ],
        )

    def configure_for_embodiment(self, embodiment: EmbodimentBase) -> None:
        """Bind gripper-dependent predicates to the selected embodiment."""
        if not self._gripper_predicates:
            return
        gripper = embodiment.get_gripper()
        for predicate in self._gripper_predicates:
            predicate.params["gripper"] = gripper

    def get_scene_cfg(self) -> Any:
        return None

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return self.termination_cfg

    def get_events_cfg(self) -> Any:
        return None

    def get_mimic_env_cfg(self, arm_mode) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]
