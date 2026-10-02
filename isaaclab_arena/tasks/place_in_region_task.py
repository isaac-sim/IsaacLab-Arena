# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Place a rigid component completely inside an explicit live box region."""

from __future__ import annotations

import math
from dataclasses import asdict

from isaaclab.managers import TerminationTermCfg

from isaaclab_arena.agentic_environment_generation.authoring_metadata import AuthoringMetadata, ParameterMetadata
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import agent_ready, register_task
from isaaclab_arena.geometry.containment import BoxRegion
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.predicates.gripper import gripper_released
from isaaclab_arena.tasks.predicates.object_settling import objects_below_velocity_thresholds
from isaaclab_arena.tasks.predicates.regions import ObjectInRegion
from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.tasks.task_transition import TaskTransition
from isaaclab_arena.tasks.terminations import check_success


@agent_ready
@register_task
class PlaceInRegionTask(TaskBase):
    """Require full collision-shape containment, settling, and optional measured gripper release.

    No prior lift or grasp is required. Containment does not establish support contact.
    The subject must have static cube/circular-cylinder colliders in one rigid frame;
    meshes, articulations, animated shapes, and unsupported spawn overrides are rejected.
    The destination's live root frame carries the region and requires unit spawn scale.
    """

    authoring_metadata = AuthoringMetadata(
        parameters={
            "region_bounds": ParameterMetadata(units="m", description="Lower and upper corners in region frame R."),
            "region_position_xyz": ParameterMetadata(units="m", description="Region origin in the destination frame."),
            "region_rotation_xyzw": ParameterMetadata(description="Unit XYZW quaternion from region to destination."),
            "floor_allowance_m": ParameterMetadata(units="m", minimum=0),
            "linear_velocity_threshold": ParameterMetadata(units="m/s", minimum=0),
            "angular_velocity_threshold": ParameterMetadata(units="rad/s", minimum=0),
            "grasp_width_m": ParameterMetadata(units="m", minimum=0, description="Omit to disable the release check."),
            "release_clearance_m": ParameterMetadata(units="m", minimum=0),
            "placement_consecutive_steps": ParameterMetadata(minimum=1),
            "episode_length_s": ParameterMetadata(units="s", minimum=0),
        },
        requires={"subject": ("rigid",), "destination": ("root_frame",)},
        constraints=(
            "Subject geometry must consist of supported static collision cubes and circular cylinders.",
            "Destination spawn scale must be unit; region bounds are explicit physical meters.",
            "Optional gripper release requires an embodiment that reports measured opening width.",
        ),
        reset_semantics="Owns no scene reset; only the progress tracker's consecutive-step counter resets.",
    )

    def __init__(
        self,
        subject: Asset,
        destination: Asset,
        region_bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
        region_position_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        region_rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        floor_allowance_m: float = 0.0,
        linear_velocity_threshold: float = 0.03,
        angular_velocity_threshold: float = 0.2,
        grasp_width_m: float | None = None,
        release_clearance_m: float = 0.0015,
        placement_consecutive_steps: int = 1,
        episode_length_s: float | None = None,
        task_description: str | None = None,
    ):
        """Configure a destination-relative placement condition.

        Args:
            subject: Rigid object to place.
            destination: Scene object whose measured root pose carries the region.
            region_bounds: Lower and upper corners in region frame R, in meters.
            region_position_xyz: Translation of R in the destination frame.
            region_rotation_xyzw: Rotation mapping R into the destination frame.
            floor_allowance_m: Contact allowance on the lower Z face only.
            linear_velocity_threshold: Strict subject linear-speed limit, in m/s.
            angular_velocity_threshold: Strict subject angular-speed limit, in rad/s.
            grasp_width_m: Object grasp width in meters; None disables the release check.
            release_clearance_m: Extra measured jaw clearance required when checking release.
            placement_consecutive_steps: Consecutive steps satisfying all conditions together.
            episode_length_s: Overall time budget in seconds.
            task_description: Optional instruction replacing the generated description.
        """
        super().__init__(
            episode_length_s, task_description or f"Place {subject.name} completely inside {destination.name}."
        )
        assert getattr(subject, "object_type", None) == ObjectType.RIGID, "Placement subject must be a rigid object"
        assert subject.name != destination.name, "Subject and destination must be distinct scene objects"
        self.subject = subject
        self.destination = destination
        self.region = BoxRegion(
            destination.name, region_bounds, region_position_xyz, region_rotation_xyzw, floor_allowance_m
        )
        for value in (linear_velocity_threshold, angular_velocity_threshold, release_clearance_m):
            assert math.isfinite(value) and value >= 0.0, "Placement thresholds must be finite and nonnegative"
        assert grasp_width_m is None or (math.isfinite(grasp_width_m) and grasp_width_m > 0.0), "Invalid grasp width"
        assert isinstance(placement_consecutive_steps, int) and not isinstance(placement_consecutive_steps, bool)
        assert placement_consecutive_steps > 0, "Placement dwell must be positive"
        self.linear_velocity_threshold = linear_velocity_threshold
        self.angular_velocity_threshold = angular_velocity_threshold
        self.grasp_width_m = grasp_width_m
        self.release_clearance_m = release_clearance_m
        self.placement_consecutive_steps = placement_consecutive_steps
        self.gripper = None

    def configure_for_embodiment(self, embodiment) -> None:
        self.gripper = embodiment.gripper

    def apply_reachability_constraints(self) -> None:
        self._apply_reachability_constraints([self.subject, self.destination])

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode):
        return None

    def get_metrics(self):
        return [SuccessRateMetric()]

    def get_termination_cfg(self) -> TaskTerminationCfg:
        predicates = [
            TerminationTermCfg(
                func=ObjectInRegion, params={"object_name": self.subject.name, "region": asdict(self.region)}
            ),
            TerminationTermCfg(
                func=objects_below_velocity_thresholds,
                params={
                    "object_names": [self.subject.name],
                    "lin_vel_threshold": self.linear_velocity_threshold,
                    "ang_vel_threshold": self.angular_velocity_threshold,
                },
            ),
        ]
        if self.grasp_width_m is not None:
            assert self.gripper is not None, "Bind an embodiment gripper before configuring the release predicate"
            predicates.append(
                TerminationTermCfg(
                    func=gripper_released,
                    params={
                        "gripper": self.gripper,
                        "grasp_width_m": self.grasp_width_m,
                        "release_clearance_m": self.release_clearance_m,
                    },
                )
            )
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="place_in_region",
                    predicate_sequence=[
                        TrueForConsecutiveStepsCfg(
                            predicate=TerminationTermCfg(func=check_success, params={"predicates": predicates}),
                            required_steps=self.placement_consecutive_steps,
                        )
                    ],
                )
            ],
        )

    @classmethod
    def success_state_transition(cls, subject: str, **_) -> TaskTransition:
        """Identify the manipulated node; current graph relations cannot express arbitrary region containment."""
        return TaskTransition(subject=subject)
