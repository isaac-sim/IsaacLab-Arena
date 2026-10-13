# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Ordered laptop opening and upright cup placement on the same table."""

from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
from isaaclab_arena.tasks.open_door_task import OpenDoorTask
from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
from isaaclab_arena.tasks.predicates.object_settling import get_rest_pose_recorder, objects_below_velocity_thresholds
from isaaclab_arena.tasks.predicates.spatial import (
    gripper_distance_from_object_exceeds_threshold,
    lateral_in_proximity,
    object_is_above_height,
    object_on_destination,
    tilt_axis_aligned,
    velocity_below_threshold,
)
from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

if TYPE_CHECKING:
    import torch

    from isaaclab_arena.embodiments.gripper import Gripper
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


class PlaceCupRightOfLaptopTask(PickAndPlaceTask):
    """Lift the cup, then hold an upright, supported placement to the laptop's right."""

    def __init__(
        self,
        laptop: Openable,
        cup: Asset,
        table: Asset,
        gripper: Gripper,
        cup_source_offset_xyz: tuple[float, float, float],
        cup_destination_offset_xyz: tuple[float, float, float],
        placement_tolerance_m: float = 0.055,
        minimum_lift_height_m: float = 0.06,
        placement_consecutive_steps: int = 10,
        gripper_retreat_distance_m: float = 0.20,
        episode_length_s: float = 45.0,
    ):
        """Configure the cup goal relative to the laptop's base frame.

        Args:
            laptop: Articulated laptop whose base frame defines the destination.
            cup: Rigid cup to lift and place.
            table: Physical support surface and source of the drop-failure height.
            gripper: Embodiment-owned gripper exposing its measured grasp-frame position.
            cup_source_offset_xyz: Required left-side cup location in the laptop frame, in meters.
            cup_destination_offset_xyz: Cup root destination in the laptop base frame, in meters.
                Only its horizontal position is constrained; contact determines support height.
            placement_tolerance_m: Maximum horizontal distance from the destination, in meters.
            minimum_lift_height_m: Required rise above the cup's recorded resting position, in meters.
            placement_consecutive_steps: Consecutive control steps satisfying all placement conditions.
            gripper_retreat_distance_m: Minimum grasp-frame distance from the placed cup's root, in meters.
            episode_length_s: Maximum duration in seconds when used as a standalone task.
        """
        assert math.isfinite(placement_tolerance_m) and placement_tolerance_m > 0.0
        assert math.isfinite(minimum_lift_height_m) and minimum_lift_height_m > 0.0
        assert math.isfinite(gripper_retreat_distance_m) and gripper_retreat_distance_m > 0.0
        assert len(cup_source_offset_xyz) == 3 and all(math.isfinite(value) for value in cup_source_offset_xyz)
        assert cup_source_offset_xyz[1] > placement_tolerance_m, "The source must be on the laptop's left"
        assert len(cup_destination_offset_xyz) == 3 and all(
            math.isfinite(coordinate) for coordinate in cup_destination_offset_xyz
        )
        self.laptop = laptop
        self.gripper = gripper
        self.cup_source_offset_xyz = cup_source_offset_xyz
        self.cup_destination_offset_xyz = cup_destination_offset_xyz
        self.placement_tolerance_m = placement_tolerance_m
        self.minimum_lift_height_m = minimum_lift_height_m
        self.gripper_retreat_distance_m = gripper_retreat_distance_m
        super().__init__(
            pick_up_object=cup,
            destination_location=table,
            background_scene=table,
            destination_object=table,
            episode_length_s=episode_length_s,
            task_description="Pick up the cup and place it upright on the table to the right of the laptop.",
            velocity_threshold=0.02,
            placement_consecutive_steps=placement_consecutive_steps,
        )

    def _cup_is_supported(self, env: IsaacLabArenaManagerBasedRLEnv) -> torch.Tensor:
        """Require table contact supporting a cup with low linear speed."""
        return object_on_destination(
            env,
            object_cfg=SceneEntityCfg(self.pick_up_object.name),
            destination_cfg=SceneEntityCfg(self.destination_location.name),
            contact_sensor_cfg=self.contact_sensor_cfg,
            force_threshold=self.force_threshold,
            velocity_threshold=self.velocity_threshold,
            support_cone_half_angle_rad=self.support_cone_half_angle_rad,
        )

    def _cup_is_settled_at_source(self, env: IsaacLabArenaManagerBasedRLEnv) -> torch.Tensor:
        """Observe a supported cup at its left-side source before counting lift."""
        settled = objects_below_velocity_thresholds(env, object_names=[self.pick_up_object.name])
        at_source = lateral_in_proximity(
            env,
            subject_name=self.pick_up_object.name,
            receiver_name=self.laptop.name,
            target_offset_xyz=self.cup_source_offset_xyz,
            tolerance_lateral=self.placement_tolerance_m,
        )
        settled_at_source = settled & at_source & self._cup_is_supported(env)
        # Only a supported source pose supplies the lift-height reference.
        get_rest_pose_recorder(env).record(
            self.pick_up_object.name,
            env.arena_world.get_position_w(self.pick_up_object.name),
            settled_at_source,
        )
        return settled_at_source

    def _cup_is_upright_at_destination(self, env: IsaacLabArenaManagerBasedRLEnv) -> torch.Tensor:
        """Require supported placement and gripper withdrawal together."""
        at_destination = lateral_in_proximity(
            env,
            subject_name=self.pick_up_object.name,
            receiver_name=self.laptop.name,
            target_offset_xyz=self.cup_destination_offset_xyz,
            tolerance_lateral=self.placement_tolerance_m,
        )
        upright = tilt_axis_aligned(
            env,
            subject_name=self.pick_up_object.name,
            receiver_name=self.destination_location.name,
            max_tilt_rad=math.radians(10.0),
        )
        resting = velocity_below_threshold(
            env,
            subject_name=self.pick_up_object.name,
            linear_velocity_threshold=self.velocity_threshold,
            angular_velocity_threshold=0.15,
        )
        # Withdrawal avoids assuming a grasp width for the cup's body, rim, or handle.
        # This checks final separation; it does not establish an earlier robot grasp.
        gripper_withdrawn = gripper_distance_from_object_exceeds_threshold(
            env,
            subject_name=self.pick_up_object.name,
            gripper=self.gripper,
            distance_threshold_m=self.gripper_retreat_distance_m,
        )
        return self._cup_is_supported(env) & at_destination & upright & resting & gripper_withdrawn

    def get_termination_cfg(self) -> TaskTerminationCfg:
        """Declare the cup sequence while retaining the parent's timeout, sensors, and failure checks."""
        termination_cfg = super().get_termination_cfg()
        termination_cfg.success = [
            CompletionCriteria(
                name="pick_and_place_cup",
                predicate_sequence=[
                    self._cup_is_settled_at_source,
                    partial(
                        object_is_above_height,
                        object_name=self.pick_up_object.name,
                        use_settled_state=True,
                        distance=self.minimum_lift_height_m,
                    ),
                    TrueForConsecutiveStepsCfg(
                        predicate=self._cup_is_upright_at_destination,
                        required_steps=self.placement_consecutive_steps,
                    ),
                ],
                description="Rest on the left, lift, then place upright on the right and withdraw the gripper.",
            )
        ]
        return termination_cfg


class LaptopCupTask(CompositeTaskBase):
    """Open the laptop before relocating the cup, keeping the laptop open at completion."""

    def __init__(
        self,
        laptop: Openable,
        cup: Asset,
        table: Asset,
        gripper: Gripper,
        cup_source_offset_xyz: tuple[float, float, float],
        cup_destination_offset_xyz: tuple[float, float, float],
        openness_threshold: float = 0.8,
        episode_length_s: float = 45.0,
        placement_tolerance_m: float = 0.055,
        minimum_lift_height_m: float = 0.06,
        gripper_retreat_distance_m: float = 0.20,
    ):
        """Configure two ordered subtasks with native Arena progress reporting.

        Args:
            laptop: Laptop exposing the Openable affordance for its lid hinge.
            cup: Rigid cup initially standing to the laptop's left.
            table: Physical wooden table supporting the laptop and cup.
            gripper: Embodiment-owned gripper used to require withdrawal after placement.
            cup_source_offset_xyz: Required left-side cup location in the laptop frame, in meters.
            cup_destination_offset_xyz: Cup destination in the laptop's base frame, in meters.
            openness_threshold: Required opening as a fraction of the hinge's joint limits.
            episode_length_s: Time budget in seconds for both subtasks together.
            placement_tolerance_m: Maximum horizontal distance from the destination, in meters.
            minimum_lift_height_m: Required rise above the cup's recorded resting position, in meters.
            gripper_retreat_distance_m: Required final grasp-frame distance from the cup root, in meters.
        """
        open_laptop = OpenDoorTask(
            openable_object=laptop,
            openness_threshold=openness_threshold,
            reset_openness=0.0,
            task_description="Open the laptop lid.",
        )
        place_cup = PlaceCupRightOfLaptopTask(
            laptop=laptop,
            cup=cup,
            table=table,
            gripper=gripper,
            cup_source_offset_xyz=cup_source_offset_xyz,
            cup_destination_offset_xyz=cup_destination_offset_xyz,
            episode_length_s=episode_length_s,
            placement_tolerance_m=placement_tolerance_m,
            minimum_lift_height_m=minimum_lift_height_m,
            gripper_retreat_distance_m=gripper_retreat_distance_m,
        )
        super().__init__(
            subtasks=[open_laptop, place_cup],
            subtasks_are_sequential=True,
            desired_subtask_success_state=[True, True],
            episode_length_s=episode_length_s,
            task_description=(
                "Open the laptop, then pick up the cup from its left and place the cup upright "
                "on the table to its right. Leave the laptop open."
            ),
        )
