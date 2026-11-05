# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Ordered laptop opening and upright cup placement on the same table."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
from isaaclab_arena.tasks.open_door_task import OpenDoorTask
from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
from isaaclab_arena.tasks.predicates.spatial import lateral_in_proximity, tilt_axis_aligned, velocity_below_threshold
from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

if TYPE_CHECKING:
    import torch

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


class PlaceCupRightOfLaptopTask(PickAndPlaceTask):
    """Lift the cup, then hold an upright, supported placement to the laptop's right."""

    def __init__(
        self,
        laptop: Openable,
        cup: Asset,
        table: Asset,
        cup_destination_offset_xyz: tuple[float, float, float],
        placement_tolerance_m: float = 0.055,
        minimum_lift_height_m: float = 0.06,
        placement_consecutive_steps: int = 10,
        episode_length_s: float = 45.0,
    ):
        """Configure the cup goal relative to the laptop's base frame.

        Args:
            laptop: Articulated laptop whose base frame defines the destination.
            cup: Rigid cup to lift and place.
            table: Physical support surface and source of the drop-failure height.
            cup_destination_offset_xyz: Cup root destination in the laptop base frame, in meters.
                Only its horizontal position is constrained; contact determines support height.
            placement_tolerance_m: Maximum horizontal distance from the destination, in meters.
            minimum_lift_height_m: Required rise above the cup's recorded resting position, in meters.
            placement_consecutive_steps: Consecutive control steps satisfying all placement conditions.
            episode_length_s: Maximum duration in seconds when used as a standalone task.
        """
        assert math.isfinite(placement_tolerance_m) and placement_tolerance_m > 0.0
        assert math.isfinite(minimum_lift_height_m) and minimum_lift_height_m > 0.0
        assert len(cup_destination_offset_xyz) == 3 and all(
            math.isfinite(coordinate) for coordinate in cup_destination_offset_xyz
        )
        self.laptop = laptop
        self.cup_destination_offset_xyz = cup_destination_offset_xyz
        self.placement_tolerance_m = placement_tolerance_m
        self.minimum_lift_height_m = minimum_lift_height_m
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

    def _cup_is_upright_at_destination(
        self,
        env: IsaacLabArenaManagerBasedRLEnv,
        supported_placement: Callable[[IsaacLabArenaManagerBasedRLEnv], torch.Tensor],
    ) -> torch.Tensor:
        """Require support, the destination, uprightness, and low speed together."""
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
        return supported_placement(env) & at_destination & upright & resting

    def get_termination_cfg(self) -> TaskTerminationCfg:
        """Retain the native settle and lift milestones and strengthen the final placement."""
        termination_cfg = super().get_termination_cfg()
        placement_criteria = termination_cfg.success[0]
        assert placement_criteria.predicate_sequence is not None
        predicate_sequence = list(placement_criteria.predicate_sequence)
        predicate_sequence[1] = partial(predicate_sequence[1], distance=self.minimum_lift_height_m)
        placement_requirement = predicate_sequence[-1]
        assert isinstance(placement_requirement, TrueForConsecutiveStepsCfg)
        predicate_sequence[-1] = replace(
            placement_requirement,
            predicate=partial(
                self._cup_is_upright_at_destination,
                supported_placement=placement_requirement.predicate,
            ),
        )
        termination_cfg.success = [
            replace(
                placement_criteria,
                name="pick_and_place_cup",
                predicate_sequence=predicate_sequence,
                description="Settle, lift, then place the cup upright on the table to the laptop's right.",
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
        cup_destination_offset_xyz: tuple[float, float, float],
        openness_threshold: float = 0.8,
        episode_length_s: float = 45.0,
        placement_tolerance_m: float = 0.055,
        minimum_lift_height_m: float = 0.06,
    ):
        """Configure two ordered subtasks with native Arena progress reporting.

        Args:
            laptop: Laptop exposing the Openable affordance for its lid hinge.
            cup: Rigid cup initially standing to the laptop's left.
            table: Physical wooden table supporting the laptop and cup.
            cup_destination_offset_xyz: Cup destination in the laptop's base frame, in meters.
            openness_threshold: Required opening as a fraction of the hinge's joint limits.
            episode_length_s: Time budget in seconds for both subtasks together.
            placement_tolerance_m: Maximum horizontal distance from the destination, in meters.
            minimum_lift_height_m: Required rise above the cup's recorded resting position, in meters.
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
            cup_destination_offset_xyz=cup_destination_offset_xyz,
            episode_length_s=episode_length_s,
            placement_tolerance_m=placement_tolerance_m,
            minimum_lift_height_m=minimum_lift_height_m,
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
