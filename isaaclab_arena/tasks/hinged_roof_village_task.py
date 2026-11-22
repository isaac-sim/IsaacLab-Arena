# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Find the blue animal under the hinged roofs of a toy village, move it to the pen, and restore the village."""

from __future__ import annotations

import math
import numpy as np
import torch
from dataclasses import field
from functools import partial
from typing import TYPE_CHECKING

from isaaclab.envs.common import ViewerCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse

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
from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


def _position_in_frame(env: IsaacLabArenaManagerBasedRLEnv, frame_name: str, object_name: str) -> torch.Tensor:
    """Return an object's origin in the root frame of another scene entity."""
    T_W_F = env.arena_world.get_pose_w(frame_name)
    t_W_O = env.arena_world.get_position_w(object_name)
    return quat_apply_inverse(T_W_F[:, 3:], t_W_O - T_W_F[:, :3])


def object_in_box(
    env: IsaacLabArenaManagerBasedRLEnv,
    frame_name: str,
    object_name: str,
    center_xy: tuple[float, float],
    half_size_m: float,
    min_height_m: float,
    max_height_m: float,
) -> torch.Tensor:
    """Return whether an object's origin lies in a vertical box given in another entity's root frame."""
    t_F_O = _position_in_frame(env, frame_name, object_name)
    return (
        ((t_F_O[:, 0] - center_xy[0]).abs() <= half_size_m)
        & ((t_F_O[:, 1] - center_xy[1]).abs() <= half_size_m)
        & (t_F_O[:, 2] >= min_height_m)
        & (t_F_O[:, 2] <= max_height_m)
    )


def roofs_closed(
    env: IsaacLabArenaManagerBasedRLEnv, village_name: str, joint_names: list[str], tolerance_rad: float
) -> torch.Tensor:
    """Return whether every roof hinge is within the tolerance of closed (zero)."""
    joint_positions = env.arena_world.get_joint_positions(village_name, tuple(joint_names))
    return (joint_positions.abs() <= tolerance_rad).all(dim=-1)


def board_on_mat(
    env: IsaacLabArenaManagerBasedRLEnv,
    village_name: str,
    mat_name: str,
    max_offset_m: float,
    max_height_m: float,
    tilt_tolerance_rad: float,
) -> torch.Tensor:
    """Return whether the village board lies flat on its mat, with its center within max_offset_m of the mat's."""
    t_M_B = _position_in_frame(env, mat_name, village_name)
    q_W_B = env.arena_world.get_pose_w(village_name)[:, 3:]
    up_B = torch.tensor([0.0, 0.0, 1.0], device=q_W_B.device).expand(q_W_B.shape[0], 3)
    flat = quat_apply(q_W_B, up_B)[:, 2] >= math.cos(tilt_tolerance_rad)
    return (
        (t_M_B[:, 0].abs() <= max_offset_m) & (t_M_B[:, 1].abs() <= max_offset_m) & (t_M_B[:, 2] <= max_height_m) & flat
    )


def animal_restored_village(
    env: IsaacLabArenaManagerBasedRLEnv,
    blue_in_pen_kwargs: dict,
    others_home_kwargs: list[dict],
    roofs_closed_kwargs: dict,
    board_on_mat_kwargs: dict,
    blue_animal_name: str,
    speed_threshold_m_s: float,
) -> torch.Tensor:
    """Return whether the blue animal rests in the pen, others are home, roofs are closed and the board is on its mat."""
    result = object_in_box(env, **blue_in_pen_kwargs)
    speed = torch.linalg.vector_norm(env.arena_world.get_root_linear_velocity_w(blue_animal_name), dim=-1)
    result &= speed < speed_threshold_m_s
    for kwargs in others_home_kwargs:
        result &= object_in_box(env, **kwargs)
    result &= roofs_closed(env, **roofs_closed_kwargs)
    result &= board_on_mat(env, **board_on_mat_kwargs)
    return result


@register_task
class HingedRoofVillageTask(TaskBase):
    """Move the blue animal from its hidden house to the pen; others stay home, roofs end closed, board stays on mat.

    The village asset owns the hidden hinge edge of each roof. This task owns the other hidden variable, the house the
    blue animal starts in: it seats the blue animal there and the other animals in the remaining houses in order, at
    construction, so the animal nodes need no poses of their own. Set the house per run with the blue animal's
    ``home_house`` variation, for example ``blue_animal.home_house.enabled=true blue_animal.home_house.houses=[3]``.
    Houses are numbered 1 to 4 for h1 (front-left), h2 (front-right), h3 (back-left) and h4 (back-right).

    Args:
        village: Hinged-roof village articulation.
        mat: Mat the village board stands on.
        pen: Pen the blue animal must end in.
        blue_animal: Animal to move to the pen.
        other_animals: Animals that must stay in their houses.
        blue_animal_house: Number of the house the blue animal starts in, 1 to 4 for h1 to h4.
        episode_length_s: Maximum episode duration in seconds.
        task_description: Natural-language task instruction.
        roof_closed_tolerance_deg: Maximum roof angle from closed, in degrees.
        pen_margin_m: Margin kept from the pen fence by the blue animal's origin.
        house_margin_m: Margin kept from the house walls by each other animal's origin.
        board_max_offset_m: Maximum offset of the board center from the mat center.
        speed_threshold_m_s: Blue-animal speed must be below this.
        success_hold_steps: Consecutive control steps the final state must hold.
    """

    def __init__(
        self,
        village: Asset,
        mat: Asset,
        pen: Asset,
        blue_animal: Asset,
        other_animals: list[Asset],
        blue_animal_house: int = 1,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        roof_closed_tolerance_deg: float = 5.0,
        pen_margin_m: float = 0.01,
        house_margin_m: float = 0.0,
        board_max_offset_m: float = 0.03,
        speed_threshold_m_s: float = 0.03,
        success_hold_steps: int = 15,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        assert success_hold_steps > 0, "success_hold_steps must be positive."
        assert len(other_animals) < len(village.house_names), "Each animal needs its own house."
        self.village = village
        self.mat = mat
        self.pen = pen
        self.blue_animal = blue_animal
        self.other_animals = list(other_animals)
        self.roof_closed_tolerance_rad = math.radians(roof_closed_tolerance_deg)
        self.pen_margin_m = pen_margin_m
        self.house_margin_m = house_margin_m
        self.board_max_offset_m = board_max_offset_m
        self.speed_threshold_m_s = speed_threshold_m_s
        self.success_hold_steps = success_hold_steps
        self.animal_bottom_offsets_m = {
            animal.name: -float(animal.get_bounding_box().min_point[0, 2])
            for animal in (blue_animal, *self.other_animals)
        }
        self.home_houses: dict[str, str] = {}
        self.set_blue_animal_house(blue_animal_house)
        blue_animal.add_variation(BlueAnimalHouseVariation(self))
        if self.task_description is None:
            self.task_description = (
                f"Put the {blue_animal.name} in the {pen.name}. Every other animal must stay in its own house, all"
                f" roofs must end closed, and the {village.name} board must stay on its {mat.name}."
            )

    def set_blue_animal_house(self, house_number: int) -> None:
        """Seat the blue animal in house h<house_number> and the other animals in the remaining houses, in order."""
        house_name = f"h{house_number}"
        assert house_name in self.village.house_names, f"house number must name one of {self.village.house_names}"
        remaining_houses = [house for house in self.village.house_names if house != house_name]
        self.home_houses = {self.blue_animal.name: house_name}
        for animal, house in zip(self.other_animals, remaining_houses):
            self.home_houses[animal.name] = house
        for animal in (self.blue_animal, *self.other_animals):
            pose = self.village.get_house_floor_pose(
                self.home_houses[animal.name], self.animal_bottom_offsets_m[animal.name]
            )
            animal.set_initial_pose(pose)

    def apply_reachability_constraints(self) -> None:
        """The robot must reach the blue animal and the pen."""
        self._apply_reachability_constraints([self.blue_animal, self.pen])

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("HingedRoofVillageTask has no Mimic configuration.")

    def _house_box_kwargs(self, animal_name: str) -> dict:
        geometry = self.village.geometry
        return dict(
            frame_name=self.village.name,
            object_name=animal_name,
            center_xy=geometry.get_house_center_B(self.home_houses[animal_name]),
            half_size_m=geometry.house_interior_half_m - self.house_margin_m,
            min_height_m=geometry.board_thickness_m - 0.01,
            max_height_m=geometry.roof_bottom_z_m,
        )

    def get_termination_cfg(self) -> TaskTerminationCfg:
        blue_name = self.blue_animal.name
        blue_in_pen_kwargs = dict(
            frame_name=self.pen.name,
            object_name=blue_name,
            center_xy=(0.0, 0.0),
            half_size_m=self.pen.interior_half_size_m - self.pen_margin_m,
            min_height_m=-0.01,
            max_height_m=self.pen.wall_height_m + 0.02,
        )
        roofs_closed_kwargs = dict(
            village_name=self.village.name,
            joint_names=self.village.roof_joint_names,
            tolerance_rad=self.roof_closed_tolerance_rad,
        )
        mat_top_m = float(self.mat.get_bounding_box().max_point[0, 2])
        board_on_mat_kwargs = dict(
            village_name=self.village.name,
            mat_name=self.mat.name,
            max_offset_m=self.board_max_offset_m,
            max_height_m=mat_top_m + 0.01,
            tilt_tolerance_rad=math.radians(5.0),
        )
        final_state = partial(
            animal_restored_village,
            blue_in_pen_kwargs=blue_in_pen_kwargs,
            others_home_kwargs=[self._house_box_kwargs(animal.name) for animal in self.other_animals],
            roofs_closed_kwargs=roofs_closed_kwargs,
            board_on_mat_kwargs=board_on_mat_kwargs,
            blue_animal_name=blue_name,
            speed_threshold_m_s=self.speed_threshold_m_s,
        )
        blue_home_kwargs = self._house_box_kwargs(blue_name)

        def blue_left_home(env: IsaacLabArenaManagerBasedRLEnv) -> torch.Tensor:
            return ~object_in_box(env, **blue_home_kwargs)

        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="blue_animal_to_pen_and_village_restored",
                    predicate_sequence=[
                        (blue_left_home, 0.2),
                        (partial(object_in_box, **blue_in_pen_kwargs), 0.3),
                        (
                            TrueForConsecutiveStepsCfg(predicate=final_state, required_steps=self.success_hold_steps),
                            0.5,
                        ),
                    ],
                ),
            ],
        )

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(lookat_object=self.village, offset=np.array([-0.6, -0.5, 0.5]))

    @classmethod
    def success_state_transition(cls, blue_animal: str, pen: str, **_) -> TaskTransition:
        """Success moves the blue animal into the pen; the env graph has no node state for the roofs yet."""
        return TaskTransition(subject=blue_animal)


@configclass
class BlueAnimalHouseVariationCfg(VariationBaseCfg):
    """Configuration for BlueAnimalHouseVariation."""

    houses: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    """Numbers of the houses (1 to 4 for h1 to h4) the start house is drawn from; a one-element list selects it."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)
    """Uniform draw of one house per build, seeded by the build seed."""


class BlueAnimalHouseVariation(BuildTimeVariationBase):
    """Draw the house the blue animal starts in once per build; the other animals fill the remaining houses.

    Args:
        task: Village task whose animals the variation seats.
        cfg: Houses to draw from. By default the variation is off.
        name: Name of the variation on the blue animal.
    """

    cfg: BlueAnimalHouseVariationCfg

    def __init__(
        self, task: HingedRoofVillageTask, cfg: BlueAnimalHouseVariationCfg | None = None, name: str = "home_house"
    ):
        super().__init__(cfg=cfg if cfg is not None else BlueAnimalHouseVariationCfg(), name=name)
        self._task = task

    def _realize_at_build_time(self) -> None:
        houses = list(self.cfg.houses)
        assert houses, "home_house.houses must not be empty."
        self._task.set_blue_animal_house(self.sampler.sample(1, houses)[0])
