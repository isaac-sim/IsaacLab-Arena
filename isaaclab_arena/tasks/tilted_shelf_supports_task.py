# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Display look-alike samples of hidden friction on a tilted shelf so that each stays put in its own bay."""

from __future__ import annotations

import math
import numpy as np
import torch
from functools import partial
from typing import TYPE_CHECKING

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.assets.tilted_shelf_supports_assets import DEFAULT_TILTED_SHELF_GEOMETRY
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

_SHELF = DEFAULT_TILTED_SHELF_GEOMETRY


def sample_position_on_board(env: IsaacLabArenaManagerBasedRLEnv, shelf_name: str, sample_name: str) -> torch.Tensor:
    """Return a sample's origin in the shelf's board frame as (u up the slope, v across, n above the surface)."""
    T_W_A = env.arena_world.get_pose_w(shelf_name)
    t_W_S = env.arena_world.get_pose_w(sample_name)[:, :3]
    t_A_S = quat_apply_inverse(T_W_A[:, 3:], t_W_S - T_W_A[:, :3])
    x_A = t_A_S[:, 0]
    z_above_edge = t_A_S[:, 2] - _SHELF.front_edge_height_m
    sin_t, cos_t = math.sin(_SHELF.tilt_rad), math.cos(_SHELF.tilt_rad)
    u = x_A * cos_t + z_above_edge * sin_t
    n = -x_A * sin_t + z_above_edge * cos_t
    return torch.stack((u, t_A_S[:, 1], n), dim=-1)


def sample_bay_index(
    env: IsaacLabArenaManagerBasedRLEnv, shelf_name: str, sample_name: str, max_height_above_board_m: float
) -> torch.Tensor:
    """Return the index of the bay a sample's origin lies in, or -1 when it lies in none."""
    t_B = sample_position_on_board(env, shelf_name, sample_name)
    on_board = (
        (t_B[:, 0] >= 0.0)
        & (t_B[:, 0] <= _SHELF.board_length_m)
        & (t_B[:, 2] >= 0.0)
        & (t_B[:, 2] <= max_height_above_board_m)
    )
    bay_index = torch.full((env.num_envs,), -1, dtype=torch.long, device=t_B.device)
    for index in range(_SHELF.bay_count):
        in_bay = on_board & ((t_B[:, 1] - _SHELF.bay_center_v_m(index)).abs() <= _SHELF.bay_width_m / 2)
        bay_index = torch.where(in_bay, torch.full_like(bay_index, index), bay_index)
    return bay_index


def samples_in_separate_bays(
    env: IsaacLabArenaManagerBasedRLEnv,
    shelf_name: str,
    sample_names: list[str],
    max_height_above_board_m: float,
) -> torch.Tensor:
    """Check that every sample lies in a bay of the shelf and no two samples share a bay."""
    bay_indices = torch.stack(
        [sample_bay_index(env, shelf_name, name, max_height_above_board_m) for name in sample_names], dim=-1
    )
    all_in_bays = (bay_indices >= 0).all(dim=-1)
    separate = torch.ones_like(all_in_bays)
    for first in range(len(sample_names)):
        for second in range(first + 1, len(sample_names)):
            separate &= bay_indices[:, first] != bay_indices[:, second]
    return all_in_bays & separate


def samples_stay_put_in_separate_bays(
    env: IsaacLabArenaManagerBasedRLEnv,
    shelf_name: str,
    sample_names: list[str],
    max_height_above_board_m: float,
    rest_linear_speed_m_s: float,
    rest_angular_speed_rad_s: float,
    gripper: Gripper | None,
    gripper_clearance_m: float,
) -> torch.Tensor:
    """Check that the samples lie in separate bays, are at rest, and, when a gripper is given, are clear of it."""
    result = samples_in_separate_bays(env, shelf_name, sample_names, max_height_above_board_m)
    world = env.arena_world
    for name in sample_names:
        result &= torch.linalg.vector_norm(world.get_root_linear_velocity_w(name), dim=-1) <= rest_linear_speed_m_s
        result &= torch.linalg.vector_norm(world.get_root_angular_velocity_w(name), dim=-1) <= rest_angular_speed_rad_s
        if gripper is not None:
            t_W_gripper = gripper.get_position_w(world)
            distance_m = torch.linalg.vector_norm(world.get_pose_w(name)[:, :3] - t_W_gripper, dim=-1)
            result &= distance_m > gripper_clearance_m
    return result


def sample_fell_off_table(
    env: IsaacLabArenaManagerBasedRLEnv, shelf_name: str, sample_names: list[str], drop_m: float
) -> torch.Tensor:
    """Return whether any sample is more than drop_m below the shelf's base, i.e. off the table."""
    base_z = env.arena_world.get_pose_w(shelf_name)[:, 2]
    fell = torch.zeros(env.num_envs, dtype=torch.bool, device=base_z.device)
    for name in sample_names:
        fell |= env.arena_world.get_pose_w(name)[:, 2] < base_z - drop_m
    return fell


@register_task
class TiltedShelfSupportsTask(TaskBase):
    """Display look-alike samples on a tilted shelf, one per bay, so that each stays put.

    Each sample's friction is hidden: a slick one slides on the bare shelf and on the grip mat and needs the stop pin, a
    medium one holds only on the grip mat, and a grippy one holds on the bare shelf. Success requires every sample to
    lie in its own bay, at rest and clear of the gripper, for the required consecutive steps. Failure occurs when a
    sample falls off the table.

    Args:
        shelf: The tilted display shelf.
        samples: Samples to display, one per bay.
        grip_mat: Grip mat the robot may lay in a bay.
        stop_pin: Stop pin the robot may insert in a bay's pocket.
        episode_length_s: Maximum episode duration in seconds.
        task_description: Natural-language task instruction.
        max_height_above_board_m: Highest a sample's origin may be above the board surface and still count as in a bay.
        rest_linear_speed_m_s: Sample linear speed must be at most this, in m/s.
        rest_angular_speed_rad_s: Sample angular speed must be at most this, in rad/s.
        gripper_clearance_m: Minimum gripper distance from each sample's origin, in meters.
        hold_steps: Consecutive control steps the samples must stay put.
        fall_drop_m: Distance below the shelf base at which a sample counts as fallen off the table, in meters.
    """

    def __init__(
        self,
        shelf: Asset,
        samples: list[Asset],
        grip_mat: Asset,
        stop_pin: Asset,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        max_height_above_board_m: float = 0.06,
        rest_linear_speed_m_s: float = 0.01,
        rest_angular_speed_rad_s: float = 0.2,
        gripper_clearance_m: float = 0.06,
        hold_steps: int = 45,
        fall_drop_m: float = 0.15,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        assert 0 < len(samples) <= _SHELF.bay_count, f"TiltedShelfSupportsTask needs 1 to {_SHELF.bay_count} samples."
        assert hold_steps > 0, "hold_steps must be positive."
        self.shelf = shelf
        self.samples = list(samples)
        self.grip_mat = grip_mat
        self.stop_pin = stop_pin
        self.max_height_above_board_m = max_height_above_board_m
        self.rest_linear_speed_m_s = rest_linear_speed_m_s
        self.rest_angular_speed_rad_s = rest_angular_speed_rad_s
        self.gripper_clearance_m = gripper_clearance_m
        self.hold_steps = hold_steps
        self.fall_drop_m = fall_drop_m
        self.gripper: Gripper | None = None
        if self.task_description is None:
            self.task_description = (
                "Display all three samples on the tilted shelf, one per bay, so that they stay put. A grip mat and a"
                " stop pin are on the table."
            )

    def configure_for_embodiment(self, embodiment: EmbodimentBase) -> None:
        """Use the embodiment's gripper, when it has one, for the gripper-clearance check."""
        self.gripper = embodiment.gripper

    def apply_reachability_constraints(self) -> None:
        """The robot must reach every sample, the grip mat and the stop pin."""
        self._apply_reachability_constraints([*self.samples, self.grip_mat, self.stop_pin])

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("TiltedShelfSupportsTask has no Mimic configuration.")

    def get_termination_cfg(self) -> TaskTerminationCfg:
        sample_names = [sample.name for sample in self.samples]
        in_bays = partial(
            samples_in_separate_bays,
            shelf_name=self.shelf.name,
            sample_names=sample_names,
            max_height_above_board_m=self.max_height_above_board_m,
        )
        # Bind the gripper lazily: configure_for_embodiment runs after the task is constructed.
        task = self

        def stay_put(env: IsaacLabArenaManagerBasedRLEnv) -> torch.Tensor:
            return samples_stay_put_in_separate_bays(
                env,
                shelf_name=task.shelf.name,
                sample_names=sample_names,
                max_height_above_board_m=task.max_height_above_board_m,
                rest_linear_speed_m_s=task.rest_linear_speed_m_s,
                rest_angular_speed_rad_s=task.rest_angular_speed_rad_s,
                gripper=task.gripper,
                gripper_clearance_m=task.gripper_clearance_m,
            )

        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="samples_stay_put_on_tilted_shelf",
                    predicate_sequence=[
                        (in_bays, 0.3),
                        (TrueForConsecutiveStepsCfg(predicate=stay_put, required_steps=self.hold_steps), 0.7),
                    ],
                ),
            ],
            failures={
                "sample_fell_off_table": TerminationTermCfg(
                    func=sample_fell_off_table,
                    params={"shelf_name": self.shelf.name, "sample_names": sample_names, "drop_m": self.fall_drop_m},
                )
            },
        )

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(lookat_object=self.shelf, offset=np.array([-0.62, -0.42, 0.42]))

    @classmethod
    def success_state_transition(cls, shelf: str, **_) -> TaskTransition:
        """Success changes the shelf's state (loaded bays); the env graph has no node state for it yet."""
        return TaskTransition(subject=shelf)
