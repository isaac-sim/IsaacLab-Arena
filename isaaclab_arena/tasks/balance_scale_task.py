# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Weigh an object of unknown mass on a passive balance scale with calibrated weights."""

from __future__ import annotations

import math
import numpy as np
import torch
from functools import partial
from typing import TYPE_CHECKING

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.balance_scale_usd import (
    BEAM_JOINT_NAME,
    DEFAULT_GEOMETRY,
    LEFT_PAN_BODY_NAME,
    LEFT_PAN_JOINT_NAME,
    RIGHT_PAN_BODY_NAME,
    RIGHT_PAN_JOINT_NAME,
)
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
from isaaclab_arena.utils.configclass import make_configclass

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.embodiments.gripper import Gripper
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv

_PAN_CENTER_P = (-DEFAULT_GEOMETRY.pan_forward_offset_m, 0.0, -DEFAULT_GEOMETRY.pan_hanger_length_m)
"""Center of a pan's floor top, in the pan frame P."""

_PAN_HALF_SIZE_M = DEFAULT_GEOMETRY.pan_floor_size_m / 2
_PAN_LOAD_HEIGHT_M = 0.12
"""Height above a pan floor up to which an object's origin counts as being in the pan."""


def _gravity_m_s2(env: IsaacLabArenaManagerBasedRLEnv) -> float:
    return abs(float(env.cfg.sim.gravity[2]))


def _object_mass_kg(env: IsaacLabArenaManagerBasedRLEnv, name: str) -> torch.Tensor:
    """Return an object's simulated mass per environment, including any mass variation applied at reset."""
    return env.scene[name].data.body_mass.torch.sum(dim=-1).to(env.device)


def _support_load_kg(env: IsaacLabArenaManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """Return the upward contact force a sensor's filtered body exerts on its sensed body, as a load in kilograms."""
    force_matrix_w = env.scene[sensor_name].data.force_matrix_w
    assert force_matrix_w is not None, f"Contact sensor '{sensor_name}' has no filtered force matrix."
    forces_W = force_matrix_w.torch
    assert forces_W.shape[1:] == (1, 1, 3), f"Contact sensor '{sensor_name}' must sense one body against one body."
    return forces_W[:, 0, 0, 2] / _gravity_m_s2(env)


def _pan_center_w(env: IsaacLabArenaManagerBasedRLEnv, scale_name: str, pan_body_name: str) -> torch.Tensor:
    """Return the world position of a pan's floor center."""
    T_W_P = env.arena_world.get_body_pose_w(scale_name, pan_body_name)
    t_P_center = torch.tensor(_PAN_CENTER_P, device=T_W_P.device, dtype=T_W_P.dtype).expand(T_W_P.shape[0], 3)
    return T_W_P[:, :3] + quat_apply(T_W_P[:, 3:], t_P_center)


def _object_in_pan(
    env: IsaacLabArenaManagerBasedRLEnv, scale_name: str, pan_body_name: str, object_name: str
) -> torch.Tensor:
    """Return whether an object's origin lies over a pan's floor and below its load height."""
    T_W_P = env.arena_world.get_body_pose_w(scale_name, pan_body_name)
    t_W_O = env.arena_world.get_pose_w(object_name)[:, :3]
    t_P_O = quat_apply_inverse(T_W_P[:, 3:], t_W_O - T_W_P[:, :3])
    offset_P = t_P_O - torch.tensor(_PAN_CENTER_P, device=t_P_O.device, dtype=t_P_O.dtype)
    return (
        (offset_P[:, 0].abs() < _PAN_HALF_SIZE_M)
        & (offset_P[:, 1].abs() < _PAN_HALF_SIZE_M)
        & (offset_P[:, 2] > -0.01)
        & (offset_P[:, 2] < _PAN_LOAD_HEIGHT_M)
    )


def _joint_velocity(env: IsaacLabArenaManagerBasedRLEnv, scale_name: str, joint_name: str) -> torch.Tensor:
    data = env.scene.articulations[scale_name].data
    return data.joint_vel.torch[:, data.joint_names.index(joint_name)]


def measured_object_rests_on_left_pan(
    env: IsaacLabArenaManagerBasedRLEnv,
    scale_name: str,
    measured_object_name: str,
    measured_object_sensor_name: str,
    support_tolerance_kg: float,
) -> torch.Tensor:
    """Check that the measured object lies in the left pan and the pan alone carries its weight."""
    in_pan = _object_in_pan(env, scale_name, LEFT_PAN_BODY_NAME, measured_object_name)
    load_kg = _support_load_kg(env, measured_object_sensor_name)
    mass_kg = _object_mass_kg(env, measured_object_name)
    return in_pan & ((load_kg - mass_kg).abs() <= support_tolerance_kg)


def counterweights_in_right_pan_kg(
    env: IsaacLabArenaManagerBasedRLEnv,
    scale_name: str,
    counterweight_names: list[str],
    counterweight_sensor_names: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mass of weights lying in the right pan, load the right pan carries from all weights), in kilograms."""
    in_pan_mass_kg = torch.zeros(env.num_envs, device=env.device)
    carried_load_kg = torch.zeros(env.num_envs, device=env.device)
    for weight_name, sensor_name in zip(counterweight_names, counterweight_sensor_names):
        in_pan = _object_in_pan(env, scale_name, RIGHT_PAN_BODY_NAME, weight_name)
        in_pan_mass_kg += in_pan.float() * _object_mass_kg(env, weight_name)
        carried_load_kg += _support_load_kg(env, sensor_name)
    return in_pan_mass_kg, carried_load_kg


def counterweight_released_in_right_pan(
    env: IsaacLabArenaManagerBasedRLEnv,
    scale_name: str,
    measured_object_name: str,
    measured_object_sensor_name: str,
    counterweight_names: list[str],
    counterweight_sensor_names: list[str],
    support_tolerance_kg: float,
) -> torch.Tensor:
    """Check that released weights rest in the right pan while the measured object still rests on the left pan.

    Every weight in the right pan must rest on it, directly or stacked, so the pan carries their full mass.
    """
    in_pan_mass_kg, carried_load_kg = counterweights_in_right_pan_kg(
        env, scale_name, counterweight_names, counterweight_sensor_names
    )
    weights_released = (in_pan_mass_kg > 0.0) & ((carried_load_kg - in_pan_mass_kg).abs() <= support_tolerance_kg)
    measured_object_in_place = measured_object_rests_on_left_pan(
        env, scale_name, measured_object_name, measured_object_sensor_name, support_tolerance_kg
    )
    return weights_released & measured_object_in_place


def scale_balanced_by_counterweights(
    env: IsaacLabArenaManagerBasedRLEnv,
    scale_name: str,
    measured_object_name: str,
    measured_object_sensor_name: str,
    counterweight_names: list[str],
    counterweight_sensor_names: list[str],
    support_tolerance_kg: float,
    mass_tolerance_kg: float,
    level_tolerance_rad: float,
    beam_speed_threshold_rad_s: float,
    pan_limit_margin_rad: float,
    gripper: Gripper | None,
    gripper_clearance_m: float,
) -> torch.Tensor:
    """Check that released weights in the right pan balance the measured object with the beam level and still.

    The beam must be within the level tolerance, away from its end stops, and slower than the speed threshold. The
    released weights' total mass must match the measured object's simulated mass within the mass tolerance, and the
    gripper, when given, must be farther than the clearance from both pans.
    """
    released = counterweight_released_in_right_pan(
        env,
        scale_name,
        measured_object_name,
        measured_object_sensor_name,
        counterweight_names,
        counterweight_sensor_names,
        support_tolerance_kg,
    )
    beam_angle_rad = env.arena_world.get_joint_position(scale_name, BEAM_JOINT_NAME)
    beam_speed_rad_s = _joint_velocity(env, scale_name, BEAM_JOINT_NAME).abs()
    level_and_still = (beam_angle_rad.abs() <= level_tolerance_rad) & (beam_speed_rad_s < beam_speed_threshold_rad_s)
    pan_limit_rad = math.radians(DEFAULT_GEOMETRY.pan_limit_deg) - pan_limit_margin_rad
    pans_free = (env.arena_world.get_joint_position(scale_name, LEFT_PAN_JOINT_NAME).abs() < pan_limit_rad) & (
        env.arena_world.get_joint_position(scale_name, RIGHT_PAN_JOINT_NAME).abs() < pan_limit_rad
    )
    in_pan_mass_kg, _ = counterweights_in_right_pan_kg(env, scale_name, counterweight_names, counterweight_sensor_names)
    mass_matches = (in_pan_mass_kg - _object_mass_kg(env, measured_object_name)).abs() <= mass_tolerance_kg
    balanced = released & level_and_still & pans_free & mass_matches
    if gripper is not None:
        t_W_gripper = gripper.get_position_w(env.arena_world)
        for pan_body_name in (LEFT_PAN_BODY_NAME, RIGHT_PAN_BODY_NAME):
            distance_m = torch.linalg.vector_norm(t_W_gripper - _pan_center_w(env, scale_name, pan_body_name), dim=-1)
            balanced &= distance_m > gripper_clearance_m
    return balanced


def measured_object_left_pan(
    env: IsaacLabArenaManagerBasedRLEnv, scale_name: str, measured_object_name: str
) -> torch.Tensor:
    """Return whether the measured object has left the left pan, for example knocked off by a swing."""
    return ~_object_in_pan(env, scale_name, LEFT_PAN_BODY_NAME, measured_object_name)


@register_task
class BalanceScaleTask(TaskBase):
    """Weigh an object of hidden mass by balancing it with calibrated weights on a passive balance scale.

    The measured object starts in the left pan. Success requires released weights in the right pan whose total mass
    matches the measured object's simulated mass, with the beam level and still and the gripper clear of both pans,
    for the required consecutive steps. Failure occurs when the measured object leaves the left pan. The measured
    object's mass is read from simulation, so a mass variation on it sets the hidden quantity per episode.

    Args:
        balance_scale: The balance scale articulation.
        measured_object: Object of unknown mass resting in the left pan.
        counterweights: Calibrated weights the robot may place in the right pan.
        episode_length_s: Maximum episode duration in seconds.
        task_description: Natural-language task instruction.
        mass_tolerance_kg: Maximum difference between the released weights' mass and the measured object's mass.
        support_tolerance_kg: Maximum difference between a pan's carried load and the mass it should carry; rejects
            loads held or pressed by the robot.
        level_tolerance_deg: Maximum beam angle from level, in degrees.
        beam_speed_threshold_rad_s: Beam angular speed must be below this, in rad/s.
        pan_limit_margin_deg: Minimum margin of each pan joint from its limit, in degrees.
        gripper_clearance_m: Minimum gripper distance from each pan's floor center, in meters.
        balance_hold_steps: Consecutive control steps the balanced state must hold.
    """

    def __init__(
        self,
        balance_scale: Asset,
        measured_object: Asset,
        counterweights: list[Asset],
        episode_length_s: float | None = None,
        task_description: str | None = None,
        mass_tolerance_kg: float = 0.002,
        support_tolerance_kg: float = 0.005,
        level_tolerance_deg: float = 1.0,
        beam_speed_threshold_rad_s: float = 0.015,
        pan_limit_margin_deg: float = 1.0,
        gripper_clearance_m: float = 0.08,
        balance_hold_steps: int = 45,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        assert counterweights, "BalanceScaleTask needs at least one counterweight."
        assert balance_hold_steps > 0, "balance_hold_steps must be positive."
        self.balance_scale = balance_scale
        self.measured_object = measured_object
        self.counterweights = list(counterweights)
        self.mass_tolerance_kg = mass_tolerance_kg
        self.support_tolerance_kg = support_tolerance_kg
        self.level_tolerance_rad = math.radians(level_tolerance_deg)
        self.beam_speed_threshold_rad_s = beam_speed_threshold_rad_s
        self.pan_limit_margin_rad = math.radians(pan_limit_margin_deg)
        self.gripper_clearance_m = gripper_clearance_m
        self.balance_hold_steps = balance_hold_steps
        self.gripper: Gripper | None = None
        self.measured_object_sensor_name = f"contact_sensor_{measured_object.name}_on_left_pan"
        self.counterweight_sensor_names = [
            f"contact_sensor_{weight.name}_on_right_pan" for weight in self.counterweights
        ]
        if self.task_description is None:
            self.task_description = (
                f"The {measured_object.name} rests on the left pan of the balance scale. Place calibrated weights on"
                " the right pan until the scale balances, release them and move the gripper away from the pans."
            )

    def configure_for_embodiment(self, embodiment: EmbodimentBase) -> None:
        """Use the embodiment's gripper, when it has one, for the gripper-clearance check."""
        self.gripper = embodiment.gripper

    def apply_reachability_constraints(self) -> None:
        """The robot must reach every counterweight."""
        self._apply_reachability_constraints(self.counterweights)

    def get_scene_cfg(self):
        scale_prim_path = self.balance_scale.get_prim_path()
        sensors = [(
            self.measured_object_sensor_name,
            ContactSensorCfg(
                prim_path=self.measured_object.get_contact_sensor_prim_path(),
                filter_prim_paths_expr=[f"{scale_prim_path}/{LEFT_PAN_BODY_NAME}"],
            ),
        )]
        for weight, sensor_name in zip(self.counterweights, self.counterweight_sensor_names):
            sensors.append((
                sensor_name,
                ContactSensorCfg(
                    prim_path=weight.get_contact_sensor_prim_path(),
                    filter_prim_paths_expr=[f"{scale_prim_path}/{RIGHT_PAN_BODY_NAME}"],
                ),
            ))
        scene_cfg_type = make_configclass("SceneCfg", [(name, ContactSensorCfg, cfg) for name, cfg in sensors])
        return scene_cfg_type()

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("BalanceScaleTask has no Mimic configuration.")

    def get_termination_cfg(self) -> TaskTerminationCfg:
        common = dict(
            scale_name=self.balance_scale.name,
            measured_object_name=self.measured_object.name,
            measured_object_sensor_name=self.measured_object_sensor_name,
            counterweight_names=[weight.name for weight in self.counterweights],
            counterweight_sensor_names=self.counterweight_sensor_names,
            support_tolerance_kg=self.support_tolerance_kg,
        )
        # Bind the gripper lazily: configure_for_embodiment runs after the task is constructed.
        task = self

        def balanced(env: IsaacLabArenaManagerBasedRLEnv) -> torch.Tensor:
            return scale_balanced_by_counterweights(
                env,
                **common,
                mass_tolerance_kg=task.mass_tolerance_kg,
                level_tolerance_rad=task.level_tolerance_rad,
                beam_speed_threshold_rad_s=task.beam_speed_threshold_rad_s,
                pan_limit_margin_rad=task.pan_limit_margin_rad,
                gripper=task.gripper,
                gripper_clearance_m=task.gripper_clearance_m,
            )

        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="weigh_on_balance_scale",
                    predicate_sequence=[
                        (partial(counterweight_released_in_right_pan, **common), 0.3),
                        (
                            TrueForConsecutiveStepsCfg(predicate=balanced, required_steps=self.balance_hold_steps),
                            0.7,
                        ),
                    ],
                ),
            ],
            failures={
                "measured_object_left_pan": TerminationTermCfg(
                    func=measured_object_left_pan,
                    params={"scale_name": self.balance_scale.name, "measured_object_name": self.measured_object.name},
                )
            },
        )

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(lookat_object=self.balance_scale, offset=np.array([-0.9, -0.6, 0.5]))

    @classmethod
    def success_state_transition(cls, balance_scale: str, **_) -> TaskTransition:
        """Success changes the scale's state (loaded and level); the env graph has no node state for it yet."""
        return TaskTransition(subject=balance_scale)
