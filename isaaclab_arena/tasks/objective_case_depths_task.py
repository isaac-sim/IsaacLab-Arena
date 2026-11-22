# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pack cylinders into an objective case whose look-alike sockets have hidden depths, then close the lid."""

from __future__ import annotations

import math
import numpy as np
import torch
from functools import partial
from typing import TYPE_CHECKING

from isaaclab.envs.common import ViewerCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.objective_case_depths_assets import LID_JOINT_NAME, ObjectiveCaseGeometry
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
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


def _cylinder_socket_state(
    env: IsaacLabArenaManagerBasedRLEnv,
    case_name: str,
    geometry: ObjectiveCaseGeometry,
    cylinder_name: str,
    cylinder_length_m: float,
    sensor_name: str,
    tilt_tolerance_rad: float,
    support_fraction: float,
    linear_speed_threshold_m_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (socket membership per socket, top-face height above the rim) for one cylinder.

    A cylinder is in socket k when its center lies over the socket opening and below the rim, its axis is within the
    tilt tolerance of the case's Z axis, it moves slower than the speed threshold, and contacts carry at least the
    support fraction of its weight (it is not held or lifted). Membership has shape (num_envs, num_sockets).
    """
    T_E_C = env.arena_world.get_pose_e(case_name)
    T_E_O = env.arena_world.get_pose_e(cylinder_name)
    q_E_C = T_E_C[:, 3:]
    t_C_O = quat_apply_inverse(q_E_C, T_E_O[:, :3] - T_E_C[:, :3])
    axis_O = torch.zeros_like(t_C_O)
    axis_O[:, 2] = 1.0
    axis_C = quat_apply_inverse(q_E_C, quat_apply(T_E_O[:, 3:], axis_O))
    # The cylinder is symmetric, so either end may face up.
    top_z_C = t_C_O[:, 2] + 0.5 * cylinder_length_m * axis_C[:, 2].abs()
    top_above_rim_m = top_z_C - geometry.case_height_m

    upright = axis_C[:, 2].abs() >= math.cos(tilt_tolerance_rad)
    slow = torch.linalg.vector_norm(env.arena_world.get_root_linear_velocity_w(cylinder_name), dim=-1)
    slow = slow < linear_speed_threshold_m_s
    mass_kg = env.scene[cylinder_name].data.body_mass.torch.sum(dim=-1).to(env.device)
    load_kg = env.scene[sensor_name].data.net_forces_w.torch[:, 0, 2] / abs(float(env.cfg.sim.gravity[2]))
    supported = load_kg >= support_fraction * mass_kg
    below_rim = t_C_O[:, 2] < geometry.case_height_m
    half_width = geometry.socket_width_m / 2
    membership = []
    for socket_index in range(geometry.num_sockets):
        over_opening = (t_C_O[:, 0].abs() < half_width) & (
            (t_C_O[:, 1] - geometry.socket_center_y_m(socket_index)).abs() < half_width
        )
        membership.append(over_opening & below_rim & upright & slow & supported)
    return torch.stack(membership, dim=-1), top_above_rim_m


def _socket_states(
    env: IsaacLabArenaManagerBasedRLEnv,
    case_name: str,
    geometry: ObjectiveCaseGeometry,
    cylinder_names: list[str],
    cylinder_lengths_m: list[float],
    sensor_names: list[str],
    tilt_tolerance_rad: float,
    support_fraction: float,
    linear_speed_threshold_m_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (membership with shape (num_envs, num_cylinders, num_sockets), top heights (num_envs, num_cylinders))."""
    memberships = []
    tops = []
    for cylinder_name, length_m, sensor_name in zip(cylinder_names, cylinder_lengths_m, sensor_names):
        membership, top_above_rim_m = _cylinder_socket_state(
            env,
            case_name,
            geometry,
            cylinder_name,
            length_m,
            sensor_name,
            tilt_tolerance_rad,
            support_fraction,
            linear_speed_threshold_m_s,
        )
        memberships.append(membership)
        tops.append(top_above_rim_m)
    return torch.stack(memberships, dim=1), torch.stack(tops, dim=1)


def all_cylinders_seated(env: IsaacLabArenaManagerBasedRLEnv, **state_kwargs) -> torch.Tensor:
    """Check that every cylinder rests upright in its own socket, released and still, at any height."""
    membership, _ = _socket_states(env, **state_kwargs)
    each_seated = membership.any(dim=-1).all(dim=-1)
    one_per_socket = (membership.sum(dim=1) <= 1).all(dim=-1)
    return each_seated & one_per_socket


def all_cylinders_flush(env: IsaacLabArenaManagerBasedRLEnv, flush_tolerance_m: float, **state_kwargs) -> torch.Tensor:
    """Check that every cylinder is seated in its own socket with its top face within tolerance of the rim."""
    membership, top_above_rim_m = _socket_states(env, **state_kwargs)
    each_flush = (membership.any(dim=-1) & (top_above_rim_m.abs() <= flush_tolerance_m)).all(dim=-1)
    one_per_socket = (membership.sum(dim=1) <= 1).all(dim=-1)
    return each_flush & one_per_socket


def case_packed_and_closed(
    env: IsaacLabArenaManagerBasedRLEnv,
    lid_closed_tolerance_rad: float,
    flush_tolerance_m: float,
    **state_kwargs,
) -> torch.Tensor:
    """Check that every cylinder sits flush in its own socket and the lid is closed."""
    lid_angle_rad = env.arena_world.get_joint_position(state_kwargs["case_name"], LID_JOINT_NAME)
    return all_cylinders_flush(env, flush_tolerance_m, **state_kwargs) & (lid_angle_rad <= lid_closed_tolerance_rad)


@register_task
class ObjectiveCaseDepthsTask(TaskBase):
    """Pack cylinders of different lengths into look-alike sockets of hidden depth and close the case lid.

    The socket depths are a permutation of the cylinder lengths, so only one assignment leaves every cylinder flush
    with the rim; any other leaves a cylinder standing proud (the lid cannot close) or sunk below the rim. Progress
    has three stages: every cylinder seated in its own socket, every cylinder flush, and then flush with the lid
    closed for the required consecutive steps. A seated cylinder must be upright, still, and carried by contacts.

    Args:
        objective_case: The objective case articulation.
        cylinders: Cylinders to pack, one per socket.
        episode_length_s: Maximum episode duration in seconds.
        task_description: Natural-language task instruction.
        flush_tolerance_m: Maximum distance between a cylinder's top face and the rim.
        lid_closed_tolerance_deg: Maximum lid angle, in degrees, that counts as closed.
        tilt_tolerance_deg: Maximum angle between a seated cylinder's axis and the case's vertical, in degrees.
        support_fraction: Minimum share of a cylinder's weight its contacts must carry; rejects held cylinders.
        linear_speed_threshold_m_s: A seated cylinder must move slower than this.
        closed_hold_steps: Consecutive control steps the packed-and-closed state must hold.
    """

    def __init__(
        self,
        objective_case: Asset,
        cylinders: list[Asset],
        episode_length_s: float | None = None,
        task_description: str | None = None,
        flush_tolerance_m: float = 0.004,
        lid_closed_tolerance_deg: float = 3.0,
        tilt_tolerance_deg: float = 8.0,
        support_fraction: float = 0.5,
        linear_speed_threshold_m_s: float = 0.02,
        closed_hold_steps: int = 15,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        assert cylinders, "ObjectiveCaseDepthsTask needs at least one cylinder."
        assert closed_hold_steps > 0, "closed_hold_steps must be positive."
        self.objective_case = objective_case
        self.cylinders = list(cylinders)
        assert len(self.cylinders) <= objective_case.geometry.num_sockets, "More cylinders than sockets."
        self.flush_tolerance_m = flush_tolerance_m
        self.lid_closed_tolerance_rad = math.radians(lid_closed_tolerance_deg)
        self.tilt_tolerance_rad = math.radians(tilt_tolerance_deg)
        self.support_fraction = support_fraction
        self.linear_speed_threshold_m_s = linear_speed_threshold_m_s
        self.closed_hold_steps = closed_hold_steps
        self.cylinder_sensor_names = [f"contact_sensor_{cylinder.name}" for cylinder in self.cylinders]
        if self.task_description is None:
            self.task_description = "Pack all four objective cylinders into the case and close the lid."

    def apply_reachability_constraints(self) -> None:
        """The robot must reach every cylinder."""
        self._apply_reachability_constraints(self.cylinders)

    def get_scene_cfg(self):
        sensors = [
            (sensor_name, ContactSensorCfg(prim_path=cylinder.get_contact_sensor_prim_path()))
            for cylinder, sensor_name in zip(self.cylinders, self.cylinder_sensor_names)
        ]
        scene_cfg_type = make_configclass("SceneCfg", [(name, ContactSensorCfg, cfg) for name, cfg in sensors])
        return scene_cfg_type()

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("ObjectiveCaseDepthsTask has no Mimic configuration.")

    def _state_kwargs(self) -> dict:
        # Read the geometry when the termination config is built: a socket-depth variation may have replaced it.
        return dict(
            case_name=self.objective_case.name,
            geometry=self.objective_case.geometry,
            cylinder_names=[cylinder.name for cylinder in self.cylinders],
            cylinder_lengths_m=[cylinder.length_m for cylinder in self.cylinders],
            sensor_names=self.cylinder_sensor_names,
            tilt_tolerance_rad=self.tilt_tolerance_rad,
            support_fraction=self.support_fraction,
            linear_speed_threshold_m_s=self.linear_speed_threshold_m_s,
        )

    def get_termination_cfg(self) -> TaskTerminationCfg:
        state_kwargs = self._state_kwargs()
        packed_and_closed = partial(
            case_packed_and_closed,
            lid_closed_tolerance_rad=self.lid_closed_tolerance_rad,
            flush_tolerance_m=self.flush_tolerance_m,
            **state_kwargs,
        )
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="pack_objective_case",
                    predicate_sequence=[
                        (partial(all_cylinders_seated, **state_kwargs), 0.3),
                        (partial(all_cylinders_flush, flush_tolerance_m=self.flush_tolerance_m, **state_kwargs), 0.3),
                        (
                            TrueForConsecutiveStepsCfg(
                                predicate=packed_and_closed, required_steps=self.closed_hold_steps
                            ),
                            0.4,
                        ),
                    ],
                ),
            ],
        )

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(lookat_object=self.objective_case, offset=np.array([-0.42, -0.42, 0.36]))

    @classmethod
    def success_state_transition(cls, objective_case: str, **_) -> TaskTransition:
        """Success changes the case's state (packed and closed); the env graph has no node state for it yet."""
        return TaskTransition(subject=objective_case)
