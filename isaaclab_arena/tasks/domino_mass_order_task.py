# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stand look-alike dominoes of hidden mass in a line so one push topples them all and rings a bell."""

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


def domino_tilt_rad(env: IsaacLabArenaManagerBasedRLEnv, domino_name: str) -> torch.Tensor:
    """Return the angle between a domino's long (local Z) axis and the world Z axis."""
    q_W_D = env.arena_world.get_pose_w(domino_name)[:, 3:]
    z_D = torch.tensor([0.0, 0.0, 1.0], device=q_W_D.device, dtype=q_W_D.dtype).expand(q_W_D.shape[0], 3)
    z_W = quat_apply(q_W_D, z_D)
    return torch.acos(z_W[:, 2].clamp(-1.0, 1.0))


def domino_over_runway(
    env: IsaacLabArenaManagerBasedRLEnv,
    domino_name: str,
    runway_name: str,
    runway_half_extent_m: tuple[float, float],
    margin_m: float,
) -> torch.Tensor:
    """Return whether a domino's center lies over the runway, extended by a margin on every side."""
    T_W_R = env.arena_world.get_pose_w(runway_name)
    t_W_D = env.arena_world.get_pose_w(domino_name)[:, :3]
    t_R_D = quat_apply_inverse(T_W_R[:, 3:], t_W_D - T_W_R[:, :3])
    return (t_R_D[:, 0].abs() < runway_half_extent_m[0] + margin_m) & (
        t_R_D[:, 1].abs() < runway_half_extent_m[1] + margin_m
    )


def dominoes_standing_on_runway(
    env: IsaacLabArenaManagerBasedRLEnv,
    domino_names: list[str],
    runway_name: str,
    runway_half_extent_m: tuple[float, float],
    runway_top_z_m: float,
    domino_half_height_m: float,
    upright_tilt_rad: float,
    speed_threshold_m_s: float,
) -> torch.Tensor:
    """Check that every domino stands upright and still on the runway."""
    standing = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for domino_name in domino_names:
        t_E_D = env.arena_world.get_pose_e(domino_name)[:, :3]
        on_top = (t_E_D[:, 2] - runway_top_z_m - domino_half_height_m).abs() < 0.01
        speed = torch.linalg.vector_norm(env.arena_world.get_root_linear_velocity_w(domino_name), dim=-1)
        standing &= (
            domino_over_runway(env, domino_name, runway_name, runway_half_extent_m, margin_m=0.0)
            & on_top
            & (domino_tilt_rad(env, domino_name) < upright_tilt_rad)
            & (speed < speed_threshold_m_s)
        )
    return standing


def bell_struck(
    env: IsaacLabArenaManagerBasedRLEnv, bell_sensor_names: list[str], force_threshold_n: float
) -> torch.Tensor:
    """Return whether any domino presses on the bell with more than the threshold force."""
    struck = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for sensor_name in bell_sensor_names:
        force_matrix_w = env.scene[sensor_name].data.force_matrix_w
        assert force_matrix_w is not None, f"Contact sensor '{sensor_name}' has no filtered force matrix."
        forces_W = force_matrix_w.torch.reshape(env.num_envs, -1, 3)
        struck |= torch.linalg.vector_norm(forces_W, dim=-1).amax(dim=-1) > force_threshold_n
    return struck


def cascade_rang_bell(
    env: IsaacLabArenaManagerBasedRLEnv,
    domino_names: list[str],
    runway_name: str,
    runway_half_extent_m: tuple[float, float],
    runway_margin_m: float,
    toppled_tilt_rad: float,
    bell_sensor_names: list[str],
    force_threshold_n: float,
) -> torch.Tensor:
    """Check that every domino has toppled over the runway and a domino rests against the bell."""
    rang = bell_struck(env, bell_sensor_names, force_threshold_n)
    for domino_name in domino_names:
        rang &= domino_tilt_rad(env, domino_name) > toppled_tilt_rad
        rang &= domino_over_runway(env, domino_name, runway_name, runway_half_extent_m, runway_margin_m)
    return rang


@register_task
class DominoMassOrderTask(TaskBase):
    """Line up look-alike dominoes on a runway so that one push topples them all and rings a bell.

    Some dominoes are heavy and some light; a light domino falling on a heavy one cannot topple it, so the heavy
    dominoes must stand first. Masses are read only by the physics: set them with each domino's ``mass_kg`` parameter
    or its mass variation. Success is staged: all dominoes standing upright on the runway, then every domino toppled
    over the runway with one resting against the bell, held for the required consecutive steps.

    Args:
        dominoes: The dominoes to line up.
        runway: Static strip the dominoes must stand on.
        bell: Kinematic bell at the end of the runway.
        episode_length_s: Maximum episode duration in seconds.
        task_description: Natural-language task instruction.
        upright_tilt_deg: Maximum tilt of a standing domino, in degrees.
        toppled_tilt_deg: Minimum tilt of a toppled domino, in degrees.
        runway_margin_m: Distance beyond the runway edges a toppled domino's center may lie.
        force_threshold_n: Contact force between a domino and the bell that counts as ringing it, in newtons.
        speed_threshold_m_s: Standing dominoes must move slower than this.
        hold_steps: Consecutive control steps the rung state must hold.
    """

    def __init__(
        self,
        dominoes: list[Asset],
        runway: Asset,
        bell: Asset,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        upright_tilt_deg: float = 8.0,
        toppled_tilt_deg: float = 25.0,
        runway_margin_m: float = 0.04,
        force_threshold_n: float = 0.02,
        speed_threshold_m_s: float = 0.02,
        hold_steps: int = 30,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        assert len(dominoes) >= 2, "DominoMassOrderTask needs at least two dominoes."
        assert hold_steps > 0, "hold_steps must be positive."
        self.dominoes = list(dominoes)
        self.runway = runway
        self.bell = bell
        self.upright_tilt_rad = math.radians(upright_tilt_deg)
        self.toppled_tilt_rad = math.radians(toppled_tilt_deg)
        self.runway_margin_m = runway_margin_m
        self.force_threshold_n = force_threshold_n
        self.speed_threshold_m_s = speed_threshold_m_s
        self.hold_steps = hold_steps
        self.bell_sensor_names = [f"contact_sensor_{domino.name}_on_bell" for domino in self.dominoes]
        if self.task_description is None:
            self.task_description = (
                "Stand the five blocks in a line on the runway so that one push on the first block rings the bell,"
                " then make it ring."
            )

    def apply_reachability_constraints(self) -> None:
        """The robot must reach every domino."""
        self._apply_reachability_constraints(self.dominoes)

    def get_scene_cfg(self):
        bell_body_path = self.bell.get_contact_sensor_prim_path()
        sensors = []
        for domino, sensor_name in zip(self.dominoes, self.bell_sensor_names):
            sensors.append((
                sensor_name,
                ContactSensorCfg(
                    prim_path=domino.get_contact_sensor_prim_path(), filter_prim_paths_expr=[bell_body_path]
                ),
            ))
        scene_cfg_type = make_configclass("SceneCfg", [(name, ContactSensorCfg, cfg) for name, cfg in sensors])
        return scene_cfg_type()

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("DominoMassOrderTask has no Mimic configuration.")

    def _runway_geometry(self) -> tuple[tuple[float, float], float]:
        """Return the runway's half extent (x, y) and its top height in the environment frame."""
        T_E_R = self.runway._get_initial_pose_as_pose()
        assert T_E_R is not None, "The runway needs a fixed initial pose."
        half_extent_m = (self.runway.width_m / 2, self.runway.length_m / 2)
        top_z_m = T_E_R.position_xyz[2] + self.runway.thickness_m / 2
        return half_extent_m, top_z_m

    def get_termination_cfg(self) -> TaskTerminationCfg:
        from isaaclab_arena.assets.domino_mass_order_assets import DOMINO_SIZE_M

        domino_names = [domino.name for domino in self.dominoes]
        runway_half_extent_m, runway_top_z_m = self._runway_geometry()
        standing = partial(
            dominoes_standing_on_runway,
            domino_names=domino_names,
            runway_name=self.runway.name,
            runway_half_extent_m=runway_half_extent_m,
            runway_top_z_m=runway_top_z_m,
            domino_half_height_m=DOMINO_SIZE_M[2] / 2,
            upright_tilt_rad=self.upright_tilt_rad,
            speed_threshold_m_s=self.speed_threshold_m_s,
        )
        rang = partial(
            cascade_rang_bell,
            domino_names=domino_names,
            runway_name=self.runway.name,
            runway_half_extent_m=runway_half_extent_m,
            runway_margin_m=self.runway_margin_m,
            toppled_tilt_rad=self.toppled_tilt_rad,
            bell_sensor_names=self.bell_sensor_names,
            force_threshold_n=self.force_threshold_n,
        )
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="ring_bell_with_domino_cascade",
                    predicate_sequence=[
                        (standing, 0.4),
                        (TrueForConsecutiveStepsCfg(predicate=rang, required_steps=self.hold_steps), 0.6),
                    ],
                ),
            ],
        )

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(lookat_object=self.runway, offset=np.array([-0.55, -0.55, 0.45]))

    @classmethod
    def success_state_transition(cls, bell: str, **_) -> TaskTransition:
        """Success rings the bell; the env graph has no node state for it."""
        return TaskTransition(subject=bell)
