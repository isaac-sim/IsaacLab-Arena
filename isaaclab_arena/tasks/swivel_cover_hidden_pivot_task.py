# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Take a core out of a fixture whose cover swivels on a hidden corner pivot, stand it on a pad, leave the cover shut."""

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
from isaaclab_arena.assets.swivel_cover_hidden_pivot_assets import COVER_JOINT_NAME
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


def _position_in_frame(env: IsaacLabArenaManagerBasedRLEnv, frame_name: str, object_name: str) -> torch.Tensor:
    """Return an object's origin in the root frame of another scene entity."""
    T_W_F = env.arena_world.get_pose_w(frame_name)
    t_W_O = env.arena_world.get_position_w(object_name)
    return quat_apply_inverse(T_W_F[:, 3:], t_W_O - T_W_F[:, :3])


def core_out_of_fixture(
    env: IsaacLabArenaManagerBasedRLEnv, fixture_name: str, core_name: str, footprint_half_size_m: float
) -> torch.Tensor:
    """Return whether the core's origin lies outside the fixture's square footprint."""
    t_F_C = _position_in_frame(env, fixture_name, core_name)
    return (t_F_C[:, 0].abs() > footprint_half_size_m) | (t_F_C[:, 1].abs() > footprint_half_size_m)


def core_upright_on_pad(
    env: IsaacLabArenaManagerBasedRLEnv,
    core_name: str,
    pad_name: str,
    core_sensor_name: str,
    pad_half_size_m: float,
    core_rest_height_m: float,
    height_tolerance_m: float,
    tilt_tolerance_rad: float,
    support_fraction: float,
) -> torch.Tensor:
    """Return whether the core stands upright on the pad and the pad carries most of its weight.

    The core's origin must lie over the pad within its half size, at its resting height above the pad origin, its
    axis must be within the tilt tolerance of the pad normal, and the pad must push up with at least support_fraction
    of the core's weight, so a core held by the robot does not count.
    """
    t_P_C = _position_in_frame(env, pad_name, core_name)
    over_pad = (t_P_C[:, 0].abs() <= pad_half_size_m) & (t_P_C[:, 1].abs() <= pad_half_size_m)
    at_rest_height = (t_P_C[:, 2] - core_rest_height_m).abs() <= height_tolerance_m
    q_W_C = env.arena_world.get_pose_w(core_name)[:, 3:]
    up_C = torch.tensor([0.0, 0.0, 1.0], device=q_W_C.device).expand(q_W_C.shape[0], 3)
    upright = quat_apply(q_W_C, up_C)[:, 2] >= math.cos(tilt_tolerance_rad)
    force_matrix_w = env.scene[core_sensor_name].data.force_matrix_w
    assert force_matrix_w is not None, f"Contact sensor '{core_sensor_name}' has no filtered force matrix."
    support_force_n = force_matrix_w.torch[:, 0, 0, 2]
    weight_n = env.scene[core_name].data.body_mass.torch.sum(dim=-1).to(env.device) * abs(float(env.cfg.sim.gravity[2]))
    supported = support_force_n >= support_fraction * weight_n
    return over_pad & at_rest_height & upright & supported


def cover_closed(
    env: IsaacLabArenaManagerBasedRLEnv, fixture_name: str, joint_name: str, tolerance_rad: float
) -> torch.Tensor:
    """Return whether the cover joint is within the tolerance of its closed position (zero)."""
    return env.arena_world.get_joint_position(fixture_name, joint_name).abs() <= tolerance_rad


def core_retrieved_and_cover_closed(
    env: IsaacLabArenaManagerBasedRLEnv,
    fixture_name: str,
    joint_name: str,
    cover_tolerance_rad: float,
    speed_threshold_m_s: float,
    **on_pad_kwargs,
) -> torch.Tensor:
    """Return whether the core rests upright on the pad, nearly still, with the fixture cover closed."""
    core_name = on_pad_kwargs["core_name"]
    still = (
        torch.linalg.vector_norm(env.arena_world.get_root_linear_velocity_w(core_name), dim=-1) < speed_threshold_m_s
    )
    return (
        core_upright_on_pad(env, **on_pad_kwargs)
        & cover_closed(env, fixture_name, joint_name, cover_tolerance_rad)
        & still
    )


@register_task
class SwivelCoverRetrievalTask(TaskBase):
    """Take a core out of a swivel-cover fixture, stand it upright on a pad, and leave the cover closed.

    The cover turns about one hidden corner, swings to one hidden side and is pulled shut by a return spring of hidden
    strength; the fixture asset owns these hidden variables. The task seats the core on the tray floor at
    construction, so the core node needs no pose of its own. Success requires the core standing still and upright on
    the pad, carried by it, with the cover closed, for the required consecutive control steps.

    Args:
        fixture: Swivel-cover fixture articulation, with a ``get_seat_pose`` method and a ``geometry`` attribute.
        core: Rigid object that starts inside the fixture.
        pad: Object the core must stand on.
        episode_length_s: Maximum episode duration in seconds.
        task_description: Natural-language task instruction.
        cover_closed_tolerance_deg: Maximum cover angle from closed, in degrees.
        height_tolerance_m: Maximum deviation of the core from its resting height on the pad.
        tilt_tolerance_deg: Maximum tilt of the core's axis from the pad normal, in degrees.
        support_fraction: Minimum share of the core's weight the pad must carry.
        speed_threshold_m_s: Core speed must be below this.
        success_hold_steps: Consecutive control steps the final state must hold.
    """

    def __init__(
        self,
        fixture: Asset,
        core: Asset,
        pad: Asset,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        cover_closed_tolerance_deg: float = 3.0,
        height_tolerance_m: float = 0.01,
        tilt_tolerance_deg: float = 10.0,
        support_fraction: float = 0.7,
        speed_threshold_m_s: float = 0.03,
        success_hold_steps: int = 15,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        assert success_hold_steps > 0, "success_hold_steps must be positive."
        self.fixture = fixture
        self.core = core
        self.pad = pad
        self.cover_closed_tolerance_rad = math.radians(cover_closed_tolerance_deg)
        self.height_tolerance_m = height_tolerance_m
        self.tilt_tolerance_rad = math.radians(tilt_tolerance_deg)
        self.support_fraction = support_fraction
        self.speed_threshold_m_s = speed_threshold_m_s
        self.success_hold_steps = success_hold_steps
        self.core_sensor_name = f"contact_sensor_{core.name}_on_{pad.name}"
        # Seat the core upright on the tray floor; the fixture's pose fixes where that is.
        core_bounds = core.get_bounding_box()
        self.core_bottom_offset_m = -float(core_bounds.min_point[0, 2])
        core.set_initial_pose(fixture.get_seat_pose(self.core_bottom_offset_m))
        pad_bounds = pad.get_bounding_box()
        self.pad_top_offset_m = float(pad_bounds.max_point[0, 2])
        self.pad_half_size_m = 0.5 * float(min(pad_bounds.size[0, 0], pad_bounds.size[0, 1]))
        if self.task_description is None:
            self.task_description = (
                f"Take the {core.name} out of the {fixture.name} and stand it on the {pad.name}. Leave the fixture"
                " cover closed when you are done."
            )

    def apply_reachability_constraints(self) -> None:
        """The robot must reach the core and the pad."""
        self._apply_reachability_constraints([self.core, self.pad])

    def get_scene_cfg(self):
        sensor_cfg = ContactSensorCfg(
            prim_path=self.core.get_contact_sensor_prim_path(),
            filter_prim_paths_expr=[self.pad.get_contact_sensor_prim_path()],
        )
        scene_cfg_type = make_configclass("SceneCfg", [(self.core_sensor_name, ContactSensorCfg, sensor_cfg)])
        return scene_cfg_type()

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("SwivelCoverRetrievalTask has no Mimic configuration.")

    def get_termination_cfg(self) -> TaskTerminationCfg:
        on_pad_kwargs = dict(
            core_name=self.core.name,
            pad_name=self.pad.name,
            core_sensor_name=self.core_sensor_name,
            pad_half_size_m=self.pad_half_size_m,
            core_rest_height_m=self.pad_top_offset_m + self.core_bottom_offset_m,
            height_tolerance_m=self.height_tolerance_m,
            tilt_tolerance_rad=self.tilt_tolerance_rad,
            support_fraction=self.support_fraction,
        )
        final_state = partial(
            core_retrieved_and_cover_closed,
            fixture_name=self.fixture.name,
            joint_name=COVER_JOINT_NAME,
            cover_tolerance_rad=self.cover_closed_tolerance_rad,
            speed_threshold_m_s=self.speed_threshold_m_s,
            **on_pad_kwargs,
        )
        footprint_half_size_m = self.fixture.geometry.outer_size_m / 2
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="retrieve_core_and_close_cover",
                    predicate_sequence=[
                        (
                            partial(
                                core_out_of_fixture,
                                fixture_name=self.fixture.name,
                                core_name=self.core.name,
                                footprint_half_size_m=footprint_half_size_m,
                            ),
                            0.3,
                        ),
                        (partial(core_upright_on_pad, **on_pad_kwargs), 0.3),
                        (
                            TrueForConsecutiveStepsCfg(predicate=final_state, required_steps=self.success_hold_steps),
                            0.4,
                        ),
                    ],
                ),
            ],
        )

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(lookat_object=self.fixture, offset=np.array([-0.55, -0.45, 0.45]))

    @classmethod
    def success_state_transition(cls, core: str, **_) -> TaskTransition:
        """Success moves the core onto the pad; the env graph has no node state for the cover yet."""
        return TaskTransition(subject=core)
