# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cantilever look-alike bars with hidden centers of mass past a line over a channel, each resting on its own."""

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
from isaaclab_arena.assets.cantilever_ledge_bars_assets import CantileverBarGeometry, CantileverLedgeGeometry
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


def bar_cantilevered_past_line(
    env: IsaacLabArenaManagerBasedRLEnv,
    bar_name: str,
    T_E_L: tuple[float, ...],
    ledge_geometry: CantileverLedgeGeometry,
    bar_geometry: CantileverBarGeometry,
    tilt_tolerance_rad: float,
    rest_height_tolerance_m: float,
    inboard_margin_m: float,
) -> torch.Tensor:
    """Check that a bar lies level on the ledge top with one end on the ledge and the other past the yellow line.

    Args:
        env: The environment.
        bar_name: Scene name of the bar.
        T_E_L: Ledge frame L in the environment frame E, as (x, y, z, qx, qy, qz, qw).
        ledge_geometry: Dimensions of the ledge fixture.
        bar_geometry: Dimensions of the bar.
        tilt_tolerance_rad: Maximum angle between the bar's long axis and the ledge top.
        rest_height_tolerance_m: Maximum distance of the bar center from its height when lying on the ledge top.
        inboard_margin_m: Minimum distance the inner end must reach past the ledge edge onto the ledge.
    """
    T_E_B = env.arena_world.get_pose_e(bar_name)
    T_E_L_tensor = torch.tensor(T_E_L, device=T_E_B.device, dtype=T_E_B.dtype).expand(T_E_B.shape[0], 7)
    q_E_L = T_E_L_tensor[:, 3:]
    t_L_B = quat_apply_inverse(q_E_L, T_E_B[:, :3] - T_E_L_tensor[:, :3])
    axis_B = torch.zeros_like(t_L_B)
    axis_B[:, 1] = 1.0
    axis_L = quat_apply_inverse(q_E_L, quat_apply(T_E_B[:, 3:], axis_B))
    half_length = bar_geometry.length_m / 2
    end_a_y = t_L_B[:, 1] + half_length * axis_L[:, 1]
    end_b_y = t_L_B[:, 1] - half_length * axis_L[:, 1]
    outer_end_y = torch.minimum(end_a_y, end_b_y)
    inner_end_y = torch.maximum(end_a_y, end_b_y)
    reaches_past_line = outer_end_y < -ledge_geometry.line_overhang_m
    inner_end_on_ledge = (inner_end_y > inboard_margin_m) & (inner_end_y < ledge_geometry.ledge_depth_m)
    over_ledge_width = t_L_B[:, 0].abs() < ledge_geometry.width_m / 2 - bar_geometry.width_m / 2
    level = axis_L[:, 2].abs() < math.sin(tilt_tolerance_rad)
    rest_height = ledge_geometry.ledge_height_m + bar_geometry.thickness_m / 2
    resting_on_top = (t_L_B[:, 2] - rest_height).abs() < rest_height_tolerance_m
    return reaches_past_line & inner_end_on_ledge & over_ledge_width & level & resting_on_top


def bar_rests_on_its_own(
    env: IsaacLabArenaManagerBasedRLEnv,
    bar_name: str,
    sensor_name: str,
    support_fraction: float,
    linear_speed_threshold_m_s: float,
    angular_speed_threshold_rad_s: float,
    gripper: Gripper | None,
    gripper_clearance_m: float,
) -> torch.Tensor:
    """Check that a bar is still, its contacts carry its weight, and the gripper, when given, is clear of it.

    A bar held by an external force has contacts carrying less than its weight; a gripper holding it is rejected by
    the clearance check.
    """
    linear_speed = torch.linalg.vector_norm(env.arena_world.get_root_linear_velocity_w(bar_name), dim=-1)
    angular_speed = torch.linalg.vector_norm(env.arena_world.get_root_angular_velocity_w(bar_name), dim=-1)
    still = (linear_speed < linear_speed_threshold_m_s) & (angular_speed < angular_speed_threshold_rad_s)
    mass_kg = env.scene[bar_name].data.body_mass.torch.sum(dim=-1).to(env.device)
    load_kg = env.scene[sensor_name].data.net_forces_w.torch[:, 0, 2] / abs(float(env.cfg.sim.gravity[2]))
    on_its_own = still & (load_kg >= support_fraction * mass_kg)
    if gripper is not None:
        t_W_gripper = gripper.get_position_w(env.arena_world)
        distance_m = torch.linalg.vector_norm(t_W_gripper - env.arena_world.get_position_w(bar_name), dim=-1)
        on_its_own &= distance_m > gripper_clearance_m
    return on_its_own


def all_bars_cantilevered(env: IsaacLabArenaManagerBasedRLEnv, bar_names: list[str], **pose_kwargs) -> torch.Tensor:
    """Check that every bar lies on the ledge reaching past the yellow line."""
    result = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for bar_name in bar_names:
        result &= bar_cantilevered_past_line(env, bar_name, **pose_kwargs)
    return result


@register_task
class CantileverLedgeBarsTask(TaskBase):
    """Set look-alike bars on a ledge so each reaches past a line over a channel and stays there on its own.

    Each bar's center of mass is hidden: toward its red end, toward its blue end, or centered. With the required
    overhang, a bar stays only when its center of mass is over the ledge: an off-center bar with its heavy end on the
    ledge, or a centered bar with the clamp-on counterweight on its inner end. Progress has two stages: every bar lies
    on the ledge past the line, and then every bar also rests there on its own (still, its weight carried by contacts
    and the gripper clear) for the required consecutive steps.

    Args:
        ledge: The ledge-and-channel fixture.
        bars: Bars to cantilever.
        counterweight: Clamp-on counterweight the robot may use; only marked as a reachability target.
        episode_length_s: Maximum episode duration in seconds.
        task_description: Natural-language task instruction.
        tilt_tolerance_deg: Maximum angle between a bar and the ledge top, in degrees.
        rest_height_tolerance_m: Maximum distance of a bar's center from its height lying on the ledge top.
        inboard_margin_m: Minimum length of a bar's inner end past the ledge edge onto the ledge.
        support_fraction: Minimum share of a bar's weight its contacts must carry.
        linear_speed_threshold_m_s: A resting bar must move slower than this.
        angular_speed_threshold_rad_s: A resting bar must turn slower than this.
        gripper_clearance_m: Minimum gripper distance from each bar's center.
        hold_steps: Consecutive control steps every bar must rest on its own past the line.
    """

    def __init__(
        self,
        ledge: Asset,
        bars: list[Asset],
        counterweight: Asset | None = None,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        tilt_tolerance_deg: float = 3.0,
        rest_height_tolerance_m: float = 0.008,
        inboard_margin_m: float = 0.005,
        support_fraction: float = 0.5,
        linear_speed_threshold_m_s: float = 0.02,
        angular_speed_threshold_rad_s: float = 0.2,
        gripper_clearance_m: float = 0.08,
        hold_steps: int = 30,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        assert bars, "CantileverLedgeBarsTask needs at least one bar."
        assert hold_steps > 0, "hold_steps must be positive."
        self.ledge = ledge
        self.bars = list(bars)
        self.counterweight = counterweight
        self.tilt_tolerance_rad = math.radians(tilt_tolerance_deg)
        self.rest_height_tolerance_m = rest_height_tolerance_m
        self.inboard_margin_m = inboard_margin_m
        self.support_fraction = support_fraction
        self.linear_speed_threshold_m_s = linear_speed_threshold_m_s
        self.angular_speed_threshold_rad_s = angular_speed_threshold_rad_s
        self.gripper_clearance_m = gripper_clearance_m
        self.hold_steps = hold_steps
        self.gripper: Gripper | None = None
        self.bar_sensor_names = [f"contact_sensor_{bar.name}" for bar in self.bars]
        if self.task_description is None:
            # The original task's wording, which calls the channel beside the ledge by another name.
            self.task_description = (
                "Set all three bars on the ledge so that each one reaches past the yellow line over the"
                " trough and stays there on its own. A clamp-on weight is available."  # codespell:ignore trough
            )

    def configure_for_embodiment(self, embodiment: EmbodimentBase) -> None:
        """Use the embodiment's gripper, when it has one, for the gripper-clearance check."""
        self.gripper = embodiment.gripper

    def apply_reachability_constraints(self) -> None:
        """The robot must reach every bar and the counterweight."""
        targets = list(self.bars)
        if self.counterweight is not None:
            targets.append(self.counterweight)
        self._apply_reachability_constraints(targets)

    def get_scene_cfg(self):
        sensors = [
            (sensor_name, ContactSensorCfg(prim_path=bar.get_contact_sensor_prim_path()))
            for bar, sensor_name in zip(self.bars, self.bar_sensor_names)
        ]
        scene_cfg_type = make_configclass("SceneCfg", [(name, ContactSensorCfg, cfg) for name, cfg in sensors])
        return scene_cfg_type()

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("CantileverLedgeBarsTask has no Mimic configuration.")

    def _pose_kwargs(self) -> dict:
        T_E_L = self.ledge.get_initial_pose()
        assert T_E_L is not None, f"The ledge '{self.ledge.name}' needs a fixed initial pose."
        return dict(
            T_E_L=(*T_E_L.position_xyz, *T_E_L.rotation_xyzw),
            ledge_geometry=self.ledge.geometry,
            bar_geometry=self.bars[0].geometry,
            tilt_tolerance_rad=self.tilt_tolerance_rad,
            rest_height_tolerance_m=self.rest_height_tolerance_m,
            inboard_margin_m=self.inboard_margin_m,
        )

    def get_termination_cfg(self) -> TaskTerminationCfg:
        pose_kwargs = self._pose_kwargs()
        bar_names = [bar.name for bar in self.bars]
        # Bind the gripper lazily: configure_for_embodiment runs after the task is constructed.
        task = self

        def all_bars_rest_past_line(env: IsaacLabArenaManagerBasedRLEnv) -> torch.Tensor:
            result = all_bars_cantilevered(env, bar_names, **pose_kwargs)
            for bar_name, sensor_name in zip(bar_names, task.bar_sensor_names):
                result &= bar_rests_on_its_own(
                    env,
                    bar_name,
                    sensor_name,
                    support_fraction=task.support_fraction,
                    linear_speed_threshold_m_s=task.linear_speed_threshold_m_s,
                    angular_speed_threshold_rad_s=task.angular_speed_threshold_rad_s,
                    gripper=task.gripper,
                    gripper_clearance_m=task.gripper_clearance_m,
                )
            return result

        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="cantilever_bars_past_line",
                    predicate_sequence=[
                        (partial(all_bars_cantilevered, bar_names=bar_names, **pose_kwargs), 0.3),
                        (
                            TrueForConsecutiveStepsCfg(
                                predicate=all_bars_rest_past_line, required_steps=self.hold_steps
                            ),
                            0.7,
                        ),
                    ],
                ),
            ],
        )

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(lookat_object=self.ledge, offset=np.array([-0.3, -0.75, 0.42]))

    @classmethod
    def success_state_transition(cls, ledge: str, **_) -> TaskTransition:
        """Success changes the bars' arrangement on the ledge; the env graph has no node state for it yet."""
        return TaskTransition(subject=ledge)
