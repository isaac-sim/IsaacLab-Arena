# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Empty a cabinet whose look-alike fronts open by hidden modes onto a tray, then close every front."""

from __future__ import annotations

import math
import numpy as np
import torch
from functools import partial
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.utils.math import quat_apply

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.assets.sliding_front_cabinet_usd import NUM_COMPARTMENTS, front_joint_name
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.predicates.spatial import object_on_destination
from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.tasks.task_transition import TaskTransition
from isaaclab_arena.utils.configclass import make_configclass
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


def front_openness(env: IsaacLabArenaManagerBasedRLEnv, cabinet_name: str) -> torch.Tensor:
    """Return each front's openness in [0, 1]: joint position magnitude over its travel, shape (num_envs, 3).

    Every front is closed at joint position 0 whatever its opening mode, so openness does not depend on the mode.
    """
    data = env.scene.articulations[cabinet_name].data
    joint_ids = [data.joint_names.index(front_joint_name(index)) for index in range(1, NUM_COMPARTMENTS + 1)]
    joint_pos = data.joint_pos.torch[:, joint_ids]
    limits = data.joint_pos_limits.torch[:, joint_ids]
    travel = torch.maximum(limits[..., 0].abs(), limits[..., 1].abs())
    return joint_pos.abs() / travel


def fronts_closed(env: IsaacLabArenaManagerBasedRLEnv, cabinet_name: str, closed_openness: float) -> torch.Tensor:
    """Check that every front's openness is at most closed_openness."""
    return (front_openness(env, cabinet_name) <= closed_openness).all(dim=-1)


def _cabinet_tilt_rad(env: IsaacLabArenaManagerBasedRLEnv, cabinet_name: str) -> torch.Tensor:
    """Return the angle between the cabinet's up axis and the world's up axis."""
    q_W_C = env.arena_world.get_pose_w(cabinet_name)[:, 3:]
    up_C = torch.tensor([0.0, 0.0, 1.0], device=q_W_C.device, dtype=q_W_C.dtype).expand(q_W_C.shape[0], 3)
    up_W = quat_apply(q_W_C, up_C)
    return torch.acos(up_W[:, 2].clamp(-1.0, 1.0))


def cabinet_in_place(
    env: IsaacLabArenaManagerBasedRLEnv,
    cabinet_name: str,
    t_E_initial: tuple[float, float, float],
    displacement_tolerance_m: float,
    tilt_tolerance_rad: float,
) -> torch.Tensor:
    """Check that the cabinet stands within the tolerances of its initial position and upright.

    Args:
        t_E_initial: The cabinet's initial position in its environment's frame E.
    """
    t_W_cabinet = env.arena_world.get_pose_w(cabinet_name)[:, :3]
    t_E_cabinet = t_W_cabinet - env.scene.env_origins
    t_E_start = torch.tensor(t_E_initial, device=t_E_cabinet.device, dtype=t_E_cabinet.dtype)
    displacement_m = torch.linalg.vector_norm(t_E_cabinet - t_E_start, dim=-1)
    return (displacement_m <= displacement_tolerance_m) & (_cabinet_tilt_rad(env, cabinet_name) <= tilt_tolerance_rad)


def cabinet_toppled(env: IsaacLabArenaManagerBasedRLEnv, cabinet_name: str, tilt_limit_rad: float) -> torch.Tensor:
    """Return whether the cabinet tilts beyond the limit, for example knocked over."""
    return _cabinet_tilt_rad(env, cabinet_name) > tilt_limit_rad


def tokens_on_tray(
    env: IsaacLabArenaManagerBasedRLEnv,
    token_names: list[str],
    sensor_names: list[str],
    tray_name: str,
    force_threshold: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Check that every token rests on the tray, supported by it and slower than the velocity threshold."""
    result = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for token_name, sensor_name in zip(token_names, sensor_names):
        result &= object_on_destination(
            env,
            object_cfg=SceneEntityCfg(token_name),
            destination_cfg=SceneEntityCfg(tray_name),
            contact_sensor_cfg=SceneEntityCfg(sensor_name),
            force_threshold=force_threshold,
            velocity_threshold=velocity_threshold,
        )
    return result


def cabinet_emptied_and_closed(
    env: IsaacLabArenaManagerBasedRLEnv,
    token_names: list[str],
    sensor_names: list[str],
    tray_name: str,
    force_threshold: float,
    velocity_threshold: float,
    cabinet_name: str,
    closed_openness: float,
    t_E_initial: tuple[float, float, float],
    displacement_tolerance_m: float,
    tilt_tolerance_rad: float,
) -> torch.Tensor:
    """Check that every token rests on the tray, every front is closed and the cabinet stands where it started."""
    return (
        tokens_on_tray(env, token_names, sensor_names, tray_name, force_threshold, velocity_threshold)
        & fronts_closed(env, cabinet_name, closed_openness)
        & cabinet_in_place(env, cabinet_name, t_E_initial, displacement_tolerance_m, tilt_tolerance_rad)
    )


@register_task
class SlidingFrontCabinetTask(TaskBase):
    """Move the token out of each compartment of a cabinet onto a tray, close every front, and leave the cabinet.

    Each front's opening mode (slide left, slide right or fold down) is hidden in the cabinet asset. Success requires
    every token to rest on the tray, every front to be closed and the cabinet within the tolerances of its initial
    pose, for the required consecutive steps. Failure occurs when a token falls below the minimum height or the
    cabinet tips over.

    Args:
        cabinet: The cabinet articulation; its initial pose must be fixed.
        tokens: The tokens, one per compartment.
        tray: The tray the tokens go on.
        episode_length_s: Maximum episode duration in seconds.
        task_description: Natural-language task instruction.
        closed_openness: A front counts as closed at or below this openness (joint position over travel).
        cabinet_displacement_tolerance_m: Maximum distance of the cabinet from its initial position, in meters.
        cabinet_tilt_tolerance_deg: Maximum tilt of the cabinet from upright, in degrees.
        cabinet_topple_deg: Tilt beyond which the cabinet counts as toppled (failure), in degrees.
        token_force_threshold: Minimum upward support force from the tray on each token, in newtons.
        token_velocity_threshold: Every token's linear speed must be below this, in m/s.
        hold_steps: Consecutive control steps the final state must hold.
        token_min_height_m: A token below this world height counts as dropped.
    """

    def __init__(
        self,
        cabinet: Asset,
        tokens: list[Asset],
        tray: Asset,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        closed_openness: float = 0.05,
        cabinet_displacement_tolerance_m: float = 0.03,
        cabinet_tilt_tolerance_deg: float = 10.0,
        cabinet_topple_deg: float = 45.0,
        token_force_threshold: float = 0.02,
        token_velocity_threshold: float = 0.01,
        hold_steps: int = 15,
        token_min_height_m: float = -0.05,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        assert len(tokens) == NUM_COMPARTMENTS, f"SlidingFrontCabinetTask needs {NUM_COMPARTMENTS} tokens."
        assert hold_steps > 0, "hold_steps must be positive."
        self.cabinet = cabinet
        self.tokens = list(tokens)
        self.tray = tray
        self.closed_openness = closed_openness
        self.cabinet_displacement_tolerance_m = cabinet_displacement_tolerance_m
        self.cabinet_tilt_tolerance_rad = math.radians(cabinet_tilt_tolerance_deg)
        self.cabinet_topple_rad = math.radians(cabinet_topple_deg)
        self.token_force_threshold = token_force_threshold
        self.token_velocity_threshold = token_velocity_threshold
        self.hold_steps = hold_steps
        self.token_min_height_m = token_min_height_m
        self.sensor_names = [f"contact_sensor_{token.name}_on_{tray.name}" for token in self.tokens]
        if self.task_description is None:
            self.task_description = (
                "Take the token out of each of the three cabinet compartments and put all three on the tray. Leave"
                " every compartment front closed and the cabinet where it stands."
            )

    def apply_reachability_constraints(self) -> None:
        """The robot must reach the tray. The tokens are anchored inside the anchored cabinet, so placement skips them."""
        self._apply_reachability_constraints([self.tray])

    def get_scene_cfg(self):
        sensors = []
        for token, sensor_name in zip(self.tokens, self.sensor_names):
            sensors.append((sensor_name, token.get_contact_sensor_cfg(contact_against_object=self.tray)))
        scene_cfg_type = make_configclass("SceneCfg", [(name, ContactSensorCfg, cfg) for name, cfg in sensors])
        return scene_cfg_type()

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("SlidingFrontCabinetTask has no Mimic configuration.")

    def get_termination_cfg(self) -> TaskTerminationCfg:
        initial_pose = self.cabinet.get_initial_pose()
        assert isinstance(initial_pose, Pose), "SlidingFrontCabinetTask needs the cabinet's initial pose to be fixed."
        token_names = [token.name for token in self.tokens]
        on_tray = dict(
            token_names=token_names,
            sensor_names=self.sensor_names,
            tray_name=self.tray.name,
            force_threshold=self.token_force_threshold,
            velocity_threshold=self.token_velocity_threshold,
        )
        final_state = partial(
            cabinet_emptied_and_closed,
            **on_tray,
            cabinet_name=self.cabinet.name,
            closed_openness=self.closed_openness,
            t_E_initial=tuple(initial_pose.position_xyz),
            displacement_tolerance_m=self.cabinet_displacement_tolerance_m,
            tilt_tolerance_rad=self.cabinet_tilt_tolerance_rad,
        )
        failures = {
            "cabinet_toppled": TerminationTermCfg(
                func=cabinet_toppled,
                params={"cabinet_name": self.cabinet.name, "tilt_limit_rad": self.cabinet_topple_rad},
            )
        }
        for token_name in token_names:
            failures[f"{token_name}_dropped"] = TerminationTermCfg(
                func=mdp_isaac_lab.root_height_below_minimum,
                params={"minimum_height": self.token_min_height_m, "asset_cfg": SceneEntityCfg(token_name)},
            )
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="empty_cabinet_onto_tray",
                    predicate_sequence=[
                        (partial(tokens_on_tray, **on_tray), 0.5),
                        (TrueForConsecutiveStepsCfg(predicate=final_state, required_steps=self.hold_steps), 0.5),
                    ],
                ),
            ],
            failures=failures,
        )

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        """Look past the robot at the cabinet's fronts and the tray in front of it."""
        initial_pose = self.cabinet.get_initial_pose()
        if not isinstance(initial_pose, Pose):
            return ViewerCfg()
        t_E_cabinet = np.array(initial_pose.position_xyz, dtype=float)
        eye = t_E_cabinet + np.array([-0.80, -0.50, 0.55])
        lookat = t_E_cabinet + np.array([-0.06, -0.12, 0.13])
        return ViewerCfg(eye=tuple(float(x) for x in eye), lookat=tuple(float(x) for x in lookat), origin_type="env")

    @classmethod
    def success_state_transition(cls, tokens: list[str], tray: str, **_) -> TaskTransition:
        """Success moves the tokens onto the tray; the env graph has no multi-subject relocation yet."""
        return TaskTransition(subject=tray)
