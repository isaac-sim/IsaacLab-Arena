# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Push a sealed crate with a hidden center of mass, without lifting it, all the way into a snug cubby."""

from __future__ import annotations

import math
import numpy as np
import torch
from typing import TYPE_CHECKING

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.crate_cubby_push_assets import DEFAULT_CUBBY_GEOMETRY, SealedCrateGeometry
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

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.embodiments.gripper import Gripper
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv

_CRATE = SealedCrateGeometry()
_CUBBY = DEFAULT_CUBBY_GEOMETRY


def _crate_points_w(env: IsaacLabArenaManagerBasedRLEnv, crate_name: str, points_C: list[tuple]) -> torch.Tensor:
    """Return crate-frame points in the world frame, with shape (num_envs, num_points, 3)."""
    T_W_C = env.arena_world.get_pose_w(crate_name)
    num_points = len(points_C)
    t_C = torch.tensor(points_C, dtype=T_W_C.dtype, device=T_W_C.device).expand(env.num_envs, num_points, 3)
    q_W_C = T_W_C[:, None, 3:].expand(env.num_envs, num_points, 4)
    return T_W_C[:, None, :3] + quat_apply(q_W_C.reshape(-1, 4), t_C.reshape(-1, 3)).reshape(env.num_envs, -1, 3)


def _crate_sole_corners_w(env: IsaacLabArenaManagerBasedRLEnv, crate_name: str) -> torch.Tensor:
    half_depth, half_width = _CRATE.depth_m / 2, _CRATE.width_m / 2
    corners_C = [(sx * half_depth, sy * half_width, 0.0) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]
    return _crate_points_w(env, crate_name, corners_C)


def crate_tilt_rad(env: IsaacLabArenaManagerBasedRLEnv, crate_name: str) -> torch.Tensor:
    """Return the angle between the crate's up axis and world up."""
    q_W_C = env.arena_world.get_pose_w(crate_name)[:, 3:]
    up_C = torch.tensor([0.0, 0.0, 1.0], dtype=q_W_C.dtype, device=q_W_C.device).expand(env.num_envs, 3)
    return torch.acos(quat_apply(q_W_C, up_C)[:, 2].clamp(-1.0, 1.0))


def crate_inside_cubby(
    env: IsaacLabArenaManagerBasedRLEnv,
    crate_name: str,
    cubby_name: str,
    insertion_margin_m: float,
    wall_contact_tolerance_m: float,
) -> torch.Tensor:
    """Check that the crate's whole footprint lies inside the cubby, at least insertion_margin_m past the entrance.

    A corner may reach wall_contact_tolerance_m into a side or back wall, so a crate resting against a wall counts.
    """
    T_W_K = env.arena_world.get_pose_w(cubby_name)
    corners_W = _crate_sole_corners_w(env, crate_name)
    num_corners = corners_W.shape[1]
    q_W_K = T_W_K[:, None, 3:].expand(env.num_envs, num_corners, 4).reshape(-1, 4)
    corners_K = quat_apply_inverse(q_W_K, (corners_W - T_W_K[:, None, :3]).reshape(-1, 3)).reshape(
        env.num_envs, num_corners, 3
    )
    inside = corners_K[..., 0] >= insertion_margin_m
    inside &= corners_K[..., 0] <= _CUBBY.interior_depth_m + wall_contact_tolerance_m
    inside &= corners_K[..., 1].abs() <= _CUBBY.interior_width_m / 2 + wall_contact_tolerance_m
    return inside.all(dim=-1)


def crate_left_standing_in_cubby(
    env: IsaacLabArenaManagerBasedRLEnv,
    crate_name: str,
    cubby_name: str,
    insertion_margin_m: float,
    wall_contact_tolerance_m: float,
    max_tilt_rad: float,
    rest_linear_speed_m_s: float,
    rest_angular_speed_rad_s: float,
    gripper: Gripper | None,
    gripper_clearance_m: float,
) -> torch.Tensor:
    """Check that the crate is inside the cubby, upright, at rest and, when a gripper is given, clear of it.

    The gripper clearance is measured from the crate's geometric center.
    """
    world = env.arena_world
    result = crate_inside_cubby(env, crate_name, cubby_name, insertion_margin_m, wall_contact_tolerance_m)
    result &= crate_tilt_rad(env, crate_name) <= max_tilt_rad
    result &= torch.linalg.vector_norm(world.get_root_linear_velocity_w(crate_name), dim=-1) <= rest_linear_speed_m_s
    result &= (
        torch.linalg.vector_norm(world.get_root_angular_velocity_w(crate_name), dim=-1) <= rest_angular_speed_rad_s
    )
    if gripper is not None:
        crate_center_W = _crate_points_w(env, crate_name, [(0.0, 0.0, _CRATE.height_m / 2)])[:, 0]
        distance_m = torch.linalg.vector_norm(gripper.get_position_w(world) - crate_center_W, dim=-1)
        result &= distance_m > gripper_clearance_m
    return result


def crate_lifted(
    env: IsaacLabArenaManagerBasedRLEnv, crate_name: str, floor_name: str, floor_thickness_m: float, lift_m: float
) -> torch.Tensor:
    """Return whether every sole corner of the crate is more than lift_m above the floor's top surface."""
    floor_top_z = env.arena_world.get_pose_w(floor_name)[:, 2] + floor_thickness_m / 2
    lowest_corner_z = _crate_sole_corners_w(env, crate_name)[..., 2].min(dim=-1).values
    return lowest_corner_z > floor_top_z + lift_m


def crate_tipped_over(env: IsaacLabArenaManagerBasedRLEnv, crate_name: str, tipped_tilt_rad: float) -> torch.Tensor:
    """Return whether the crate has tilted past tipped_tilt_rad from upright."""
    return crate_tilt_rad(env, crate_name) > tipped_tilt_rad


class CrateMishandlingLatch:
    """Remember, per environment and episode, whether the crate has tipped over or been lifted.

    The failure terminations end an episode at the moment the crate tips or is lifted; this latch keeps such an episode
    from reaching success afterwards when something steps it on without honoring terminations, for example a crate
    caught upright again by the cubby walls. It detects a new episode by the episode step counter going down.

    Args:
        crate_name: Scene name of the crate.
        floor_name: Scene name of the floor the crate stands on.
        floor_thickness_m: Thickness of the floor, whose pose is at its center, in meters.
        lift_m: Height above the floor at which a crate with all corners raised counts as lifted, in meters.
        tipped_tilt_rad: Tilt from upright at which the crate counts as tipped over, in radians.
    """

    def __init__(
        self, crate_name: str, floor_name: str, floor_thickness_m: float, lift_m: float, tipped_tilt_rad: float
    ):
        self.crate_name = crate_name
        self.floor_name = floor_name
        self.floor_thickness_m = floor_thickness_m
        self.lift_m = lift_m
        self.tipped_tilt_rad = tipped_tilt_rad
        self._mishandled: torch.Tensor | None = None
        self._last_episode_step: torch.Tensor | None = None

    def update(self, env: IsaacLabArenaManagerBasedRLEnv) -> torch.Tensor:
        """Fold the crate's current state into the latch and return whether it was mishandled this episode."""
        episode_step = env.episode_length_buf.clone()
        if self._mishandled is None:
            self._mishandled = torch.zeros(env.num_envs, dtype=torch.bool, device=episode_step.device)
            self._last_episode_step = episode_step
        new_episode = episode_step < self._last_episode_step
        self._mishandled &= ~new_episode
        self._last_episode_step = episode_step
        self._mishandled |= crate_tipped_over(env, self.crate_name, self.tipped_tilt_rad)
        self._mishandled |= crate_lifted(env, self.crate_name, self.floor_name, self.floor_thickness_m, self.lift_m)
        return self._mishandled.clone()


@register_task
class CrateCubbyPushTask(TaskBase):
    """Push a sealed crate, without lifting it, fully into a snug cubby and leave it standing there.

    The crate's center of mass is hidden. A push too high on its rear face tips it about its leading edge, and a push
    that misses its center of mass sideways turns it so it jams in the cubby entrance. Success requires the crate's
    whole footprint inside the cubby, upright, at rest and clear of the gripper for the required consecutive steps,
    with the crate never tipped over or lifted earlier in the episode. Failure occurs when the crate is lifted off the
    floor or tips over.

    Args:
        crate: The sealed crate.
        cubby: The cubby shelf.
        floor: The floor mat the crate slides on.
        episode_length_s: Maximum episode duration in seconds.
        task_description: Natural-language task instruction.
        insertion_margin_m: Minimum distance of the crate's rear corners past the cubby entrance, in meters.
        wall_contact_tolerance_m: Distance a crate corner may reach into a cubby wall, for contact offsets, in meters.
        max_tilt_deg: Maximum crate tilt from upright for success, in degrees.
        rest_linear_speed_m_s: Crate linear speed must be at most this, in m/s.
        rest_angular_speed_rad_s: Crate angular speed must be at most this, in rad/s.
        gripper_clearance_m: Minimum gripper distance from the crate's geometric center, in meters.
        hold_steps: Consecutive control steps the final state must hold.
        lift_m: Height above the floor at which a crate with all corners raised counts as lifted, in meters.
        tipped_tilt_deg: Tilt from upright at which the crate counts as tipped over, in degrees. The default is about
            where a front-heavy crate's center of mass passes over its leading edge, atan(0.03 / 0.08); past it the
            crate falls unless something catches it.
    """

    def __init__(
        self,
        crate: Asset,
        cubby: Asset,
        floor: Asset,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        insertion_margin_m: float = 0.005,
        wall_contact_tolerance_m: float = 0.003,
        max_tilt_deg: float = 8.0,
        rest_linear_speed_m_s: float = 0.01,
        rest_angular_speed_rad_s: float = 0.1,
        gripper_clearance_m: float = 0.11,
        hold_steps: int = 30,
        lift_m: float = 0.02,
        tipped_tilt_deg: float = 20.0,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        assert hold_steps > 0, "hold_steps must be positive."
        self.crate = crate
        self.cubby = cubby
        self.floor = floor
        self.insertion_margin_m = insertion_margin_m
        self.wall_contact_tolerance_m = wall_contact_tolerance_m
        self.max_tilt_rad = math.radians(max_tilt_deg)
        self.rest_linear_speed_m_s = rest_linear_speed_m_s
        self.rest_angular_speed_rad_s = rest_angular_speed_rad_s
        self.gripper_clearance_m = gripper_clearance_m
        self.hold_steps = hold_steps
        self.lift_m = lift_m
        self.tipped_tilt_rad = math.radians(tipped_tilt_deg)
        self.gripper: Gripper | None = None
        if self.task_description is None:
            self.task_description = (
                "Push the sealed crate all the way into the cubby under the shelf and leave it standing there. The"
                " crate cannot be lifted."
            )

    def configure_for_embodiment(self, embodiment: EmbodimentBase) -> None:
        """Use the embodiment's gripper, when it has one, for the gripper-clearance check."""
        self.gripper = embodiment.gripper

    def apply_reachability_constraints(self) -> None:
        """The robot must reach the crate."""
        self._apply_reachability_constraints([self.crate])

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("CrateCubbyPushTask has no Mimic configuration.")

    def get_termination_cfg(self) -> TaskTerminationCfg:
        floor_thickness_m = getattr(self.floor, "thickness_m", 0.0)
        # Both success stages update one shared latch; exactly one of them is evaluated per step until success.
        # Closures, unlike partials, keep the latch shared when the configuration is deep-copied.
        latch = CrateMishandlingLatch(
            self.crate.name, self.floor.name, floor_thickness_m, self.lift_m, self.tipped_tilt_rad
        )
        # Bind the gripper lazily: configure_for_embodiment runs after the task is constructed.
        task = self

        def inside_and_handled(env: IsaacLabArenaManagerBasedRLEnv) -> torch.Tensor:
            mishandled = latch.update(env)
            inside = crate_inside_cubby(
                env,
                crate_name=task.crate.name,
                cubby_name=task.cubby.name,
                insertion_margin_m=task.insertion_margin_m,
                wall_contact_tolerance_m=task.wall_contact_tolerance_m,
            )
            return inside & ~mishandled

        def standing_in_cubby(env: IsaacLabArenaManagerBasedRLEnv) -> torch.Tensor:
            mishandled = latch.update(env)
            return ~mishandled & crate_left_standing_in_cubby(
                env,
                crate_name=task.crate.name,
                cubby_name=task.cubby.name,
                insertion_margin_m=task.insertion_margin_m,
                wall_contact_tolerance_m=task.wall_contact_tolerance_m,
                max_tilt_rad=task.max_tilt_rad,
                rest_linear_speed_m_s=task.rest_linear_speed_m_s,
                rest_angular_speed_rad_s=task.rest_angular_speed_rad_s,
                gripper=task.gripper,
                gripper_clearance_m=task.gripper_clearance_m,
            )

        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="crate_pushed_into_cubby",
                    predicate_sequence=[
                        (inside_and_handled, 0.5),
                        (TrueForConsecutiveStepsCfg(predicate=standing_in_cubby, required_steps=self.hold_steps), 0.5),
                    ],
                ),
            ],
            failures={
                "crate_lifted": TerminationTermCfg(
                    func=crate_lifted,
                    params={
                        "crate_name": self.crate.name,
                        "floor_name": self.floor.name,
                        "floor_thickness_m": floor_thickness_m,
                        "lift_m": self.lift_m,
                    },
                ),
                "crate_tipped_over": TerminationTermCfg(
                    func=crate_tipped_over,
                    params={"crate_name": self.crate.name, "tipped_tilt_rad": self.tipped_tilt_rad},
                ),
            },
        )

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(lookat_object=self.cubby, offset=np.array([0.38, -0.55, 0.3]))

    @classmethod
    def success_state_transition(cls, crate: str, cubby: str, **_) -> TaskTransition:
        """Success moves the crate into the cubby; the env graph has no node state for it yet."""
        return TaskTransition(subject=crate)
