# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Current terminated cable-routing objective over Arena's native Cable state."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.cable import Cable
from isaaclab_arena.assets.object_base import ObjectBase
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

from .geometry import capsule_centerline

if TYPE_CHECKING:
    from .scene import CableRoutingVariant, TerminatedCableGoal


def _torch(value):
    return value.torch if hasattr(value, "torch") else value


def terminated_cable_route_success(
    env,
    *,
    cable_asset_name: str,
    peg_asset_names: tuple[str, ...],
    port_asset_name: str,
    goal: TerminatedCableGoal,
    cable_half_lengths: tuple[float, ...],
) -> torch.Tensor:
    """Score the current CAP occupied regions after policy termination."""

    def points_in_region(points, fixture, region, *, frame_z: float = 0.0):
        fixture_position = _torch(fixture.data.root_pos_w)
        local = points - fixture_position[:, None, :]
        x_min, y_min, x_max, y_max = region
        return (
            (local[..., 0] >= x_min)
            & (local[..., 0] <= x_max)
            & (local[..., 1] >= y_min)
            & (local[..., 1] <= y_max)
            & (local[..., 2] >= frame_z)
            & (local[..., 2] <= frame_z + 0.15)
        )

    cable = env.scene[cable_asset_name]
    cable_body_q = _torch(cable.data.segment_pose_w)
    half_lengths = cable_body_q.new_tensor(cable_half_lengths)
    cable_points = capsule_centerline(cable_body_q, half_lengths[None].expand(env.num_envs, -1))

    result = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for peg_name, region, minimum in zip(
        peg_asset_names,
        goal.seat_regions,
        goal.seat_min_fractions,
        strict=True,
    ):
        fraction = (
            points_in_region(
                cable_points,
                env.scene[peg_name],
                region,
                frame_z=-0.5 * 0.0235,
            )
            .to(torch.float64)
            .mean(dim=1)
        )
        result &= fraction >= minimum

    port = env.scene[port_asset_name]
    in_port = points_in_region(cable_points, port, goal.port_region)
    result &= in_port.to(torch.float64).mean(dim=1) >= goal.port_min_fraction
    result &= points_in_region(cable_points[:, -1:], port, goal.port_region)[:, 0]

    tcp_offset = cable_points.new_tensor((0.0, 0.0, -0.1347))
    robot = env.scene["right_robot"]
    body_ids, _ = robot.find_bodies("right_gripper")
    assert len(body_ids) == 1, "right_robot must contain exactly one right_gripper body"
    body_position = _torch(robot.data.body_link_pos_w)[:, body_ids[0]]
    body_orientation = _torch(robot.data.body_link_quat_w)[:, body_ids[0]]
    tcp_position_w = body_position + math_utils.quat_apply(
        body_orientation,
        tcp_offset.expand(env.num_envs, -1),
    )
    result &= torch.linalg.vector_norm(cable_points - tcp_position_w[:, None, :], dim=-1).amin(dim=1) > (
        goal.min_tcp_distance
    )

    cable_velocity = _torch(cable.data.segment_velocity_w)[..., :3]
    mean_speed = torch.linalg.vector_norm(cable_velocity, dim=-1).mean(dim=1)
    result &= mean_speed <= goal.max_mean_speed
    # CAP's Arena fork lets a policy explicitly declare that it has finished.  The
    # corresponding generic Arena API has not landed here yet, so retain the same
    # terminal-only scoring semantics and fall back to evaluating at the horizon.
    policy_termination = getattr(env, "external_policy_termination_buf", None)
    if policy_termination is None:
        policy_termination = torch.zeros_like(result)
    result &= policy_termination | (env.episode_length_buf >= env.max_episode_length)
    return result


@configclass
class CableRoutingEventsCfg:
    """Reset the robots, fixtures, and native Cable to authored defaults."""

    reset_scene: EventTerm = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


@register_task
class CableRoutingTaskV2(TaskBase):
    """Weave every guide and place the released free end in the port."""

    def __init__(
        self,
        *,
        cable: Cable,
        pegs: Sequence[ObjectBase],
        port: ObjectBase,
        variant: CableRoutingVariant,
    ) -> None:
        assert isinstance(cable, Cable), "cable must be an Arena Cable asset."
        assert len(pegs) == len(
            variant.terminated_goal.seat_regions
        ), "The current goal requires one occupied region per peg."
        super().__init__(
            episode_length_s=variant.episode_length_s,
            task_description=variant.task_description,
        )
        cable_half_lengths = tuple(
            0.5 * math.dist(start, end)
            for start, end in zip(
                variant.cable_local_positions[:-1],
                variant.cable_local_positions[1:],
                strict=True,
            )
        )
        self.variant = variant
        self.cable = cable
        self.pegs = tuple(pegs)
        self.port = port
        self._events_cfg = CableRoutingEventsCfg()
        self._terminations_cfg = TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                ProgressObjective(
                    name="cable_routing",
                    predicate_sequence=[
                        partial(
                            terminated_cable_route_success,
                            cable_asset_name=cable.name,
                            peg_asset_names=tuple(peg.name for peg in pegs),
                            port_asset_name=port.name,
                            goal=variant.terminated_goal,
                            cable_half_lengths=cable_half_lengths,
                        ),
                    ],
                ),
            ],
        )

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return self._terminations_cfg

    def get_events_cfg(self):
        return self._events_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        return None

    def get_metrics(self):
        return [SuccessRateMetric()]

    def get_recorder_term_cfg(self):
        from .appearance import camera_warmup_recorder_cfg

        return camera_warmup_recorder_cfg()

    def get_viewer_cfg(self) -> ViewerCfg:
        from .scene import BOARD_TOP_Z

        return ViewerCfg(eye=(1.25, -1.10, 1.55), lookat=(0.0125, 0.0, BOARD_TOP_Z))
