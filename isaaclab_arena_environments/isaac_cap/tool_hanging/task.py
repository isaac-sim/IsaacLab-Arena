# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Every tool must sit on its fixture: geometry satisfied, in contact, and at rest."""

from __future__ import annotations

import torch
from dataclasses import MISSING, dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.predicates.composite import CompositePredicate
from isaaclab_arena.tasks.predicates.spatial import velocity_below_threshold
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.terminations import SuccessMode
from isaaclab_arena.utils.configclass import make_configclass

from .geometry import LoopOnRod, PointInBox, goal_geometry_from_dict


@dataclass(frozen=True)
class HangingGoal:
    """One tool scored against one fixture."""

    tool: Asset
    fixture: Asset
    geometry: LoopOnRod | PointInBox

    @property
    def contact_sensor_name(self) -> str:
        return f"contact_sensor_{self.tool.name}_{self.fixture.name}"


def goal_geometry_satisfied(env, tool_name: str, fixture_name: str, geometry: LoopOnRod | PointInBox) -> torch.Tensor:
    """Evaluate the goal geometry on the live tool and fixture poses."""
    return geometry.evaluate(env.arena_world.get_pose_w(tool_name), env.arena_world.get_pose_w(fixture_name))


def contact_force_above(env, sensor_name: str, force_threshold: float) -> torch.Tensor:
    """Check the strongest filtered contact force reported by a sensor."""
    force_w = env.scene[sensor_name].data.force_matrix_w.torch
    magnitude = torch.linalg.vector_norm(force_w, dim=-1)
    return magnitude.reshape(magnitude.shape[0], -1).amax(dim=-1) >= force_threshold


@configclass
class TerminationsCfg:
    """Episode timeout and the all-goals success condition."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.time_out, time_out=True)
    success: TerminationTermCfg = MISSING


class ToolHangingTask(TaskBase):
    """Hang or stow every listed tool on its fixture and hold that state for consecutive steps."""

    def __init__(
        self,
        tools: list[Asset],
        fixtures: list[Asset],
        goals: list[dict],
        contact_force_n: float = 0.2,
        speed_m_s: float = 0.15,
        consecutive_success_steps: int = 1,
        episode_length_s: float | None = None,
        task_description: str | None = None,
    ):
        """
        Args:
            tools: Manipulable tools; every tool must appear in exactly one goal.
            fixtures: Kinematic fixtures the tools are scored against.
            goals: One mapping per tool with ``tool``, ``fixture``, and either ``loop``/``rod``
                (plus optional ``alternative_loops``) or ``containment`` geometry.
            contact_force_n: Minimum tool-fixture contact force for a goal to count.
            speed_m_s: Maximum tool linear speed for a goal to count.
            consecutive_success_steps: Steps every goal must hold simultaneously before success.
            episode_length_s: Episode timeout.
            task_description: Natural-language instruction for the policy.
        """
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        assert contact_force_n >= 0.0, "contact_force_n must be non-negative."
        assert (
            speed_m_s > 0.0 and consecutive_success_steps > 0
        ), "speed_m_s and consecutive_success_steps must be positive."
        tool_by_name = {tool.name: tool for tool in tools}
        fixture_by_name = {fixture.name: fixture for fixture in fixtures}
        self.goals = [
            HangingGoal(tool_by_name[goal["tool"]], fixture_by_name[goal["fixture"]], goal_geometry_from_dict(goal))
            for goal in goals
        ]
        assert sorted(goal.tool.name for goal in self.goals) == sorted(tool_by_name), "Every tool needs one goal."
        self.contact_force_n = contact_force_n
        self.speed_m_s = speed_m_s
        self.consecutive_success_steps = consecutive_success_steps

    def get_scene_cfg(self):
        sensors = [
            (goal.contact_sensor_name, goal.tool.get_contact_sensor_cfg(contact_against_object=goal.fixture))
            for goal in self.goals
        ]
        return make_configclass("ToolHangingSceneCfg", [(name, type(cfg), cfg) for name, cfg in sensors])()

    def get_termination_cfg(self):
        predicates = []
        for goal in self.goals:
            predicates += [
                TerminationTermCfg(
                    func=goal_geometry_satisfied,
                    params={"tool_name": goal.tool.name, "fixture_name": goal.fixture.name, "geometry": goal.geometry},
                ),
                TerminationTermCfg(
                    func=contact_force_above,
                    params={"sensor_name": goal.contact_sensor_name, "force_threshold": self.contact_force_n},
                ),
                TerminationTermCfg(
                    func=velocity_below_threshold,
                    params={"subject_name": goal.tool.name, "linear_velocity_threshold": self.speed_m_s},
                ),
            ]
        return TerminationsCfg(
            success=TerminationTermCfg(
                func=CompositePredicate,
                params={
                    "predicates": predicates,
                    "mode": SuccessMode.ALL,
                    "consecutive_steps": self.consecutive_success_steps,
                },
            )
        )

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode):
        return None

    def get_metrics(self):
        return [SuccessRateMetric()]
