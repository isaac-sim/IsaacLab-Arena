# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Goal-reached metric for deformable lift evaluation."""

from __future__ import annotations

import numpy as np
import torch

import warp as wp
from isaaclab.envs.manager_based_rl_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import combine_frame_transforms

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg


def _deformable_goal_reached(
    env: ManagerBasedEnv,
    *,
    command_name: str,
    minimal_height: float,
    position_tolerance: float,
    robot_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    desired_pos_b = command[:, :3]
    desired_pos_w, _ = combine_frame_transforms(
        wp.to_torch(robot.data.root_pos_w),
        wp.to_torch(robot.data.root_quat_w),
        desired_pos_b,
    )
    com_w = wp.to_torch(asset.data.root_pos_w)
    distance = torch.linalg.norm(desired_pos_w - com_w, dim=1)
    return (com_w[:, 2] > minimal_height) & (distance < position_tolerance)


class DeformableGoalReachedRecorder(RecorderTerm):
    """Record whether the deformable reached its command goal at any point in the episode."""

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.name = cfg.name
        self.command_name = cfg.command_name
        self.minimal_height = cfg.minimal_height
        self.position_tolerance = cfg.position_tolerance
        self.robot_cfg = cfg.robot_cfg
        self.asset_cfg = cfg.asset_cfg
        self._ever_reached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._first_reset = True

    def _update_state(self) -> None:
        self._ever_reached |= _deformable_goal_reached(
            self._env,
            command_name=self.command_name,
            minimal_height=self.minimal_height,
            position_tolerance=self.position_tolerance,
            robot_cfg=self.robot_cfg,
            asset_cfg=self.asset_cfg,
        )

    def record_post_step(self):
        self._update_state()
        return None, None

    def record_pre_reset(self, env_ids):
        if self._first_reset:
            self._first_reset = False
            return None, None
        self._update_state()
        reached = self._ever_reached[env_ids].clone()
        self._ever_reached[env_ids] = False
        return self.name, reached


@configclass
class DeformableGoalReachedRecorderCfg(RecorderTermCfg):
    """Recorder config for the deformable goal-reached metric."""

    class_type: type[RecorderTerm] = DeformableGoalReachedRecorder
    name: str = "deformable_goal_reached"
    command_name: str = "deformable_pose"
    minimal_height: float = 0.075
    position_tolerance: float = 0.05
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable")


def compute_deformable_goal_reached_rate(recorded_metric_data: list[np.ndarray]) -> float:
    """Compute the fraction of episodes whose deformable reached the commanded goal."""
    if len(recorded_metric_data) == 0:
        return 0.0
    goal_reached = np.concatenate([np.asarray(data, dtype=bool).reshape(-1) for data in recorded_metric_data])
    if goal_reached.size == 0:
        return 0.0
    return float(np.mean(goal_reached))


class DeformableGoalReachedRateMetric(MetricBase):
    """Non-terminating goal-reached rate for deformable lift evaluation."""

    name = "deformable_goal_reached_rate"
    recorder_term_name = "deformable_goal_reached"

    def __init__(
        self,
        command_name: str = "deformable_pose",
        minimal_height: float = 0.075,
        position_tolerance: float = 0.05,
        robot_cfg: SceneEntityCfg | None = None,
        asset_cfg: SceneEntityCfg | None = None,
    ):
        self.command_name = command_name
        self.minimal_height = minimal_height
        self.position_tolerance = position_tolerance
        self.robot_cfg = robot_cfg if robot_cfg is not None else SceneEntityCfg("robot")
        self.asset_cfg = asset_cfg if asset_cfg is not None else SceneEntityCfg("deformable")

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return DeformableGoalReachedRecorderCfg(
            name=self.recorder_term_name,
            command_name=self.command_name,
            minimal_height=self.minimal_height,
            position_tolerance=self.position_tolerance,
            robot_cfg=self.robot_cfg,
            asset_cfg=self.asset_cfg,
        )

    def get_metric_term_cfg(self) -> MetricTermCfg:
        return MetricTermCfg(
            compute_metric_func=compute_deformable_goal_reached_rate,
            params={},
            recorder_term_name=self.recorder_term_name,
        )
