# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Standard Vision-Language Navigation metrics.

Metrics implemented:
  - PathLength:       mean cumulative Euclidean distance traversed.
  - DistanceToGoal:   geodesic distance from the final robot position to
                      the goal, estimated via the ground-truth reference
                      path and a KDTree.
  - Success:          fraction of episodes where DistanceToGoal < radius.
  - SPL:              Success weighted by (shortest-path / actual-path).

All metrics rely on a single ``RobotPositionRecorder`` that logs the
robot base position at every simulation step.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from scipy.spatial import KDTree

from isaaclab.envs.manager_based_rl_env import ManagerBasedEnv
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.metrics.metric_base import MetricBase


# ========================================================================== #
# Recorder: robot position per step                                          #
# ========================================================================== #


class RobotPositionRecorder(RecorderTerm):
    """Records the world-frame robot base position ``[x, y, z]`` each step."""

    name = "robot_position_w"

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def record_post_step(self):
        """Return (key, data) after each simulation step.

        Returns a torch.Tensor (not numpy) because IsaacLab's
        RecorderManager calls ``.clone()`` on the recorded values.
        """
        return self.name, self._env.scene["robot"].data.root_pos_w.clone()


@configclass
class RobotPositionRecorderCfg(RecorderTermCfg):
    """Config for :class:`RobotPositionRecorder`."""

    class_type: type[RecorderTerm] = RobotPositionRecorder


# ========================================================================== #
# Helpers                                                                    #
# ========================================================================== #


def _episode_pos_array(episode_data: np.ndarray) -> np.ndarray:
    """Normalize recorded position data to ``[T, 3]``."""
    arr = np.asarray(episode_data)
    if arr.ndim == 3:
        # [T, num_envs, 3] -> take env 0
        arr = arr[:, 0, :]
    return arr


def _xy(arr: np.ndarray) -> np.ndarray:
    """Project positions to the XY (horizontal) plane.

    **Why XY only?**
    The robot's ``root_pos_w`` reports the *pelvis* height (~0.9m above
    ground), while the reference-path waypoints from the VLN-CE dataset
    record *floor-level* positions (~0.17m).  Using full 3-D distance
    would inflate the KDTree matching error and make geodesic distances
    unreliable.

    **Current dataset**: ``vln_ce_isaac_v1.json.gz`` contains only
    single-floor navigation episodes (start-z values vary by < 0.1m
    within each scene).  XY distance is the standard metric for
    single-floor VLN benchmarks (consistent with Habitat VLN-CE).

    **Future multi-floor support**: If episodes with stairs or
    multiple floors are added, this function should be replaced by
    a proper 3-D geodesic computation that accounts for the
    pelvis-to-floor offset (subtract ~0.88m from robot z before
    comparing with waypoint z).
    """
    return arr[..., :2]


# ========================================================================== #
# Path Length                                                                #
# ========================================================================== #


class PathLengthMetric(MetricBase):
    """Mean cumulative Euclidean path length over recorded episodes."""

    name = "path_length"
    recorder_term_name = RobotPositionRecorder.name

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return RobotPositionRecorderCfg()

    def compute_metric_from_recording(self, recorded_metric_data: List[np.ndarray]) -> float:
        if not recorded_metric_data:
            return 0.0

        lengths: List[float] = []
        for ep_data in recorded_metric_data:
            pos = _episode_pos_array(ep_data)
            if pos.shape[0] < 2:
                lengths.append(0.0)
                continue
            pos_xy = _xy(pos)
            diffs = pos_xy[1:] - pos_xy[:-1]
            lengths.append(float(np.linalg.norm(diffs, axis=-1).sum()))
        return float(np.mean(lengths))


# ========================================================================== #
# Distance-To-Goal                                                           #
# ========================================================================== #


class DistanceToGoalMetric(MetricBase):
    """Geodesic distance from the final robot position to the goal.

    The distance is estimated by:
      1. Finding the closest ground-truth waypoint to the robot via KDTree.
      2. Summing the segment lengths from that waypoint to the last waypoint.
    """

    name = "distance_to_goal"
    recorder_term_name = RobotPositionRecorder.name

    def __init__(self, gt_waypoints_per_episode: Sequence[Sequence[Sequence[float]]]):
        """
        Args:
            gt_waypoints_per_episode: For each episode ``i``, a list of 3-D
                waypoints ``[[x, y, z], ...]`` describing the reference path.
        """
        super().__init__()
        self._gt_waypoints = [np.asarray(wps, dtype=np.float32) for wps in gt_waypoints_per_episode]

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return RobotPositionRecorderCfg()

    def compute_metric_from_recording(self, recorded_metric_data: List[np.ndarray]) -> float:
        if not recorded_metric_data:
            return 0.0
        num_eps = min(len(recorded_metric_data), len(self._gt_waypoints))

        distances: List[float] = []
        for ep_idx in range(num_eps):
            ep_data = recorded_metric_data[ep_idx]
            pos = _episode_pos_array(ep_data)
            if pos.shape[0] == 0:
                distances.append(0.0)
                continue

            final_pos_xy = _xy(pos[-1])
            gt_wps = self._gt_waypoints[ep_idx]
            gt_wps_xy = _xy(gt_wps)
            tree = KDTree(gt_wps_xy)
            closest_dist, closest_idx = tree.query(final_pos_xy)
            total = float(closest_dist)
            for i in range(closest_idx, len(gt_wps_xy) - 1):
                total += float(np.linalg.norm(gt_wps_xy[i + 1] - gt_wps_xy[i]))
            distances.append(total)

        return float(np.mean(distances))


# ========================================================================== #
# Success                                                                    #
# ========================================================================== #


class SuccessMetric(MetricBase):
    """Fraction of episodes where the final distance-to-goal < radius."""

    name = "success"
    recorder_term_name = RobotPositionRecorder.name

    def __init__(
        self,
        gt_waypoints_per_episode: Sequence[Sequence[Sequence[float]]],
        success_radius_per_episode: Sequence[float],
    ):
        super().__init__()
        assert len(gt_waypoints_per_episode) == len(success_radius_per_episode)
        self._dist_metric = DistanceToGoalMetric(gt_waypoints_per_episode)
        self._radii = list(success_radius_per_episode)

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return self._dist_metric.get_recorder_term_cfg()

    def compute_metric_from_recording(self, recorded_metric_data: List[np.ndarray]) -> float:
        if not recorded_metric_data:
            return 0.0
        num_eps = min(len(recorded_metric_data), len(self._radii))

        successes: List[float] = []
        for ep_idx in range(num_eps):
            ep_data = recorded_metric_data[ep_idx]
            tmp = DistanceToGoalMetric([self._dist_metric._gt_waypoints[ep_idx]])
            d = tmp.compute_metric_from_recording([ep_data])
            successes.append(1.0 if d < self._radii[ep_idx] else 0.0)
        return float(np.mean(successes))


# ========================================================================== #
# SPL (Success weighted by Path Length)                                      #
# ========================================================================== #


class SPLMetric(MetricBase):
    r"""SPL = mean_i [ S_i * d_i / max(d_i, l_i) ].

    Where:
      - S_i: binary success indicator
      - d_i: shortest-path distance (sum of GT waypoint segments)
      - l_i: actual path length
    """

    name = "spl"
    recorder_term_name = RobotPositionRecorder.name

    def __init__(
        self,
        gt_waypoints_per_episode: Sequence[Sequence[Sequence[float]]],
        success_radius_per_episode: Sequence[float],
    ):
        super().__init__()
        assert len(gt_waypoints_per_episode) == len(success_radius_per_episode)

        # Pre-compute d_i (shortest path from start to goal along GT waypoints)
        self._shortest: List[float] = []
        for wps in gt_waypoints_per_episode:
            arr = _xy(np.asarray(wps, dtype=np.float32))
            if arr.shape[0] < 2:
                self._shortest.append(0.0)
            else:
                self._shortest.append(float(np.linalg.norm(arr[1:] - arr[:-1], axis=-1).sum()))

        self._success_metric = SuccessMetric(gt_waypoints_per_episode, success_radius_per_episode)

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return RobotPositionRecorderCfg()

    def compute_metric_from_recording(self, recorded_metric_data: List[np.ndarray]) -> float:
        if not recorded_metric_data:
            return 0.0
        num_eps = min(len(recorded_metric_data), len(self._shortest))

        spl_values: List[float] = []
        for i in range(num_eps):
            ep_data = recorded_metric_data[i]
            pos = _episode_pos_array(ep_data)

            # Actual path length l_i (XY)
            pos_xy = _xy(pos)
            if pos_xy.shape[0] < 2:
                l_i = 0.0
            else:
                l_i = float(np.linalg.norm(pos_xy[1:] - pos_xy[:-1], axis=-1).sum())

            # Success S_i
            tmp_succ = SuccessMetric(
                [self._success_metric._dist_metric._gt_waypoints[i]],
                [self._success_metric._radii[i]],
            )
            s_i = tmp_succ.compute_metric_from_recording([ep_data])

            d_i = self._shortest[i]
            if d_i <= 0.0:
                spl_values.append(0.0)
            else:
                spl_values.append(float(s_i * d_i / max(d_i, l_i if l_i > 0.0 else d_i)))

        return float(np.mean(spl_values))
