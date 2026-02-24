# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Standard Vision-Language Navigation metrics.

Metrics implemented:
  - PathLength:       mean cumulative Euclidean distance traversed in XY.
  - DistanceToGoal:   approximate geodesic distance from the final robot
                      position to the goal, estimated via the ground-truth
                      reference path and a KDTree.
  - OracleSuccess:    fraction of episodes where the trajectory comes within
                      the success radius of the goal at least once.
  - Success:          fraction of episodes where the agent explicitly
                      issues STOP and the final DistanceToGoal < radius.
  - SPL:              Success weighted by path efficiency.

All metrics rely on a single recorder that logs the robot base position
plus whether the agent has emitted STOP at every simulation step.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
from scipy.spatial import KDTree

from isaaclab.envs.manager_based_rl_env import ManagerBasedEnv
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.metrics.metric_base import MetricBase


# ========================================================================== #
# Recorder: robot position per step                                          #
# ========================================================================== #


class RobotPositionRecorder(RecorderTerm):
    """Record robot root position plus STOP state after each step.

    The exported tensor has shape ``[N, 4]`` and stores:
      - ``[..., 0:3]``: world-frame root position ``[x, y, z]``
      - ``[..., 3]``:   STOP flag as ``0.0`` or ``1.0``

    Keeping the STOP state in the same recorder lets offline metrics follow
    the Anderson et al. protocol: an episode is successful only if the agent
    explicitly signals completion.
    """

    name = "robot_position_w"

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def record_post_step(self):
        """Return (key, data) after each simulation step."""
        root_pos = self._env.scene["robot"].data.root_pos_w
        stop_flag = self._env.extras.get("vln_stop_called")
        if stop_flag is None:
            stop_flag = np.zeros((self._env.num_envs,), dtype=np.float32)
        if hasattr(stop_flag, "to"):
            stop_tensor = stop_flag.to(device=root_pos.device, dtype=root_pos.dtype)
        else:
            stop_tensor = torch.as_tensor(stop_flag, device=root_pos.device, dtype=root_pos.dtype)
        stop_tensor = stop_tensor.unsqueeze(-1)
        return self.name, torch.cat((root_pos, stop_tensor), dim=-1).clone()


@configclass
class RobotPositionRecorderCfg(RecorderTermCfg):
    """Config for :class:`RobotPositionRecorder`."""

    class_type: type[RecorderTerm] = RobotPositionRecorder


# ========================================================================== #
# Helpers                                                                    #
# ========================================================================== #


def _episode_state_array(episode_data: np.ndarray) -> np.ndarray:
    """Normalize recorded per-step state data to ``[T, D]``."""
    arr = np.asarray(episode_data)
    if arr.ndim == 3:
        # [T, num_envs, D] -> take env 0
        arr = arr[:, 0, :]
    return arr


def _episode_pos_array(episode_data: np.ndarray) -> np.ndarray:
    """Extract recorded positions as an array of shape ``[T, 3]``."""
    arr = _episode_state_array(episode_data)
    if arr.ndim != 2 or arr.shape[-1] < 3:
        raise ValueError(f"Expected episode data with at least 3 columns, got shape {arr.shape}")
    return arr[:, :3]


def _episode_stop_called(episode_data: np.ndarray) -> bool | None:
    """Return whether STOP was emitted in this episode.

    Returns ``None`` when reading legacy recordings that do not contain the
    STOP flag column. New recordings export STOP in column 3.
    """
    arr = _episode_state_array(episode_data)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[-1] < 4:
        return None
    return bool(np.any(arr[:, 3] > 0.5))


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


def _path_length_xy(pos: np.ndarray) -> float:
    """Return cumulative XY path length for one episode."""
    pos_xy = _xy(pos)
    if pos_xy.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(pos_xy[1:] - pos_xy[:-1], axis=-1).sum())


def _reference_path_suffix_lengths(gt_wps_xy: np.ndarray) -> np.ndarray:
    """Return suffix lengths from each waypoint to the goal."""
    if gt_wps_xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    suffix = np.zeros((gt_wps_xy.shape[0],), dtype=np.float32)
    if gt_wps_xy.shape[0] > 1:
        segment_lengths = np.linalg.norm(gt_wps_xy[1:] - gt_wps_xy[:-1], axis=-1)
        suffix[:-1] = np.cumsum(segment_lengths[::-1], dtype=np.float32)[::-1]
    return suffix


def _approx_goal_distances_along_reference(pos: np.ndarray, gt_wps: np.ndarray) -> np.ndarray:
    """Approximate geodesic distances from episode positions to the goal.

    For each recorded position, we:
      1. Find the closest reference waypoint in XY via KDTree.
      2. Add the remaining reference-path suffix length from that waypoint.
    """
    pos_xy = _xy(pos)
    if pos_xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    gt_wps_xy = _xy(gt_wps)
    if gt_wps_xy.shape[0] == 0:
        return np.zeros((pos_xy.shape[0],), dtype=np.float32)

    tree = KDTree(gt_wps_xy)
    suffix_lengths = _reference_path_suffix_lengths(gt_wps_xy)
    closest_dist, closest_idx = tree.query(pos_xy)
    closest_idx = np.asarray(closest_idx, dtype=np.int64)
    return np.asarray(closest_dist, dtype=np.float32) + suffix_lengths[closest_idx]


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
            lengths.append(_path_length_xy(pos))
        return float(np.mean(lengths))


# ========================================================================== #
# Distance-To-Goal                                                           #
# ========================================================================== #


class DistanceToGoalMetric(MetricBase):
    """Approximate geodesic distance from the final robot position to the goal.

    Habitat-Lab defines distance-to-goal as a simulator geodesic query. In the
    current Isaac Sim benchmark we do not yet have an equivalent navmesh
    backend, so we estimate the distance by:
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
            dists = _approx_goal_distances_along_reference(pos, self._gt_waypoints[ep_idx])
            distances.append(float(dists[-1]) if dists.size > 0 else 0.0)

        return float(np.mean(distances))


# ========================================================================== #
# Oracle Success                                                             #
# ========================================================================== #


class OracleSuccessMetric(MetricBase):
    """Oracle Success from Habitat/VLN-CE-style distance-to-goal traces.

    This follows the standard embodied-navigation definition:

    ``oracle_success = 1[min_t d_t < r]``

    where ``d_t`` is the per-step distance-to-goal measure and ``r`` is the
    success radius. In this Arena benchmark, ``d_t`` currently comes from the
    same approximate distance surrogate used by :class:`DistanceToGoalMetric`.
    """

    name = "oracle_success"
    recorder_term_name = RobotPositionRecorder.name

    def __init__(
        self,
        gt_waypoints_per_episode: Sequence[Sequence[Sequence[float]]],
        success_radius_per_episode: Sequence[float],
    ):
        super().__init__()
        assert len(gt_waypoints_per_episode) == len(success_radius_per_episode)
        self._gt_waypoints = [np.asarray(wps, dtype=np.float32) for wps in gt_waypoints_per_episode]
        self._radii = list(success_radius_per_episode)

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return RobotPositionRecorderCfg()

    def compute_metric_from_recording(self, recorded_metric_data: List[np.ndarray]) -> float:
        if not recorded_metric_data:
            return 0.0
        num_eps = min(len(recorded_metric_data), len(self._radii))

        successes: List[float] = []
        for ep_idx in range(num_eps):
            pos = _episode_pos_array(recorded_metric_data[ep_idx])
            dists = _approx_goal_distances_along_reference(pos, self._gt_waypoints[ep_idx])
            if dists.size == 0:
                successes.append(0.0)
                continue
            oracle_dist = float(np.min(dists))
            successes.append(1.0 if oracle_dist < self._radii[ep_idx] else 0.0)
        return float(np.mean(successes))


# ========================================================================== #
# Success                                                                    #
# ========================================================================== #


class SuccessMetric(MetricBase):
    """Fraction of episodes that STOP near the goal.

    By default this follows the embodied-navigation recommendation from
    Anderson et al.: an episode is successful only if the agent explicitly
    emits STOP and its final distance-to-goal is within the success radius.
    In this benchmark the final distance is currently the same approximate
    surrogate used by :class:`DistanceToGoalMetric`.
    """

    name = "success"
    recorder_term_name = RobotPositionRecorder.name

    def __init__(
        self,
        gt_waypoints_per_episode: Sequence[Sequence[Sequence[float]]],
        success_radius_per_episode: Sequence[float],
        require_stop_signal: bool = True,
    ):
        super().__init__()
        assert len(gt_waypoints_per_episode) == len(success_radius_per_episode)
        self._dist_metric = DistanceToGoalMetric(gt_waypoints_per_episode)
        self._radii = list(success_radius_per_episode)
        self._require_stop_signal = require_stop_signal

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return self._dist_metric.get_recorder_term_cfg()

    def _compute_episode_success(self, ep_idx: int, ep_data: np.ndarray) -> float:
        """Return success for one episode using the final recorded position."""
        stop_called = _episode_stop_called(ep_data)
        if self._require_stop_signal and stop_called is False:
            return 0.0

        pos = _episode_pos_array(ep_data)
        dists = _approx_goal_distances_along_reference(pos, self._dist_metric._gt_waypoints[ep_idx])
        if dists.size == 0:
            return 0.0
        final_dist = float(dists[-1])
        return 1.0 if final_dist < self._radii[ep_idx] else 0.0

    def compute_metric_from_recording(self, recorded_metric_data: List[np.ndarray]) -> float:
        if not recorded_metric_data:
            return 0.0
        num_eps = min(len(recorded_metric_data), len(self._radii))

        successes: List[float] = []
        for ep_idx in range(num_eps):
            successes.append(self._compute_episode_success(ep_idx, recorded_metric_data[ep_idx]))
        return float(np.mean(successes))


# ========================================================================== #
# SPL (Success weighted by Path Length)                                      #
# ========================================================================== #


class SPLMetric(MetricBase):
    r"""SPL = mean_i [ S_i * d_i / max(d_i, l_i) ].

    Where:
      - S_i: binary success indicator
      - d_i: shortest-path distance from episode start to goal
      - l_i: actual path length
    """

    name = "spl"
    recorder_term_name = RobotPositionRecorder.name

    def __init__(
        self,
        gt_waypoints_per_episode: Sequence[Sequence[Sequence[float]]],
        success_radius_per_episode: Sequence[float],
        shortest_path_distance_per_episode: Sequence[float] | None = None,
        require_stop_signal: bool = True,
    ):
        super().__init__()
        assert len(gt_waypoints_per_episode) == len(success_radius_per_episode)

        self._shortest: List[float] = []
        if shortest_path_distance_per_episode is not None:
            assert len(shortest_path_distance_per_episode) == len(gt_waypoints_per_episode)
            self._shortest = [max(0.0, float(d)) for d in shortest_path_distance_per_episode]
        else:
            # Fallback for datasets that do not expose the official start-goal
            # geodesic distance. This uses the reference path length as a proxy.
            for wps in gt_waypoints_per_episode:
                arr = _xy(np.asarray(wps, dtype=np.float32))
                suffix_lengths = _reference_path_suffix_lengths(arr)
                self._shortest.append(float(suffix_lengths[0]) if suffix_lengths.size > 0 else 0.0)

        self._success_metric = SuccessMetric(
            gt_waypoints_per_episode,
            success_radius_per_episode,
            require_stop_signal=require_stop_signal,
        )

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
            l_i = _path_length_xy(pos)

            # Success S_i
            s_i = self._success_metric._compute_episode_success(i, ep_data)

            d_i = self._shortest[i]
            if d_i <= 0.0:
                spl_values.append(0.0)
            else:
                spl_values.append(float(s_i * d_i / max(d_i, l_i if l_i > 0.0 else d_i)))

        return float(np.mean(spl_values))
