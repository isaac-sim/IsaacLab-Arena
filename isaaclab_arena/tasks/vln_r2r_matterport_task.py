# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""VLN navigation task for IsaacLab Arena.

This module provides:
  - ``VlnEpisodeCfg``:  configuration for a single VLN episode.
  - ``VlnBenchmarkCfg``: sequential episode sampler (supports multi-env).
  - ``VlnNavTask``:      TaskBase implementation that wires episodes,
                          terminations, events, and metrics together.
  - Event / termination helpers that reset the robot pose and check goal
    arrival at runtime.

Multi-env support:
  When ``num_envs > 1``, each environment gets its own episode, goal
  position, and instruction.  Episode data is stored in ``env.extras``
  with per-env arrays (e.g. ``current_goal_pos`` has shape ``[N, 3]``).
  Note that each env also loads a full copy of the Matterport scene,
  which may require significant GPU memory.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.tasks.task_base import TaskBase

from isaaclab_arena.metrics.vln_metrics import (
    DistanceToGoalMetric,
    OracleSuccessMetric,
    PathLengthMetric,
    SPLMetric,
    SuccessMetric,
)


# ========================================================================== #
# Dataset loading                                                            #
# ========================================================================== #


def read_episodes(r2r_dataset_path: str) -> List[Dict[str, Any]]:
    """Read R2R-style episodes from a gzipped JSON file.

    Expected format::

        {"episodes": [ {episode_dict}, ... ]}
    """
    with gzip.open(r2r_dataset_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return data["episodes"]


# ========================================================================== #
# Episode configuration                                                      #
# ========================================================================== #


@configclass
class VlnEpisodeCfg:
    """Configuration for a single VLN episode."""

    scene_id: str = MISSING
    episode_id: int = MISSING
    start_pos: Tuple[float, float, float] = MISSING
    start_quat_wxyz: Tuple[float, float, float, float] = MISSING
    goal_pos: Tuple[float, float, float] = MISSING
    goal_radius: float = MISSING
    geodesic_distance: float = MISSING
    instruction_text: str = MISSING
    reference_path: List[Tuple[float, float, float]] = MISSING


@configclass
class VlnBenchmarkCfg:
    """Sequential episode sampler wrapping a list of :class:`VlnEpisodeCfg`.

    Supports multi-env: ``sample_episodes(n)`` returns ``n`` episodes,
    one per environment, advancing the internal index sequentially.
    """

    episodes: list[VlnEpisodeCfg] = MISSING
    current_index: int = 0
    robot_root_height_offset: float = 0.0

    def sample_episode(self) -> VlnEpisodeCfg:
        """Sample a single episode (for single-env or per-env calls)."""
        ep = self.episodes[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.episodes)
        return ep

    def sample_episodes(self, n: int) -> list[VlnEpisodeCfg]:
        """Sample ``n`` episodes sequentially (one per env)."""
        return [self.sample_episode() for _ in range(n)]


def build_vln_episode_from_raw(raw_ep: dict) -> VlnEpisodeCfg:
    """Convert a raw R2R-style episode dict into :class:`VlnEpisodeCfg`."""
    scene_id = os.path.splitext(os.path.basename(raw_ep["scene_id"]))[0]
    start_pos = tuple(raw_ep["start_position"])
    start_quat = tuple(raw_ep["start_rotation"])
    goals = raw_ep.get("goals") or []
    if goals:
        goal_pos = tuple(goals[0]["position"])
        goal_radius = float(goals[0].get("radius", 3.0))
    else:
        goal_pos = tuple(raw_ep["reference_path"][-1])
        goal_radius = 3.0
    geodesic_distance = float(raw_ep.get("info", {}).get("geodesic_distance", 0.0))
    instruction_text = raw_ep["instruction"]["instruction_text"]
    reference_path = [tuple(p) for p in raw_ep["reference_path"]]

    return VlnEpisodeCfg(
        scene_id=scene_id,
        episode_id=int(raw_ep["episode_id"]),
        start_pos=start_pos,
        start_quat_wxyz=start_quat,
        goal_pos=goal_pos,
        goal_radius=goal_radius,
        geodesic_distance=geodesic_distance,
        instruction_text=instruction_text,
        reference_path=reference_path,
    )


# ========================================================================== #
# Termination helpers                                                        #
# ========================================================================== #


def _log_episode_end(env, reason: str, env_ids: torch.Tensor) -> None:
    """Log distance-to-goal when an episode ends."""
    if not env_ids.any():
        return
    try:
        ids = env_ids.nonzero(as_tuple=False).flatten().tolist()
        root_pos = env.scene["robot"].data.root_pos_w.cpu().numpy()
        goal_pos = env.extras.get("current_goal_pos")
        ep_ids = env.extras.get("current_episode_id")
        for i in ids:
            pos = root_pos[i]
            goal_str = ""
            if goal_pos is not None:
                g = np.asarray(goal_pos[i])
                dist = float(np.linalg.norm(pos - g))
                goal_str = f" goal=[{g[0]:.1f},{g[1]:.1f},{g[2]:.2f}] dist={dist:.2f}"
            ep_str = f" ep={ep_ids[i]}" if ep_ids is not None else ""
            print(f"[VlnNavTask] EPISODE END ({reason}) env={i}{ep_str} robot=[{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.2f}]{goal_str}")
    except Exception:
        pass


def vln_stop_term(env) -> torch.Tensor:
    """Termination term: True when the VLM policy signals STOP.

    The policy sets ``env.extras["vln_stop_called"]`` (a bool tensor of
    shape ``[N]``) when the VLM outputs a stop command (zero velocity
    and zero duration).
    """
    flag = env.extras.get("vln_stop_called")
    if flag is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    result = torch.as_tensor(flag, device=env.device, dtype=torch.bool)
    if result.any():
        _log_episode_end(env, "STOP", result)
    return result


def vln_stuck_term(env, velocity_threshold: float = 0.1, patience: int = 1000) -> torch.Tensor:
    """Termination term: True when the robot has been stuck for too long.

    A robot is considered stuck when its linear velocity magnitude stays
    below ``velocity_threshold`` for ``patience`` consecutive steps.
    Per-env counters are stored in ``env.extras["_vln_stuck_counter"]``.
    """
    num_envs = env.num_envs
    device = env.device

    root_vel = env.scene["robot"].data.root_vel_w[:num_envs, :3]  # [N, 3]
    speed = torch.norm(root_vel, dim=-1)  # [N]

    if "_vln_stuck_counter" not in env.extras:
        env.extras["_vln_stuck_counter"] = torch.zeros(num_envs, dtype=torch.long, device=device)

    counter = env.extras["_vln_stuck_counter"]
    is_slow = speed < velocity_threshold
    counter = torch.where(is_slow, counter + 1, torch.zeros_like(counter))
    env.extras["_vln_stuck_counter"] = counter

    stuck_mask = counter >= patience
    if stuck_mask.any():
        _log_episode_end(env, f"STUCK({patience}steps)", stuck_mask)
    return stuck_mask


def vln_success_term(env, position_tolerance: float = 0.3) -> torch.Tensor:
    """Termination term: True when the robot is within *position_tolerance* of the goal.

    Supports multi-env: ``env.extras["current_goal_pos"]`` should have
    shape ``[N, 3]`` where ``N = num_envs``.
    """
    num_envs = env.num_envs
    root_state = env.scene["robot"].data.root_state_w  # [N, 13]
    goal_pos_np = env.extras.get("current_goal_pos")
    if goal_pos_np is None:
        return torch.zeros(num_envs, dtype=torch.bool, device=root_state.device)

    goal_pos = torch.as_tensor(goal_pos_np, device=root_state.device, dtype=root_state.dtype)
    # Ensure shape [N, 3]
    if goal_pos.ndim == 1:
        goal_pos = goal_pos.unsqueeze(0).expand(num_envs, -1)
    elif goal_pos.shape[0] == 1 and num_envs > 1:
        goal_pos = goal_pos.expand(num_envs, -1)

    robot_pos = root_state[..., :3]
    dist = torch.linalg.norm(robot_pos - goal_pos, dim=-1)
    result = dist < position_tolerance
    if result.any():
        _log_episode_end(env, f"SUCCESS(d<{position_tolerance}m)", result)
    return result


def vln_disabled_success_term(env, position_tolerance: float = 0.0) -> torch.Tensor:
    """Disabled proximity-success termination.

    When following the embodied-navigation evaluation protocol from
    Anderson et al., episodes should only be deemed successful if the
    agent explicitly emits STOP/DONE. We therefore keep the termination
    slot but disable auto-success by default.
    """
    _ = position_tolerance
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def vln_time_out_term(env) -> torch.Tensor:
    """Wraps standard time_out to add EPISODE END logging."""
    result = mdp_isaac_lab.time_out(env)
    if result.any():
        _log_episode_end(env, "TIME_OUT", result)
    return result


@configclass
class VlnTerminationsCfg:
    """Termination terms for the VLN task."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=vln_time_out_term, time_out=True)
    success: TerminationTermCfg = MISSING
    stop: TerminationTermCfg = TerminationTermCfg(func=vln_stop_term)
    stuck: TerminationTermCfg = MISSING


# ========================================================================== #
# Event helpers (reset robot pose from dataset)                              #
# ========================================================================== #


def reset_robot_and_goal_from_vln_dataset(
    env,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
    vln_cfg: VlnBenchmarkCfg,
):
    """Event callback executed on environment reset.

    Supports multi-env: only the environments in ``env_ids`` are reset.
    Each reset env gets a new episode sampled sequentially from the dataset.
    Episode metadata is stored per-env in ``env.extras``.

    Storage layout in ``env.extras``:
      - ``current_goal_pos``:        np.ndarray shape ``[N, 3]``
      - ``current_instruction``:     list[str] length ``N``
      - ``current_reference_path``:  list[np.ndarray] length ``N``
      - ``current_scene_id``:        list[str] length ``N``
      - ``current_episode_id``:      list[int] length ``N``
    """
    num_envs = env.num_envs
    env_ids_list = env_ids.cpu().tolist()
    num_reset = len(env_ids_list)

    # Initialize per-env storage on first call
    if "current_goal_pos" not in env.extras:
        env.extras["current_goal_pos"] = np.zeros((num_envs, 3), dtype=np.float32)
        env.extras["current_instruction"] = [""] * num_envs
        env.extras["current_reference_path"] = [np.zeros((1, 3), dtype=np.float32)] * num_envs
        env.extras["current_scene_id"] = [""] * num_envs
        env.extras["current_episode_id"] = [0] * num_envs
        env.extras["vln_stop_called"] = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        env.extras["_vln_stuck_counter"] = torch.zeros(num_envs, dtype=torch.long, device=env.device)

    # Clear stop/stuck flags for reset envs
    if "vln_stop_called" in env.extras:
        env.extras["vln_stop_called"][env_ids] = False
    if "_vln_stuck_counter" in env.extras:
        env.extras["_vln_stuck_counter"][env_ids] = 0

    # Sample one episode per reset env
    episodes = vln_cfg.sample_episodes(num_reset)

    # Build batched start poses for all reset envs
    start_poses = torch.zeros(num_reset, 7, dtype=torch.float32, device=env.device)
    for i, ep in enumerate(episodes):
        start_poses[i, :3] = torch.tensor(ep.start_pos)
        start_poses[i, 2] += vln_cfg.robot_root_height_offset
        start_poses[i, 3:7] = torch.tensor(ep.start_quat_wxyz)

    # Teleport all reset robots and zero velocities in one batched call
    robot = env.scene[robot_cfg.name]
    robot.write_root_pose_to_sim(start_poses, env_ids=env_ids)
    zero_vel = torch.zeros(num_reset, 6, dtype=torch.float32, device=env.device)
    robot.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

    # Update per-env metadata
    for i, env_id in enumerate(env_ids_list):
        ep = episodes[i]
        env.extras["current_goal_pos"][env_id] = np.array(ep.goal_pos, dtype=np.float32)
        env.extras["current_instruction"][env_id] = ep.instruction_text
        env.extras["current_reference_path"][env_id] = np.array(ep.reference_path, dtype=np.float32)
        env.extras["current_scene_id"][env_id] = ep.scene_id
        env.extras["current_episode_id"][env_id] = ep.episode_id
        print(
            f"[VlnNavTask] NEW EPISODE env={env_id} ep={ep.episode_id} scene={ep.scene_id}"
            f" start=[{ep.start_pos[0]:.1f},{ep.start_pos[1]:.1f},{ep.start_pos[2]:.2f}]"
            f" root_z_offset={vln_cfg.robot_root_height_offset:.2f}"
            f" goal=[{ep.goal_pos[0]:.1f},{ep.goal_pos[1]:.1f},{ep.goal_pos[2]:.2f}]"
            f" instruction={ep.instruction_text[:80]}..."
        )


@configclass
class VlnEventsCfg:
    """Events configuration for the VLN task."""

    reset_robot_and_goal: EventTermCfg = MISSING


# ========================================================================== #
# VLN Navigation Task                                                        #
# ========================================================================== #


class VlnR2rMatterportTask(TaskBase):
    """VLN task using R2R (Room-to-Room) episodes in Matterport scenes.

    This task:
      - Reads episodes from a gzipped VLN-CE R2R dataset.
      - Provides termination (time-out, goal-reached, VLM stop, stuck).
      - Registers VLN metrics (Oracle Success, SPL, Success, PathLength, DistanceToGoal).
      - On reset, teleports the robot to the episode start pose and stores
        the instruction / goal in ``env.extras``.

    For other VLN datasets (RxR, REVERIE) or scene types (Gibson),
    create a new task class following this pattern.
    """

    def __init__(
        self,
        robot: Asset,
        r2r_dataset_path: str,
        episode_indices: Optional[Sequence[int]] = None,
        episode_length_s: float = 60.0,
        success_radius: float = 3.0,
        scene_filter: Optional[str] = None,
        robot_root_height_offset: float = 0.0,
        require_stop_for_success: bool = True,
    ):
        """
        Args:
            robot: The robot embodiment asset.
            r2r_dataset_path: Path to the gzipped R2R dataset JSON file.
            episode_indices: Optional subset of episode indices to evaluate.
            episode_length_s: Maximum episode duration in seconds.
            success_radius: Fallback distance threshold (meters) for goal
                success when the dataset episode does not define a goal radius.
            scene_filter: If set, only load episodes whose scene_id contains
                this string (e.g. ``"zsNo4HB9uLZ"``).  This is critical
                because the dataset contains episodes from many scenes but
                only one scene USD is loaded at a time.
            robot_root_height_offset: Added to dataset start_position.z when
                teleporting the robot root on reset. Useful when dataset
                positions are floor-level but the robot root is pelvis-level.
            require_stop_for_success: If True, only STOP/DONE can end a
                successful episode. If False, the benchmark falls back to
                proximity-based success termination.
        """
        super().__init__(
            episode_length_s=episode_length_s,
            task_description="Navigate to the target location following the instruction.",
        )
        self.robot = robot
        self.success_radius = success_radius
        self.require_stop_for_success = require_stop_for_success

        # Load episodes: filter by scene FIRST, then slice by index range.
        # This ensures --episode_start/end refer to indices within the
        # filtered (scene-specific) episode list, not the global dataset.
        raw_episodes = read_episodes(r2r_dataset_path)

        if scene_filter is not None:
            before = len(raw_episodes)
            raw_episodes = [ep for ep in raw_episodes if scene_filter in ep.get("scene_id", "")]
            print(
                f"[VlnR2rMatterportTask] Scene filter '{scene_filter}': {before} -> {len(raw_episodes)} episodes"
            )
            if not raw_episodes:
                raise ValueError(
                    f"No episodes match scene_filter='{scene_filter}'. "
                    f"Check that --usd_path matches the dataset scenes."
                )

        if episode_indices is not None:
            raw_episodes = [raw_episodes[i] for i in episode_indices if i < len(raw_episodes)]
            print(f"[VlnR2rMatterportTask] Episode indices: {len(raw_episodes)} episodes selected")

        vln_episodes = [build_vln_episode_from_raw(ep) for ep in raw_episodes]
        self.vln_benchmark_cfg = VlnBenchmarkCfg(
            episodes=vln_episodes,
            robot_root_height_offset=robot_root_height_offset,
        )

        # Pre-compute data for metrics
        self._gt_waypoints_per_episode: list[list[list[float]]] = [
            [list(p) for p in ep.reference_path] for ep in vln_episodes
        ]
        self._success_radius_per_episode: list[float] = [
            float(ep.goal_radius) if ep.goal_radius > 0.0 else success_radius
            for ep in vln_episodes
        ]
        self._shortest_path_distance_per_episode: list[float] | None = None
        if all(ep.geodesic_distance > 0.0 for ep in vln_episodes):
            self._shortest_path_distance_per_episode = [
                float(ep.geodesic_distance) for ep in vln_episodes
            ]

    # ------------------------------------------------------------------ #
    # TaskBase interface                                                   #
    # ------------------------------------------------------------------ #

    def get_scene_cfg(self):
        """VLN scene assets (Matterport background) are added by the Environment, not the Task."""
        return None

    def get_termination_cfg(self) -> VlnTerminationsCfg:
        success_term_func = vln_disabled_success_term if self.require_stop_for_success else vln_success_term
        return VlnTerminationsCfg(
            success=TerminationTermCfg(
                func=success_term_func,
                params={"position_tolerance": self.success_radius},
            ),
            stuck=TerminationTermCfg(
                func=vln_stuck_term,
                params={"velocity_threshold": 0.1, "patience": 1000},
            ),
        )

    def get_events_cfg(self) -> VlnEventsCfg:
        cfg = VlnEventsCfg(
            reset_robot_and_goal=EventTermCfg(
                func=reset_robot_and_goal_from_vln_dataset,
                mode="reset",
                params={
                    "robot_cfg": SceneEntityCfg("robot"),
                    "vln_cfg": self.vln_benchmark_cfg,
                },
            ),
        )
        return cfg

    def get_mimic_env_cfg(self, arm_mode):
        # VLN does not use mimic / imitation learning datagen
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [
            PathLengthMetric(),
            DistanceToGoalMetric(self._gt_waypoints_per_episode),
            OracleSuccessMetric(
                self._gt_waypoints_per_episode,
                self._success_radius_per_episode,
            ),
            SuccessMetric(
                self._gt_waypoints_per_episode,
                self._success_radius_per_episode,
                require_stop_signal=self.require_stop_for_success,
            ),
            SPLMetric(
                self._gt_waypoints_per_episode,
                self._success_radius_per_episode,
                shortest_path_distance_per_episode=self._shortest_path_distance_per_episode,
                require_stop_signal=self.require_stop_for_success,
            ),
        ]

    def get_task_description(self) -> str | None:
        """Return a static task description for the VLN benchmark.

        This returns a generic description set at init time.  It does NOT
        return the per-episode navigation instruction, because:

          1. Arena's ``policy_runner`` calls ``get_task_description()`` once
             at startup via ``task.get_task_description()``, before any
             episode is loaded.
          2. VLN instructions change every episode and are per-env.

        Per-episode instructions are stored in
        ``env.extras["current_instruction"]`` (a list of length ``num_envs``)
        and are automatically picked up by ``VlnPolicy`` on each step via
        ``_check_instruction_update()``.
        """
        return self.task_description

    def get_viewer_cfg(self) -> ViewerCfg:
        # Third-person camera behind and above the robot, looking slightly ahead.
        # Kept close to avoid clipping through Matterport walls.
        return ViewerCfg(
            eye=(-1.5, 0.0, 1.5),
            lookat=(1.0, 0.0, 0.8),
            origin_type="asset_root",
            asset_name="robot",
            env_index=0,
        )

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        return env_cfg
