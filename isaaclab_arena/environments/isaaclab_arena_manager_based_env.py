# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_arena.environments.arena_world import ArenaWorld
from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
    IsaacLabArenaManagerBasedRLEnvCfg,
    apply_arena_global_settings,
)
from isaaclab_arena.metrics.metric_data import MetricsDataCollection
from isaaclab_arena.metrics.metrics_manager import MetricsManager
from isaaclab_arena.recording.arena_recorder_manager import ArenaRecorderManager
from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderManager
from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder
from isaaclab_arena.variations.variation_recorder import VariationRecorder

if TYPE_CHECKING:
    from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn

    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker


class IsaacLabArenaManagerBasedRLEnv(ManagerBasedRLEnv):
    """Arena extension to ManagerBasedRLEnv that adds additional Arena-specific functionality."""

    cfg: IsaacLabArenaManagerBasedRLEnvCfg

    def __init__(
        self,
        cfg: IsaacLabArenaManagerBasedRLEnvCfg,
        render_mode: str | None = None,
        variation_recorder: VariationRecorder | None = None,
        **kwargs,
    ):
        apply_arena_global_settings()
        self._arena_world: ArenaWorld | None = None
        self._progress_tracker: ProgressTracker | None = None
        self._object_initial_rest_pose_recorder = ObjectInitialRestPoseRecorder(
            num_envs=cfg.scene.num_envs, device=cfg.sim.device
        )
        self._variation_recorder = variation_recorder
        if variation_recorder is not None:
            # Bind so run-time variation draws can be attributed to the current episode index.
            variation_recorder.bind_env(self)
        self._episode_indices: dict[int, int] = {}
        self._episode_limit: int | None = None
        self._started_episode_count = 0
        self._completed_episode_count = 0
        self._active_episode_mask = torch.zeros(cfg.scene.num_envs, dtype=torch.bool, device=cfg.sim.device)
        self._reset_env_ids = torch.empty(0, dtype=torch.long, device=cfg.sim.device)
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    @property
    def arena_world(self) -> ArenaWorld:
        """The environment's live Arena scene queries and cached geometry."""
        assert self._arena_world is not None, "ArenaWorld is unavailable before managers are loaded."
        return self._arena_world

    @property
    def progress_tracker(self) -> ProgressTracker | None:
        """The ProgressTracker owned by TaskSuccessTerm, or None if not initialized."""
        return self._progress_tracker

    @property
    def variation_recorder(self) -> VariationRecorder | None:
        """The recorder of variation samples, or ``None`` if the env was not built with one."""
        return self._variation_recorder

    @property
    def object_initial_rest_pose_recorder(self) -> ObjectInitialRestPoseRecorder:
        """The recorder of initial object rest poses. Used when object_settled predicate is enabled by task progress tracking."""
        return self._object_initial_rest_pose_recorder

    @property
    def episode_recorder(self) -> EpisodeRecorderManager:
        """The per-episode recorder."""
        return self.episode_recorder_manager

    @property
    def active_episode_mask(self) -> torch.Tensor:
        """Environments currently assigned an episode, including any replacement started this step."""
        return self._active_episode_mask

    @property
    def completed_episode_count(self) -> int:
        """Number of episodes finalized by this environment."""
        return self._completed_episode_count

    @property
    def reset_env_ids(self) -> torch.Tensor:
        """Environments that started an episode during the latest step or explicit reset."""
        return self._reset_env_ids

    def load_managers(self) -> None:
        assert self._arena_world is None, "ArenaWorld is already initialized."
        self._arena_world = ArenaWorld(self.scene)
        # Isaac Lab hardcodes RecorderManager. Defer its configuration so only the Arena
        # recorder opens a dataset. Remove this workaround when upstream provides a factory.
        recorder_cfg = self.cfg.recorders
        self.cfg.recorders = None
        try:
            super().load_managers()
        finally:
            self.cfg.recorders = recorder_cfg
        self.recorder_manager = ArenaRecorderManager(recorder_cfg, self)
        self.metrics_manager = MetricsManager(self.cfg.metrics, self)
        self.episode_recorder_manager = EpisodeRecorderManager(self.cfg.episode_recorders, self)

    def get_language_instruction(self) -> str | None:
        """Return the language instruction that is passed to the policy."""
        return self.cfg.task_description

    def get_episode_index(self, env_id: int) -> int:
        """Return the current episode index, retaining the last index when an environment becomes inactive."""
        return self._episode_indices.get(env_id, 0)

    def configure_episode_limit(self, num_episodes: int) -> None:
        """Set the total number of episodes to start and finish, before the initial reset.

        Args:
            num_episodes: Positive episode budget shared by all environments.
        """
        assert self._started_episode_count == 0, "Configure the episode limit before the initial reset."
        assert num_episodes > 0, "The episode limit must be positive."
        self._episode_limit = num_episodes

    def reset(
        self, seed: int | None = None, env_ids: Sequence[int] | None = None, options: dict | None = None
    ) -> tuple[VecEnvObs, dict]:
        """Initialize episodes, allowing subsequent explicit resets only without an episode limit."""
        assert (
            self._episode_limit is None or self._started_episode_count == 0
        ), "Episode-limited rollouts use one initial reset and automatic replacements during step()."
        self._reset_env_ids = self._reset_env_ids[:0]
        return super().reset(seed=seed, env_ids=env_ids, options=options)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Step all environments and report completions only for assigned episodes."""
        active_before_step = self._active_episode_mask.clone()
        self._reset_env_ids = self._reset_env_ids[:0]
        observations, rewards, terminated, truncated, extras = super().step(action)
        # Report an episode's completion even if its environment became inactive during this step.
        return observations, rewards, terminated & active_before_step, truncated & active_before_step, extras

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        requested_env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).sort().values
        finishing_env_ids = requested_env_ids[self._active_episode_mask[requested_env_ids]]
        if len(finishing_env_ids) > 0:
            # Record the JSONL result with the finishing episode's index
            # before starting any replacements.
            self.episode_recorder_manager.record_pre_reset(finishing_env_ids)
            self._completed_episode_count += len(finishing_env_ids)
            self._active_episode_mask[finishing_env_ids] = False

        episode_start_env_ids = requested_env_ids
        if self._episode_limit is not None:
            if self._started_episode_count > 0:
                episode_start_env_ids = finishing_env_ids
            remaining_episode_starts = self._episode_limit - self._started_episode_count
            episode_start_env_ids = episode_start_env_ids[:remaining_episode_starts]
        if len(episode_start_env_ids) == 0:
            return

        # Reset-mode variation draws must refer to the episode being started.
        for env_id in episode_start_env_ids.tolist():
            self._episode_indices[env_id] = self._episode_indices.get(env_id, -1) + 1
        self._started_episode_count += len(episode_start_env_ids)
        self._active_episode_mask[episode_start_env_ids] = True
        self._reset_env_ids = torch.cat((self._reset_env_ids, episode_start_env_ids))
        super()._reset_idx(episode_start_env_ids)

    def compute_metrics(self) -> MetricsDataCollection:
        """Compute all registered metrics.

        Returns:
            A MetricsDataCollection instance.
        """
        return self.metrics_manager.compute()
