# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_arena.environments.arena_world import ArenaWorld
from isaaclab_arena.environments.episode_scheduler import EpisodeScheduler
from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
    IsaacLabArenaManagerBasedRLEnvCfg,
    apply_arena_global_settings,
)
from isaaclab_arena.metrics.metric_data import MetricsDataCollection
from isaaclab_arena.metrics.metrics_manager import MetricsManager
from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderManager
from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder
from isaaclab_arena.variations.variation_recorder import VariationRecorder

if TYPE_CHECKING:
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
        self._episode_scheduler = EpisodeScheduler(
            num_envs=cfg.scene.num_envs,
            device=cfg.sim.device,
        )
        if variation_recorder is not None:
            # Bind so run-time variation draws can be attributed to the current episode index.
            variation_recorder.bind_env(self)
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
    def episode_scheduler(self) -> EpisodeScheduler:
        """Episode assignments and progress; only this environment changes them."""
        return self._episode_scheduler

    @property
    def active_episode_mask(self) -> torch.Tensor:
        """Parallel environments currently assigned an episode."""
        return self._episode_scheduler.active_episode_mask

    def set_episode_limit(self, episode_limit: int | None) -> None:
        """Set the episode limit before the initial reset.

        Args:
            episode_limit: Exact episode count, or None for no limit.
        """
        self._episode_scheduler.set_episode_limit(episode_limit)

    @property
    def object_initial_rest_pose_recorder(self) -> ObjectInitialRestPoseRecorder:
        """The recorder of initial object rest poses. Used when object_settled predicate is enabled by task progress tracking."""
        return self._object_initial_rest_pose_recorder

    @property
    def episode_recorder(self) -> EpisodeRecorderManager:
        """The per-episode recorder."""
        return self.episode_recorder_manager

    def load_managers(self) -> None:
        assert self._arena_world is None, "ArenaWorld is already initialized."
        self._arena_world = ArenaWorld(self.scene)
        super().load_managers()
        self.metrics_manager = MetricsManager(self.cfg.metrics, self)
        self.episode_recorder_manager = EpisodeRecorderManager(self.cfg.episode_recorders, self)

    def get_language_instruction(self) -> str | None:
        """Return the language instruction that is passed to the policy."""
        return self.cfg.task_description

    def get_episode_index(self, env_id: int) -> int:
        """Return the current or most recent episode's index within this environment."""
        return self._episode_scheduler.get_episode_index_in_env(env_id)

    def _finish_episodes(self, env_ids: torch.Tensor) -> None:
        completed_env_ids = env_ids[self.active_episode_mask[env_ids]]
        if not len(completed_env_ids):
            return
        # Both recorders need the terminal state and the finishing episode's assignment.
        self.episode_recorder_manager.finish_episodes(completed_env_ids)
        super()._finish_episodes(completed_env_ids)
        self._episode_scheduler.finish_episodes(completed_env_ids)

    def _select_episode_start_env_ids(self, candidate_env_ids: torch.Tensor) -> torch.Tensor:
        return self._episode_scheduler.start_episodes(candidate_env_ids)

    def _validate_reset_request(self, reset_kind: str) -> None:
        assert self._episode_scheduler.episode_limit is None or (
            self._episode_scheduler.num_episodes_started == 0 and reset_kind == "reset"
        ), (
            f"Cannot request {reset_kind} during finite evaluation; only the initial reset is allowed. "
            "Create a new environment for a new run."
        )

    def compute_metrics(self) -> MetricsDataCollection:
        """Compute all registered metrics.

        Returns:
            A MetricsDataCollection instance.
        """
        return self.metrics_manager.compute()
