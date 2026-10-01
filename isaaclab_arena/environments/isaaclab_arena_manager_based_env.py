# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gymnasium as gym
import torch
from copy import copy
from typing import TYPE_CHECKING, Any

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg, RecorderTermCfg

from isaaclab_arena.environments.arena_world import ArenaWorld
from isaaclab_arena.environments.episode_scheduler import EpisodeScheduler
from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
    IsaacLabArenaManagerBasedRLEnvCfg,
    apply_arena_global_settings,
)
from isaaclab_arena.metrics.metric_data import MetricsDataCollection
from isaaclab_arena.metrics.metrics_manager import MetricsManager
from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderManager, EpisodeResultsRecorderCfg
from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder
from isaaclab_arena.utils.configclass import combine_configclass_instances
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
        # Recorder composition belongs to this environment, not the caller's configuration instance.
        cfg = copy(cfg)
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

    def set_episode_limit(self, episode_limit: int | None) -> None:
        """Set the episode limit before the initial reset.

        Args:
            episode_limit: Exact episode count, or None for no limit.
        """
        assert (
            episode_limit is None or self.cfg.autoreset_mode == gym.vector.AutoresetMode.DISABLED
        ), "Finite episode limits require autoreset_mode=DISABLED so the caller controls replacement episodes."
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
        has_dataset_recorders = self.cfg.recorders is not None and any(
            isinstance(term_cfg, RecorderTermCfg) for term_cfg in vars(self.cfg.recorders).values()
        )
        self.cfg.recorders = combine_configclass_instances(
            "ArenaRecorderManagerCfg",
            self.cfg.recorders,
            EpisodeResultsRecorderCfg(),
            bases=(RecorderManagerBaseCfg,),
        )
        if not has_dataset_recorders:
            self.cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE
        super().load_managers()
        self.metrics_manager = MetricsManager(self.cfg.metrics, self)
        self.episode_recorder_manager = EpisodeRecorderManager(self.cfg.episode_recorders, self)

    def get_language_instruction(self) -> str | None:
        """Return the language instruction that is passed to the policy."""
        return self.cfg.task_description

    def get_episode_index(self, env_id: int) -> int:
        """Return the current or most recent episode's index within this environment."""
        return self._episode_scheduler.get_episode_index_in_env(env_id)

    def reset(
        self,
        env_ids: torch.Tensor | slice | None = None,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """Start episodes within the remaining budget without interrupting finite runs."""
        if self._episode_scheduler.episode_limit is not None:
            candidate_env_ids = self._episode_env_ids(env_ids)
            assert not self.active_episode_mask[
                candidate_env_ids
            ].any(), "Cannot reset active environments during finite evaluation. Wait for their episodes to finish."
            env_ids = self._episode_scheduler.select_episode_start_env_ids(candidate_env_ids)
            if len(env_ids) == 0:
                return self.obs_buf, self.extras
        return super().reset(env_ids=env_ids, seed=seed, options=options)

    def reset_to(self, state, env_ids=None, seed=None, is_relative=False):
        """Restore state for an unlimited run, preserving the supplied environment order."""
        assert self._episode_scheduler.episode_limit is None, "Cannot restore states during finite evaluation."
        return super().reset_to(state, env_ids=env_ids, seed=seed, is_relative=is_relative)

    def step(self, action: torch.Tensor):
        result = super().step(action)
        if self.cfg.autoreset_mode == gym.vector.AutoresetMode.DISABLED:
            # Isaac Lab records terminal data before returning; assignments remain available until then.
            completed_env_ids = (result[2] | result[3]).nonzero().flatten()
            if len(completed_env_ids):
                self._episode_scheduler.finish_episodes(completed_env_ids)
        return result

    def _reset_idx(self, env_ids: torch.Tensor | slice) -> None:
        episode_env_ids = self._episode_env_ids(env_ids)
        previous_episode_env_ids = [
            env_id
            for env_id in episode_env_ids.tolist()
            if self._episode_scheduler.get_global_episode_index(env_id) is not None
        ]
        if previous_episode_env_ids:
            self._episode_scheduler.finish_episodes(previous_episode_env_ids)
        # Reset events draw variation samples using the new episode identities.
        started_env_ids = self._episode_scheduler.start_episodes(episode_env_ids)
        assert len(started_env_ids) == len(episode_env_ids), "The requested reset exceeds the remaining episode limit."
        super()._reset_idx(env_ids)

    def _episode_env_ids(self, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
        """Normalize an environment selection for episode assignment bookkeeping."""
        if env_ids is None:
            env_ids = slice(None)
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def compute_metrics(self) -> MetricsDataCollection:
        """Compute all registered metrics.

        Returns:
            A MetricsDataCollection instance.
        """
        return self.metrics_manager.compute()
