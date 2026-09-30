# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg, RecorderTermCfg

from isaaclab_arena.environments.arena_world import ArenaWorld
from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
    IsaacLabArenaManagerBasedRLEnvCfg,
    apply_arena_global_settings,
)
from isaaclab_arena.environments.object_initial_rest_pose_recorder import ObjectInitialRestPoseRecorder
from isaaclab_arena.metrics.metric_data import MetricsDataCollection
from isaaclab_arena.metrics.metrics_manager import MetricsManager
from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderManager
from isaaclab_arena.terms.recorders import RecordInitialRestPoses
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
        self._object_initial_rest_pose_recorder: ObjectInitialRestPoseRecorder | None = None
        self._variation_recorder = variation_recorder
        if variation_recorder is not None:
            # Bind so run-time variation draws can be attributed to the current episode index.
            variation_recorder.bind_env(self)
        # Per-env count of completed episodes; advanced in ``_reset_idx``.
        self._episode_counts: dict[int, int] = {}
        # The initial reset has no finished episode to record or count.
        self._first_reset = True
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
        """The environment-owned recorder of each object's first resting position in an episode."""
        assert self._object_initial_rest_pose_recorder is not None, "Rest-pose recording requires loaded managers."
        return self._object_initial_rest_pose_recorder

    @property
    def episode_recorder(self) -> EpisodeRecorderManager:
        """The per-episode recorder."""
        return self.episode_recorder_manager

    def load_managers(self) -> None:
        assert self._arena_world is None, "ArenaWorld is already initialized."
        self._arena_world = ArenaWorld(self.scene)
        self._object_initial_rest_pose_recorder = ObjectInitialRestPoseRecorder(
            self.scene, self.arena_world, self.cfg.initial_rest_pose_recording
        )
        # Install after callers have selected their demonstration or evaluation recorders.
        if not self.cfg.recorders:
            self.cfg.recorders = RecorderManagerBaseCfg()
        if not any(isinstance(term, RecorderTermCfg) for term in vars(self.cfg.recorders).values()):
            # This callback stores no dataset data and must not enable file export by itself.
            self.cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE
        self.cfg.recorders.record_initial_rest_poses = RecorderTermCfg(class_type=RecordInitialRestPoses)
        super().load_managers()
        self.metrics_manager = MetricsManager(self.cfg.metrics, self)
        self.episode_recorder_manager = EpisodeRecorderManager(self.cfg.episode_recorders, self)

    def get_language_instruction(self) -> str | None:
        """Return the language instruction that is passed to the policy."""
        return self.cfg.task_description

    def get_episode_index(self, env_id: int) -> int:
        """Return the index of the current episode in ``env_id``."""
        return self._episode_counts.get(env_id, 0)

    def _advance_episode_indices(self, env_ids: Sequence[int]) -> None:
        """Advance the per-env episode counter for each episode in ``env_ids``."""
        for env_id in env_ids:
            env_id = int(env_id)
            self._episode_counts[env_id] = self._episode_counts.get(env_id, 0) + 1

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        # The initial reset touches every env before any episode has run; nothing to record or count.
        if self._first_reset:
            self._first_reset = False
        else:
            # Preserve the finished episode until recording and episode attribution are complete.
            self.episode_recorder_manager.record_pre_reset(env_ids)
            self._advance_episode_indices(env_ids)
        super()._reset_idx(env_ids)
        self.object_initial_rest_pose_recorder.reset(env_ids)

    def compute_metrics(self) -> MetricsDataCollection:
        """Compute all registered metrics.

        Returns:
            A MetricsDataCollection instance.
        """
        return self.metrics_manager.compute()
