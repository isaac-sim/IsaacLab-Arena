# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderManager
from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder
from isaaclab_arena.variations.variation_recorder import VariationRecorder

if TYPE_CHECKING:
    import torch


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
        self._object_initial_rest_pose_recorder = ObjectInitialRestPoseRecorder(
            num_envs=cfg.scene.num_envs, device=cfg.sim.device
        )
        self._variation_recorder = variation_recorder
        if variation_recorder is not None:
            # Bind so run-time variation draws can be attributed to the current episode index.
            variation_recorder.bind_env(self)
        self._episode_counts: dict[int, int] = {}
        """Per-environment episode indices; failed reset attempts may leave gaps."""
        self._started_env_ids: set[int] = set()
        """Environments whose starting state was captured successfully and may be recorded."""
        self._defer_episode_recorder_reset: bool = False
        """Defer capture until reset_to has restored its supplied scene state."""
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    @property
    def arena_world(self) -> ArenaWorld:
        """The environment's live Arena scene queries and cached geometry."""
        assert self._arena_world is not None, "ArenaWorld is unavailable before managers are loaded."
        return self._arena_world

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
        """Return the index of the current episode in ``env_id``."""
        return self._episode_counts.get(env_id, 0)

    def _start_episode_recording(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        """Mark episodes recordable only after all terms capture their starting state."""
        self.episode_recorder_manager.reset(env_ids)
        ids = range(self.num_envs) if env_ids is None else env_ids
        self._started_env_ids.update(int(env_id) for env_id in ids)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        ids = [int(env_id) for env_id in env_ids]
        finished_env_ids = [env_id for env_id in ids if env_id in self._started_env_ids]
        # Clear before recording or reset events can raise, so retries cannot write stale or duplicate JSONL rows.
        self._started_env_ids.difference_update(ids)
        if finished_env_ids:
            # Record the finishing episode before reset changes its state and index.
            self.episode_recorder_manager.record_pre_reset(finished_env_ids)
        # Reserve an index before reset events, including retries, so variation draws never reuse one.
        for env_id in ids:
            self._episode_counts[env_id] = self._episode_counts.get(env_id, -1) + 1
        super()._reset_idx(env_ids)
        if not self._defer_episode_recorder_reset:
            self._start_episode_recording(env_ids)

    def reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: Sequence[int] | torch.Tensor | None,
        seed: int | None = None,
        is_relative: bool = False,
    ):
        """Restore supplied scene state before resetting episode recorder terms.

        Args:
            state: Scene state in the format returned by scene.get_state().
            env_ids: Environments to reset, or None for all environments.
            seed: Optional seed for the reset.
            is_relative: Whether supplied poses are relative to environment origins.

        Returns:
            Observations and extras from the reset.
        """
        # The base implementation calls _reset_idx before applying the supplied state.
        self._defer_episode_recorder_reset = True
        try:
            result = super().reset_to(state, env_ids, seed=seed, is_relative=is_relative)
        finally:
            self._defer_episode_recorder_reset = False
        # Failed restores remain unrecordable; a later successful reset starts a fresh episode.
        self._start_episode_recording(env_ids)
        return result

    def compute_metrics(self) -> MetricsDataCollection:
        """Compute all registered metrics.

        Returns:
            A MetricsDataCollection instance.
        """
        return self.metrics_manager.compute()
