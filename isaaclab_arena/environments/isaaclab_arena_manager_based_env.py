# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.metrics.metric_data import MetricsDataCollection
from isaaclab_arena.metrics.metrics_manager import MetricsManager
from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderManager
from isaaclab_arena.variations.variation_recorder import VariationRecorder


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
        self._variation_recorder = variation_recorder
        if variation_recorder is not None:
            # Bind so run-time variation draws can be attributed to the current episode index.
            variation_recorder.bind_env(self)
        # Per-env count of completed episodes; advanced in ``_reset_idx``.
        self._episode_counts: dict[int, int] = {}
        # The initial reset touches every env before any episode has run; skip it.
        self._first_reset = True
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    @property
    def variation_recorder(self) -> VariationRecorder | None:
        """The recorder of variation samples, or ``None`` if the env was not built with one."""
        if self._variation_recorder is None:
            print(
                "[WARNING] variation_recorder is None; no variation samples were recorded. "
                "Build the env through ArenaEnvBuilder to record variations."
            )
        return self._variation_recorder

    @property
    def episode_recorder(self) -> EpisodeRecorderManager:
        """The per-episode recorder."""
        return self.episode_recorder_manager

    def load_managers(self) -> None:
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
            super()._reset_idx(env_ids)
            return
        # Runs recorder before super() so the just-finished episode is still intact.
        self.episode_recorder_manager.record_pre_reset(env_ids)
        # Advance before super() so reset-mode variation draws are tagged with the episode they begin.
        self._advance_episode_indices(env_ids)
        super()._reset_idx(env_ids)

    def compute_metrics(self) -> MetricsDataCollection:
        """Compute all registered metrics.

        Returns:
            A MetricsDataCollection instance.
        """
        return self.metrics_manager.compute()
