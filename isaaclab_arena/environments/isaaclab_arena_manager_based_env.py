# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.evaluation.episode_results_recorder import EpisodeResultsRecorder
from isaaclab_arena.metrics.metric_data import MetricsDataCollection
from isaaclab_arena.metrics.metrics_manager import MetricsManager
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
        # Empty recorder that only buffers per-episode captures. The caller sets its run-level
        # metadata after the fact and requests a write (passing in the path) when done.
        self._episode_results_recorder = EpisodeResultsRecorder()
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
    def episode_results_recorder(self) -> EpisodeResultsRecorder:
        """The per-episode results recorder; set its ``metadata`` and call ``write(path)`` to persist."""
        return self._episode_results_recorder

    def load_managers(self) -> None:
        super().load_managers()
        self.metrics_manager = MetricsManager(self.cfg.metrics, self)

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        # Capture BEFORE super() runs reset events (placement/variation) and resets the
        # termination/episode-length buffers, so the just-finished episode is still intact.
        self._episode_results_recorder.record_episode(self, env_ids)
        super()._reset_idx(env_ids)

    def compute_metrics(self) -> MetricsDataCollection:
        """Compute all registered metrics.

        Returns:
            A MetricsDataCollection instance.
        """
        return self.metrics_manager.compute()
