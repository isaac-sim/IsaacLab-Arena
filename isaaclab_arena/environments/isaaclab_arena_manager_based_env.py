# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.metrics.metric_data import MetricsDataCollection
from isaaclab_arena.metrics.metrics_manager import MetricsManager
from isaaclab_arena.variations.variations_recorder import VariationRecorder


class IsaacLabArenaManagerBasedRLEnv(ManagerBasedRLEnv):
    """Arena extension to ManagerBasedRLEnv that adds metrics."""

    cfg: IsaacLabArenaManagerBasedRLEnvCfg

    def __init__(
        self,
        cfg: IsaacLabArenaManagerBasedRLEnvCfg,
        render_mode: str | None = None,
        variations_recorder: VariationRecorder | None = None,
        **kwargs,
    ):
        assert variations_recorder is not None, (
            "IsaacLabArenaManagerBasedRLEnv requires a variations_recorder. Build the env through "
            "ArenaEnvBuilder (which creates one in compose_manager_cfg and threads it via env_kwargs), "
            "or pass env_kwargs into gym.make when constructing the env directly."
        )
        # Stored before super().__init__ because load_managers() runs inside it.
        self.variations_recorder = variations_recorder
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def load_managers(self) -> None:
        super().load_managers()
        self.metrics_manager = MetricsManager(self.cfg.metrics, self)

    def compute_metrics(self) -> MetricsDataCollection:
        """Compute all registered metrics.

        Returns:
            A MetricsDataCollection instance.
        """
        return self.metrics_manager.compute()
