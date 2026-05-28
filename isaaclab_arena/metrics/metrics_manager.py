# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any

from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg
from isaaclab_arena.metrics.metrics import (
    get_metric_recorder_dataset_path,
    get_num_episodes,
    get_recorded_metric_data,
)

if TYPE_CHECKING:
    from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv


class MetricsManager:
    """Run-time manager that computes metrics offline from a recorded HDF5 dataset.

    Mirrors the Isaac Lab manager pattern (:class:`isaaclab.managers.RewardManager`,
    :class:`isaaclab.managers.TerminationManager`, ...) but is intentionally lightweight:
    metric computation runs *after* a rollout has finished, so there is no need for
    per-step hooks or the full :class:`isaaclab.managers.ManagerBase` machinery.

    The manager parses a configclass container of :class:`MetricTermCfg` instances --
    one field per metric -- and exposes :meth:`compute` to load the recorded data for
    each term and invoke ``term_cfg.func(recorded_data, **term_cfg.params)``.
    """

    def __init__(self, cfg: object | None, env: ManagerBasedRLEnv):
        """Initialize the metrics manager.

        Args:
            cfg: A configclass container with one ``MetricTermCfg`` field per metric,
                or ``None`` if the environment has no metrics.
            env: The environment owning this manager. Used at compute time to locate
                the recorder dataset.
        """
        self._env = env
        self._term_names: list[str] = []
        self._term_cfgs: list[MetricTermCfg] = []
        self._prepare_terms(cfg)

    def _prepare_terms(self, cfg: object | None) -> None:
        if cfg is None:
            return
        cfg_items = cfg.items() if isinstance(cfg, dict) else cfg.__dict__.items()
        for term_name, term_cfg in cfg_items:
            if term_cfg is None:
                continue
            if not isinstance(term_cfg, MetricTermCfg):
                raise TypeError(
                    f"Configuration for the metric term '{term_name}' is not of type MetricTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)

    @property
    def active_terms(self) -> list[str]:
        """Names of the metric terms registered on this manager."""
        return list(self._term_names)

    def compute(self, dataset_path: pathlib.Path | None = None) -> dict[str, Any]:
        """Compute every registered metric from the recorded HDF5 dataset.

        Args:
            dataset_path: Path to the recorded HDF5 dataset. If ``None``, the path is
                resolved from the environment's recorder manager configuration.

        Returns:
            A dictionary mapping metric name to metric value. Always includes a
            ``"num_episodes"`` entry with the number of completed episodes.
        """
        if dataset_path is None:
            dataset_path = get_metric_recorder_dataset_path(self._env)
        metrics_data: dict[str, Any] = {}
        for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
            recorded_metric_data = get_recorded_metric_data(dataset_path, term_cfg.recorder_term_name)
            metrics_data[term_name] = term_cfg.func(recorded_metric_data, **term_cfg.params)
        metrics_data["num_episodes"] = get_num_episodes(dataset_path)
        return metrics_data
