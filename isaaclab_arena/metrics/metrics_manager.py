# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_arena.metrics.metric_data import MetricData, MetricsDataCollection
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg
from isaaclab_arena.metrics.metrics import get_metric_recorder_dataset_path, get_num_episodes, get_recorded_metric_data

if TYPE_CHECKING:
    from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv


class MetricsManager:
    """Run-time manager that computes metrics.

    The manager parses a configclass container of `MetricTermCfg` instances --
    one field per metric -- and exposes a method `compute` to load the recorded data for
    each term and compute the metric.
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
        for term_name, term_cfg in cfg.__dict__.items():
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)

    @property
    def active_terms(self) -> list[str]:
        """Names of the metric terms registered on this manager."""
        return list(self._term_names)

    def compute(self) -> MetricsDataCollection:
        """Compute every registered metric from the recorded HDF5 dataset.

        Returns:
            A MetricsData instance containing the data for all metrics.
        """
        dataset_path = get_metric_recorder_dataset_path(self._env)
        metric_data_entries: dict[str, MetricData] = {}
        for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
            recorded_data = get_recorded_metric_data(dataset_path, term_cfg.recorder_term_name)
            metrics_value = term_cfg.compute_metric_func(recorded_data, **term_cfg.params)
            metric_data_entries[term_name] = MetricData(
                term_name=term_name, term_cfg=term_cfg, recorded_data=recorded_data, metric_value=metrics_value
            )
        # Combine the metric data into a MetricsData instance.
        metrics_data_collection = MetricsDataCollection(
            num_episodes=get_num_episodes(dataset_path), metric_data_entries=metric_data_entries
        )
        return metrics_data_collection
