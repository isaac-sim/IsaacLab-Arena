# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Terminal service diagnostics recorded through Arena's standard metrics interface."""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg

from .model import ServiceStatus

SERVICE_METRIC_NAMES = (
    "completion_fraction",
    "battery_tests",
    "airflow_tests",
    "unnecessary_replacements",
    "elapsed_seconds",
    "isolation_violation",
)
"""Column names for terminal samples and the aggregate service_diagnostics metric."""


def terminal_service_values(status: ServiceStatus) -> tuple[float, ...]:
    """Return terminal diagnostics in SERVICE_METRIC_NAMES order."""
    completion_conditions = (
        status.battery_verified,
        status.airway_serviced,
        status.vacuum_verified,
        status.originals_preserved,
        status.kit_complete,
        status.station_reset,
    )
    return (
        sum(completion_conditions) / len(completion_conditions),
        float(status.battery_test_count),
        float(status.airflow_test_count),
        float(status.unnecessary_replacements),
        status.elapsed_s,
        float(status.dependency_violation),
    )


class ServiceRecorder(RecorderTerm):
    """Capture each selected environment's terminal status before reset clears it."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.name = cfg.name
        self._first_reset = True

    def record_pre_reset(self, env_ids):
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if self._first_reset:
            assert len(env_ids) == self._env.num_envs, "The initial environment reset must include every environment."
            self._first_reset = False
            return None, None
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        statuses = self._env.task_runtime.statuses
        values = []
        for env_id in env_ids:
            values.append(terminal_service_values(statuses[env_id]))
        samples = torch.tensor(values, dtype=torch.float32, device=self._env.device)
        return self.name, samples.reshape(-1, len(SERVICE_METRIC_NAMES))


@configclass
class ServiceRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = ServiceRecorder
    name: str = "service_diagnostics"


def compute_service_metrics(recorded_metric_data: list[np.ndarray]) -> list[float]:
    """Average terminal samples in SERVICE_METRIC_NAMES order across completed episodes."""
    if not recorded_metric_data:
        return [0.0] * len(SERVICE_METRIC_NAMES)
    rows = np.concatenate(recorded_metric_data, axis=0)
    assert rows.shape == (len(recorded_metric_data), len(SERVICE_METRIC_NAMES)), (
        "Expected one terminal service diagnostic row per episode, "
        f"got {rows.shape} for {len(recorded_metric_data)} episodes."
    )
    assert np.isfinite(rows).all(), "Service diagnostic samples must be finite."
    return np.mean(rows, axis=0).tolist()


class ServiceMetrics(MetricBase):
    """Report terminal completion, test counts, replacement cost, time, and isolation violations."""

    name = "service_diagnostics"
    recorder_term_name = "service_diagnostics"

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return ServiceRecorderCfg(name=self.recorder_term_name)

    def get_metric_term_cfg(self) -> MetricTermCfg:
        return MetricTermCfg(
            compute_metric_func=compute_service_metrics,
            params={},
            recorder_term_name=self.recorder_term_name,
        )
