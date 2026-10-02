# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify service diagnostics at Arena's terminal recording boundary."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_service_metric_records_terminal_selection_and_averages_episodes(simulation_app):
    import numpy as np
    import torch
    from types import SimpleNamespace

    from isaaclab_arena_environments.return_to_service.metrics import (
        SERVICE_METRIC_NAMES,
        ServiceRecorder,
        ServiceRecorderCfg,
        compute_service_metrics,
    )
    from isaaclab_arena_environments.return_to_service.model import ServiceStatus

    completed = ServiceStatus(
        success=True,
        battery_verified=True,
        airway_serviced=True,
        vacuum_verified=True,
        originals_preserved=True,
        kit_complete=True,
        station_reset=True,
        battery_test_count=2,
        airflow_test_count=4,
        elapsed_s=120.0,
    )
    failed = ServiceStatus(
        battery_verified=True,
        airway_serviced=True,
        originals_preserved=True,
        battery_test_count=1,
        airflow_test_count=1,
        unnecessary_replacements=2,
        dependency_violation=True,
        elapsed_s=240.0,
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        return_to_service=SimpleNamespace(statuses=[completed, failed]),
    )
    recorder = ServiceRecorder(ServiceRecorderCfg(), env)
    assert recorder.record_pre_reset(None) == (None, None), "Initial reset must not emit a fabricated episode."
    name, sample = recorder.record_pre_reset(torch.tensor([1]))
    assert name == "service_diagnostics"
    assert sample.tolist() == [[0.5, 1.0, 1.0, 2.0, 240.0, 1.0]]
    assert env.return_to_service.statuses[0] is completed, "Reading one terminal state must not reset any model."
    _, sample = recorder.record_pre_reset([0, 1])
    episode_rows = [row[None, :].numpy() for row in sample]
    assert compute_service_metrics(episode_rows) == [0.75, 1.5, 2.5, 1.0, 180.0, 0.5]
    assert compute_service_metrics([]) == [0.0] * len(SERVICE_METRIC_NAMES)
    assert np.isfinite(compute_service_metrics(episode_rows)).all()
    return True


def test_service_metric_records_terminal_selection_and_averages_episodes():
    assert run_function_with_persistent_simulation_app(
        _test_service_metric_records_terminal_selection_and_averages_episodes
    )
