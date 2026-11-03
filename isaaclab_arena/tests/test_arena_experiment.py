# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test Arena Experiment declarations and execution ordering."""

from dataclasses import dataclass

import pytest

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from isaaclab_arena.policy.policy_base import PolicyCfg


@dataclass
class _EnvironmentCfg(ArenaEnvironmentCfg):
    pass


@dataclass
class _PolicyCfg(PolicyCfg):
    pass


def _run(name: str) -> ArenaRunCfg:
    return ArenaRunCfg(name=name, environment=_EnvironmentCfg(), policy=_PolicyCfg())


def test_experiment_cfg_preserves_named_run_order():
    second = _run("second")
    first = _run("first")

    experiment_cfg = ArenaExperimentCfg(runs={"second": second, "first": first}, name="experiment")

    assert list(experiment_cfg.runs) == ["second", "first"]
    assert list(experiment_cfg.runs.values()) == [second, first]


@pytest.mark.parametrize(
    ("runs", "error"),
    [
        ({}, "must contain at least one Run"),
        ({"": _run("run")}, "names must be non-empty strings"),
        ({"run": object()}, "must be an ArenaRunCfg"),
        ({"mapping_name": _run("run_name")}, "cannot be overridden"),
    ],
)
def test_experiment_cfg_rejects_invalid_named_runs(runs, error):
    with pytest.raises(AssertionError, match=error):
        ArenaExperimentCfg(runs=runs, name="experiment")


def test_experiment_cfg_exports_timing_kpis_by_default():
    experiment_cfg = ArenaExperimentCfg(runs={"run": _run("run")}, name="experiment")

    assert experiment_cfg.kpi_exporters == ["timing"]


def test_experiment_cfg_default_kpi_exporters_are_not_shared_between_instances():
    first = ArenaExperimentCfg(runs={"run": _run("run")}, name="first")
    second = ArenaExperimentCfg(runs={"run": _run("run")}, name="second")

    first.kpi_exporters.append("success_rate")

    assert second.kpi_exporters == ["timing"]


@pytest.mark.parametrize("kpi_exporters", [[], ["success_rate", "timing"]])
def test_experiment_cfg_accepts_registered_kpi_exporters_or_none(kpi_exporters):
    experiment_cfg = ArenaExperimentCfg(runs={"run": _run("run")}, name="experiment", kpi_exporters=kpi_exporters)

    assert experiment_cfg.kpi_exporters == kpi_exporters


@pytest.mark.parametrize(
    ("kpi_exporters", "error"),
    [
        (["latency"], r"Unknown KPI exporters \['latency'\].*\['success_rate', 'timing'\]"),
        (["timing", "timing"], r"duplicate KPI exporters \['timing'\]"),
        ("timing", "must be a list of names"),
        ([1], "must be a list of names"),
    ],
)
def test_experiment_cfg_rejects_invalid_kpi_exporters(kpi_exporters, error):
    with pytest.raises(AssertionError, match=error):
        ArenaExperimentCfg(runs={"run": _run("run")}, name="experiment", kpi_exporters=kpi_exporters)
