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

    experiment_cfg = ArenaExperimentCfg(runs={"second": second, "first": first})

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
        ArenaExperimentCfg(runs=runs)
