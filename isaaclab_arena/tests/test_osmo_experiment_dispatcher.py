# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify Arena Experiment discovery and dispatch to OSMO workflows."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from isaaclab_arena.evaluation.experiment_runner_cfg import ExperimentRunnerCfg
from osmo import experiment_dispatcher


def _write_experiment(path, policy_values: str) -> None:
    path.write_text(
        f"""runs:
- name: run
  environment:
    type: test
  policy:
{policy_values}
""",
        encoding="utf-8",
    )


def test_discovers_unresolved_openpi_policy_values(tmp_path):
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        """runs:
- name: local
  environment:
    type: test
  policy:
    type: zero_action
- name: remote
  environment:
    type: test
  policy:
    type: pi0_remote
    policy_variant: pi05
""",
        encoding="utf-8",
    )

    assert experiment_dispatcher._get_openpi_run_policy_values(str(experiment_path)) == {
        "remote": {"type": "pi0_remote", "policy_variant": "pi05"}
    }


def test_dispatches_non_openpi_experiment_to_base_workflow(tmp_path, monkeypatch):
    experiment_path = tmp_path / "experiment.yaml"
    _write_experiment(experiment_path, "    type: zero_action")
    experiment_runner_cfg = ExperimentRunnerCfg(experiment_config=str(experiment_path))
    osmo_cfg = SimpleNamespace(openpi_server=SimpleNamespace(policy_variant="pi05"))
    received = None

    class Workflow:
        def __init__(self, *, cfg, experiment_runner_cfg):
            nonlocal received
            received = (cfg, experiment_runner_cfg)

        def submit_workflow(self):
            return 23

    monkeypatch.setattr(experiment_dispatcher, "load_arena_experiment_workflow_cfg", lambda _path: osmo_cfg)
    monkeypatch.setattr(experiment_dispatcher, "ArenaExperimentWorkflow", Workflow)
    monkeypatch.setattr(
        experiment_dispatcher,
        "OpenPiArenaExperimentWorkflow",
        lambda **_kwargs: pytest.fail("non-OpenPI Experiment must not start an OpenPI server"),
    )

    assert experiment_dispatcher.submit_experiment_to_osmo(experiment_runner_cfg, "osmo.yaml") == 23
    assert received == (osmo_cfg, experiment_runner_cfg)


def test_dispatches_all_openpi_runs_to_one_server_workflow(tmp_path, monkeypatch):
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        """runs:
- name: first
  environment:
    type: test
  policy:
    type: pi0_remote
- name: second
  environment:
    type: test
  policy:
    type: pi0_remote
    policy_variant: pi05
""",
        encoding="utf-8",
    )
    experiment_runner_cfg = ExperimentRunnerCfg(experiment_config=str(experiment_path))
    osmo_cfg = SimpleNamespace(openpi_server=SimpleNamespace(policy_variant="pi05"))
    received = None

    class Workflow:
        def __init__(self, *, cfg, experiment_runner_cfg, openpi_run_names):
            nonlocal received
            received = (cfg, experiment_runner_cfg, openpi_run_names)

        def submit_workflow(self):
            return 29

    monkeypatch.setattr(experiment_dispatcher, "load_arena_experiment_workflow_cfg", lambda _path: osmo_cfg)
    monkeypatch.setattr(
        experiment_dispatcher,
        "ArenaExperimentWorkflow",
        lambda **_kwargs: pytest.fail("OpenPI Experiment must use its server workflow"),
    )
    monkeypatch.setattr(experiment_dispatcher, "OpenPiArenaExperimentWorkflow", Workflow)

    assert experiment_dispatcher.submit_experiment_to_osmo(experiment_runner_cfg) == 29
    assert received == (osmo_cfg, experiment_runner_cfg, ["first", "second"])


def test_rejects_openpi_variant_mismatch(tmp_path, monkeypatch):
    experiment_path = tmp_path / "experiment.yaml"
    _write_experiment(
        experiment_path,
        "    type: pi0_remote\n    policy_variant: pi0",
    )
    cfg = ExperimentRunnerCfg(experiment_config=str(experiment_path))
    osmo_cfg = SimpleNamespace(openpi_server=SimpleNamespace(policy_variant="pi05"))
    monkeypatch.setattr(experiment_dispatcher, "load_arena_experiment_workflow_cfg", lambda _path: osmo_cfg)

    with pytest.raises(AssertionError, match="one shared OpenPI server variant"):
        experiment_dispatcher.submit_experiment_to_osmo(cfg)


def test_applies_openpi_variant_override_before_validation(tmp_path, monkeypatch):
    experiment_path = tmp_path / "experiment.yaml"
    _write_experiment(experiment_path, "    type: pi0_remote")
    cfg = ExperimentRunnerCfg(
        experiment_config=str(experiment_path),
        experiment_overrides=["runs.run.policy.policy_variant=pi0"],
    )
    osmo_cfg = SimpleNamespace(openpi_server=SimpleNamespace(policy_variant="pi05"))
    monkeypatch.setattr(experiment_dispatcher, "load_arena_experiment_workflow_cfg", lambda _path: osmo_cfg)

    with pytest.raises(AssertionError, match="one shared OpenPI server variant"):
        experiment_dispatcher.submit_experiment_to_osmo(cfg)
