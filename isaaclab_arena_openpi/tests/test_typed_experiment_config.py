# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify OpenPI policy composition through typed Arena Experiment YAML."""

from pathlib import Path

import pytest

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation import arena_experiment_config_loader
from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
from isaaclab_arena.hydra.experiment_composition import load_arena_experiment_from_yaml
from isaaclab_arena_openpi.policy.pi0_remote_config import Pi0RemotePolicyCfg


def _write_openpi_experiment(path: Path, *, adapter: str = "droid") -> Path:
    path.write_text(
        f"""
runs:
- name: remote
  environment:
    type: test_environment
  policy:
    type: pi0_remote
    openpi_embodiment_adapter: {adapter}
    remote_host: localhost
    remote_port: 8000
  rollout_limit:
    num_steps: 1
""",
        encoding="utf-8",
    )
    return path


def test_typed_openpi_experiment_composes_runtime_endpoint_overrides(tmp_path, monkeypatch):
    """Register OpenPI and apply the runtime server endpoint through Hydra overrides."""
    experiment_path = _write_openpi_experiment(tmp_path / "openpi.yaml")
    monkeypatch.setattr(
        arena_experiment_config_loader,
        "_registered_environment_cfg_types",
        lambda: {"test_environment": ArenaEnvironmentCfg},
    )

    experiment = load_arena_experiment_from_config_file(
        experiment_path,
        device="cuda:1",
        overrides=[
            "runs.remote.policy.remote_host='{{host:policy_server}}'",
            "runs.remote.policy.remote_port=8123",
        ],
    )

    assert len(experiment) == 1
    assert isinstance(experiment[0].policy, Pi0RemotePolicyCfg)
    assert experiment[0].policy.remote_host == "{{host:policy_server}}"
    assert experiment[0].policy.remote_port == 8123
    assert experiment[0].environment_builder.device == "cuda:1"


def test_typed_openpi_experiment_rejects_unknown_adapter(tmp_path):
    """Reject adapter names after Hydra constructs the typed policy config."""
    experiment_path = _write_openpi_experiment(tmp_path / "openpi.yaml", adapter="unknown")

    with pytest.raises(ValueError, match="Unknown openpi_embodiment_adapter 'unknown'; expected 'droid'"):
        load_arena_experiment_from_yaml(
            experiment_path,
            environment_cfg_types={"test_environment": ArenaEnvironmentCfg},
            policy_cfg_types={"pi0_remote": Pi0RemotePolicyCfg},
        )
