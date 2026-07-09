# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Load Arena Experiments from JSON or YAML configuration files."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from isaaclab_arena.assets.registries import EnvironmentRegistry, PolicyRegistry
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation.arena_experiment import ArenaExperiment
from isaaclab_arena.evaluation.legacy_eval_config import run_cfgs_from_legacy_eval_config
from isaaclab_arena.hydra.arena_experiment import load_arena_experiment_from_yaml
from isaaclab_arena.policy.policy_base import PolicyCfg
from isaaclab_arena_environments.cli import ensure_environments_registered


def load_arena_experiment_from_config_file(
    experiment_config_path: str | Path,
    *,
    device: str,
    overrides: list[str] | None = None,
) -> ArenaExperiment:
    """Load a JSON or YAML Arena Experiment and apply its process device.

    Call this only after SimulationApp starts because resolving the complete
    configuration registries loads Isaac Sim modules.

    Args:
        experiment_config_path: Path to a legacy JSON or typed YAML Experiment.
        device: Process-wide simulation device applied to every Run.
        overrides: Hydra overrides applied to typed YAML Experiments.

    Returns:
        The ordered typed Runs that make up the Experiment.
    """
    path = Path(experiment_config_path)
    assert path.is_file(), f"Experiment config does not exist: '{path}'"

    suffix = path.suffix.lower()
    assert suffix in {".json", ".yaml", ".yml"}, f"Experiment config must use .json, .yaml, or .yml, got '{path}'"

    if suffix == ".json":
        assert not overrides, "Experiment overrides are supported only for typed YAML Experiments"
        with path.open(encoding="utf-8") as experiment_config_file:
            legacy_experiment_config = json.load(experiment_config_file)
        return run_cfgs_from_legacy_eval_config(legacy_experiment_config, device=device)

    yaml_experiment = load_arena_experiment_from_yaml(
        path,
        environment_cfg_types=_registered_environment_cfg_types(),
        policy_cfg_types=_registered_policy_cfg_types(),
        overrides=overrides,
    )

    experiment_with_process_device: ArenaExperiment = []
    for run_config in yaml_experiment:
        environment_builder_with_process_device = replace(run_config.environment_builder, device=device)
        run_config_with_process_device = replace(
            run_config,
            environment_builder=environment_builder_with_process_device,
        )
        experiment_with_process_device.append(run_config_with_process_device)
    return experiment_with_process_device


def _registered_environment_cfg_types() -> dict[str, type[ArenaEnvironmentCfg]]:
    """Return registered environment selector names and their config types."""
    ensure_environments_registered()
    registry = EnvironmentRegistry()
    return {
        name: registry.get_environment_cfg_type(registry.get_component_by_name(name))
        for name in registry.get_all_keys()
    }


def _registered_policy_cfg_types() -> dict[str, type[PolicyCfg]]:
    """Return registered policy selector names and their config types."""
    registry = PolicyRegistry()
    return {
        name: registry.get_policy_cfg_type(registry.get_component_by_name(name)) for name in registry.get_all_keys()
    }
