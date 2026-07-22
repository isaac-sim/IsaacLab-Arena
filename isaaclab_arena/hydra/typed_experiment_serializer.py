# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Serialize effective typed Arena Experiments to YAML."""

import yaml
from enum import Enum
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from isaaclab_arena.assets.registries import EnvironmentRegistry, PolicyRegistry
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from isaaclab_arena.evaluation.legacy_graph_environment_cli import LegacyGraphEnvironmentCfg


def serialize_arena_experiment_to_yaml(experiment_cfg: ArenaExperimentCfg) -> str:
    """Serialize an effective Arena Experiment to a typed Experiment YAML-formatted string.

    The output contains the fully composed configuration, including resolved
    defaults and overrides. Source formatting and comments are not preserved.

    Args:
        experiment_cfg: Effective typed Arena Experiment to serialize.

    Returns:
        A YAML-formatted string accepted by the typed Arena Experiment loader.
    """
    assert isinstance(experiment_cfg, ArenaExperimentCfg)
    environment_registry = EnvironmentRegistry()
    policy_registry = PolicyRegistry()
    run_values_by_name = {}
    for run_name, run_cfg in experiment_cfg.runs.items():
        assert isinstance(run_cfg, ArenaRunCfg)
        run_values = OmegaConf.to_container(
            OmegaConf.structured(run_cfg),
            resolve=True,
            enum_to_str=True,
        )
        assert isinstance(run_values, dict)
        assert run_values.pop("name") == run_name

        run_values["environment"] = _environment_yaml_values(environment_registry, run_cfg, run_values["environment"])
        policy_type = policy_registry.get_policy_type_for_cfg(run_cfg.policy)
        policy_selector = policy_type.name
        if not policy_type.__module__.startswith("isaaclab_arena.policy."):
            policy_selector = f"{policy_type.__module__}.{policy_type.__qualname__}"
        run_values["policy"] = {"type": policy_selector, **run_values["policy"]}
        run_values_by_name[run_name] = _to_yaml_values(run_values)
    return yaml.safe_dump({"runs": run_values_by_name}, sort_keys=False)


def _environment_yaml_values(
    environment_registry: EnvironmentRegistry,
    run_cfg: ArenaRunCfg,
    dumped_environment_values: dict[str, Any],
) -> dict[str, Any]:
    """Return one Run's environment section with the type selector the loader expects."""
    if isinstance(run_cfg.environment, LegacyGraphEnvironmentCfg):
        # Graph-YAML environments serialize from their original source values; the derived
        # arena_env_args tokens are an execution detail the loader rebuilds on reload.
        assert (
            run_cfg.environment.env_graph_spec_yaml
        ), "Graph-YAML environment cannot be serialized because it does not record its graph-spec YAML path"
        return {"type": run_cfg.environment.env_graph_spec_yaml, **run_cfg.environment.environment_values}
    environment_type = environment_registry.get_factory_type_for_cfg(run_cfg.environment)
    return {"type": environment_type.name, **dumped_environment_values}


def _to_yaml_values(value: Any) -> Any:
    """Convert structured-config leaf values into safe YAML primitives."""
    if isinstance(value, dict):
        return {key: _to_yaml_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_yaml_values(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value
