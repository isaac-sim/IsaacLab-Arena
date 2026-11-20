# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Serialize effective typed Arena Experiments to YAML."""

import yaml
from enum import Enum
from importlib import import_module
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
            policy_selector = _importable_class_path(policy_type)
        run_values["policy"] = {"type": policy_selector, **run_values["policy"]}
        run_values_by_name[run_name] = _to_yaml_values(run_values)
    experiment_values: dict[str, Any] = {
        "name": experiment_cfg.name,
        "kpi_exporters": list(experiment_cfg.kpi_exporters),
        "runs": run_values_by_name,
    }
    return yaml.safe_dump(experiment_values, sort_keys=False)


def _environment_yaml_values(
    environment_registry: EnvironmentRegistry,
    run_cfg: ArenaRunCfg,
    dumped_environment_values: dict[str, Any],
) -> dict[str, Any]:
    """Return one Run's environment section."""
    if isinstance(run_cfg.environment, LegacyGraphEnvironmentCfg):
        return {"type": run_cfg.environment.env_spec_path, **run_cfg.environment.per_run_overrides}
    environment_type = environment_registry.get_factory_type_for_cfg(run_cfg.environment)
    # A process that loads the saved Experiment registers only the environments in isaaclab_arena_environments
    # by name. Other factories are registered when their class path is imported.
    if environment_type.__module__.startswith("isaaclab_arena_environments."):
        environment_selector = environment_type.name
    else:
        environment_selector = _importable_class_path(environment_type)
    return {"type": environment_selector, **dumped_environment_values}


def _importable_class_path(class_type: type) -> str:
    """Return the ``module_path.ClassName`` path that imports class_type when an Experiment selects it.

    A class nested in another class or defined in a function has no such path.
    """
    class_path = f"{class_type.__module__}.{class_type.__name__}"
    module_attribute = getattr(import_module(class_type.__module__), class_type.__name__, None)
    assert module_attribute is class_type, (
        f"{class_type.__module__}.{class_type.__qualname__} cannot be imported as {class_path!r}. Define it at the"
        " top level of its module so a saved Experiment can select it by class path."
    )
    return class_path


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
