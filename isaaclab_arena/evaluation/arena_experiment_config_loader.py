# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Load Arena Experiments from JSON or YAML configuration files."""

from __future__ import annotations

import json
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any

from isaaclab_arena.assets.registries import EnvironmentRegistry, PolicyRegistry
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from isaaclab_arena.evaluation.legacy_environment_cli_args import legacy_environment_args_to_cli_args
from isaaclab_arena.evaluation.legacy_eval_config import run_cfgs_from_legacy_eval_config
from isaaclab_arena.evaluation.legacy_graph_environment_cli import LegacyGraphEnvironmentCfg
from isaaclab_arena.hydra.typed_experiment_loader import load_arena_experiment_from_yaml
from isaaclab_arena.policy.policy_base import PolicyCfg
from isaaclab_arena_environments.cli import ensure_environments_registered


def validate_experiment_config_path(experiment_config: str | Path) -> Path:
    """Return an existing Experiment configuration path with a supported suffix."""
    experiment_config_path = Path(experiment_config)
    assert experiment_config_path.is_file(), f"Experiment config does not exist: '{experiment_config_path}'"
    assert experiment_config_path.suffix.lower() in {
        ".json",
        ".yaml",
        ".yml",
    }, f"Experiment config must use .json, .yaml, or .yml, got '{experiment_config_path}'"
    return experiment_config_path


def load_arena_experiment_from_config_file(
    experiment_config_path: str | Path,
    *,
    device: str,
    overrides: list[str] | None = None,
) -> ArenaExperimentCfg:
    """Load a JSON or YAML Arena Experiment and apply its process device.

    Args:
        experiment_config_path: Path to a legacy JSON or typed YAML Experiment.
        device: Process-wide simulation device applied to every Run.
        overrides: Hydra overrides applied to typed YAML Experiments.

    Returns:
        The loaded, typed Experiment configuration with the process device applied.
    """
    path = validate_experiment_config_path(experiment_config_path)

    if path.suffix.lower() == ".json":
        assert not overrides, "Experiment overrides are supported only for typed YAML Experiments"
        with path.open(encoding="utf-8") as experiment_config_file:
            legacy_experiment_config = json.load(experiment_config_file)
        run_cfgs = run_cfgs_from_legacy_eval_config(legacy_experiment_config, device=device)
        return ArenaExperimentCfg(runs={run_cfg.name: run_cfg for run_cfg in run_cfgs})

    experiment_cfg = load_arena_experiment_from_yaml(
        path,
        environment_cfg_types=_registered_environment_cfg_types(),
        policy_cfg_type_resolver=_resolve_policy_cfg_type_from_name_or_class_path,
        graph_environment_cfg_factory=_graph_environment_cfg_from_yaml_values,
        overrides=overrides,
    )

    # TODO(cvolk, 2026-07-09): [typed-config-migration] Make device a process-level
    # evaluation setting shared by AppLauncher and Run execution. Then remove device
    # from ArenaEnvBuilderCfg and delete this per-Run copy.
    runs_with_process_device: dict[str, ArenaRunCfg] = {}
    for run_name, run_config in experiment_cfg.runs.items():
        environment_builder_with_process_device = replace(run_config.environment_builder, device=device)
        run_config_with_process_device = replace(
            run_config,
            environment_builder=environment_builder_with_process_device,
        )
        runs_with_process_device[run_name] = run_config_with_process_device
    return ArenaExperimentCfg(runs=runs_with_process_device)


# TODO(cvolk, 2026-07-07): [typed-config-migration] Delete this factory when graph-YAML
# environments have a typed configuration and no longer use the argparse compatibility path.
def _graph_environment_cfg_from_yaml_values(
    env_graph_spec_yaml: str,
    environment_values: dict[str, Any],
    environment_builder_values: dict[str, Any],
) -> LegacyGraphEnvironmentCfg:
    """Create the temporary graph-YAML compatibility config from typed YAML Run values.

    The environment and environment_builder values are rendered as CLI tokens for the
    existing graph-environment argparse path, mirroring the legacy JSON frontend. The
    environment_builder section additionally composes into the Run's typed builder
    config as usual.
    """
    arena_env_args: dict[str, Any] = {
        "environment": env_graph_spec_yaml,
        **environment_builder_values,
        **environment_values,
    }
    return LegacyGraphEnvironmentCfg(arena_env_args=legacy_environment_args_to_cli_args(arena_env_args))


def _registered_environment_cfg_types() -> dict[str, type[ArenaEnvironmentCfg]]:
    """Return registered environment selector names and their config types."""
    ensure_environments_registered()
    registry = EnvironmentRegistry()
    environment_cfg_types: dict[str, type[ArenaEnvironmentCfg]] = {}
    for name in registry.get_all_keys():
        environment_factory_type = registry.get_component_by_name(name)
        environment_cfg_types[name] = registry.get_environment_cfg_type(environment_factory_type)
    return environment_cfg_types


def _resolve_policy_cfg_type_from_name_or_class_path(policy_name_or_class_path: str) -> type[PolicyCfg]:
    """Return the config type for a registered policy name or dotted class path."""
    registry = PolicyRegistry()
    if registry.is_registered(policy_name_or_class_path):
        policy_type = registry.get_policy(policy_name_or_class_path)
    else:
        assert (
            "." in policy_name_or_class_path
        ), f"Policy type must be a registered name or dotted Python class path, got {policy_name_or_class_path!r}"
        module_path, class_name = policy_name_or_class_path.rsplit(".", 1)
        policy_type = getattr(import_module(module_path), class_name)
    return registry.get_policy_cfg_type(policy_type)
