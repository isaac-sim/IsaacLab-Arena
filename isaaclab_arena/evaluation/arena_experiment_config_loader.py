# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Load Arena Experiments from JSON or YAML configuration files."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from importlib import import_module
from pathlib import Path
from typing import Any

from isaaclab_arena.assets.registries import EnvironmentRegistry, PolicyRegistry
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from isaaclab_arena.evaluation.legacy_eval_config import run_cfgs_from_legacy_eval_config
from isaaclab_arena.hydra.experiment_composition import compose_arena_run, load_arena_experiment_from_yaml
from isaaclab_arena.policy.policy_base import PolicyCfg

_INLINE_RUN_NAME = "preview"
_INLINE_POLICY_TYPE = "zero_action"


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
        overrides=overrides,
    )

    return _with_process_device(experiment_cfg, device)


def compose_inline_arena_experiment(
    environment_name: str,
    *,
    device: str,
    environment_builder_values: dict[str, Any],
    shared_environment_values: dict[str, Any],
    num_steps: int | None,
    num_episodes: int | None,
    overrides: list[str] | None = None,
) -> ArenaExperimentCfg:
    """Compose one zero-action preview Run from CLI-provided values.

    Args:
        environment_name: Registered environment selected for the preview.
        device: Process-wide simulation device applied to the Run.
        environment_builder_values: Base values for the typed environment builder config.
        shared_environment_values: Shared CLI values copied into matching environment config fields.
        num_steps: Step limit for the preview, or None for an episode-driven preview.
        num_episodes: Episode limit for the preview, or None for a step-driven preview.
        overrides: Hydra overrides relative to the implicit preview Run.

    Returns:
        A one-Run typed Arena Experiment.
    """
    environment_cfg_types = _registered_environment_cfg_types()
    available_environment_names = ", ".join(sorted(environment_cfg_types))
    assert (
        environment_name in environment_cfg_types
    ), f"Unknown registered environment '{environment_name}'. Available environments: {available_environment_names}"

    environment_cfg_type = environment_cfg_types[environment_name]
    environment_field_names = {config_field.name for config_field in fields(environment_cfg_type)}
    if shared_environment_values.get("enable_cameras", False):
        assert (
            "enable_cameras" in environment_field_names
        ), f"Environment '{environment_name}' does not expose camera support in its typed configuration"
    environment_values = {
        field_name: value
        for field_name, value in shared_environment_values.items()
        if field_name in environment_field_names
    }
    environment_values["type"] = environment_name

    run_cfg = compose_arena_run(
        _INLINE_RUN_NAME,
        {
            "environment": environment_values,
            "policy": {"type": _INLINE_POLICY_TYPE},
            "environment_builder": environment_builder_values,
            "rollout_limit": {"num_steps": num_steps, "num_episodes": num_episodes},
        },
        environment_cfg_types=environment_cfg_types,
        policy_cfg_type_resolver=_resolve_policy_cfg_type_from_name_or_class_path,
        overrides=overrides,
        source=f"inline environment '{environment_name}'",
    )
    return _with_process_device(ArenaExperimentCfg(runs={_INLINE_RUN_NAME: run_cfg}), device)


def _with_process_device(experiment_cfg: ArenaExperimentCfg, device: str) -> ArenaExperimentCfg:
    """Apply one process device to every Run in an Experiment."""
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


def _registered_environment_cfg_types() -> dict[str, type[ArenaEnvironmentCfg]]:
    """Return registered environment selector names and their config types."""
    from isaaclab_arena_environments.cli import ensure_environments_registered

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
