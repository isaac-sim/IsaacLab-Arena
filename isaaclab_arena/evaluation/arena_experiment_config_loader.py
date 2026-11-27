# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Load Arena Experiments from YAML configuration files."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from isaaclab_arena.utils.class_path import import_class_from_path

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
    from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
    from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


def load_arena_experiment_from_config_file(
    experiment_config_path: str | Path,
    *,
    device: str,
    overrides: list[str] | None = None,
) -> ArenaExperimentCfg:
    """Load a YAML Arena Experiment and apply its process device.

    Args:
        experiment_config_path: Path to a typed YAML Experiment.
        device: Process-wide simulation device applied to every Run.
        overrides: Hydra overrides applied to the Experiment.

    Returns:
        The loaded, typed Experiment configuration with the process device applied.
    """
    from isaaclab_arena.hydra.experiment_values import read_experiment_values

    return build_arena_experiment(read_experiment_values(experiment_config_path, overrides), device=device)


def build_arena_experiment(experiment_values: dict[str, Any], *, device: str) -> ArenaExperimentCfg:
    """Build the typed Experiment from its values with the registered types, and apply the process device.

    Args:
        experiment_values: The values returned by read_experiment_values.
        device: Process-wide simulation device applied to every Run.

    Returns:
        The typed Experiment configuration with the process device applied.
    """
    from isaaclab_arena.hydra.typed_experiment_loader import build_typed_experiment

    experiment_cfg = build_typed_experiment(
        experiment_values,
        environment_cfg_types=_registered_environment_cfg_types(),
        policy_cfg_type_resolver=_resolve_policy_cfg_type_from_name_or_class_path,
    )

    # TODO(cvolk, 2026-07-09): [typed-config-migration] Make device a process-level
    # evaluation setting shared by AppLauncher and Run execution. Then remove device
    # from ArenaEnvBuilderCfg and delete this per-Run copy.
    runs_with_process_device = {}
    for run_name, run_config in experiment_cfg.runs.items():
        environment_builder_with_process_device = replace(run_config.environment_builder, device=device)
        run_config_with_process_device = replace(
            run_config,
            environment_builder=environment_builder_with_process_device,
        )
        runs_with_process_device[run_name] = run_config_with_process_device
    return replace(experiment_cfg, runs=runs_with_process_device)


def _registered_environment_cfg_types() -> dict[str, type[ArenaEnvironmentCfg]]:
    """Return registered environment selector names and their config types."""
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena_environments.cli import ensure_environments_registered

    ensure_environments_registered()
    return EnvironmentRegistry().get_environment_cfg_types()


def resolve_policy_type_from_name_or_class_path(policy_name_or_class_path: str) -> type[PolicyBase]:
    """Return the policy class for a registered policy name or dotted class path.

    Args:
        policy_name_or_class_path: Registered policy name, or a ``module_path.ClassName`` path.
    """
    from isaaclab_arena.assets.registries import PolicyRegistry

    registry = PolicyRegistry()
    if registry.is_registered(policy_name_or_class_path):
        return registry.get_policy(policy_name_or_class_path)
    assert (
        "." in policy_name_or_class_path
    ), f"Policy type must be a registered name or dotted Python class path, got {policy_name_or_class_path!r}"
    return import_class_from_path(policy_name_or_class_path)


def _resolve_policy_cfg_type_from_name_or_class_path(policy_name_or_class_path: str) -> type[PolicyCfg]:
    """Return the config type for a registered policy name or dotted class path."""
    from isaaclab_arena.assets.registries import PolicyRegistry

    policy_type = resolve_policy_type_from_name_or_class_path(policy_name_or_class_path)
    return PolicyRegistry().get_policy_cfg_type(policy_type)
