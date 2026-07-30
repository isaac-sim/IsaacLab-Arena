# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read values out of a typed Arena Experiment before its Runs are composed."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from hydra.core.override_parser.overrides_parser import OverridesParser

from isaaclab_arena.hydra.typed_experiment_loader import load_experiment_run_definitions_from_yaml


def find_environment_field_values(
    yaml_path: str | Path,
    field_name: str,
    overrides: list[str] | None = None,
    default: Any = None,
) -> dict[str, Any]:
    """Return the value each Run gives one environment field, keyed by Run name.

    Composition needs registered environment types, which are only importable once Isaac Sim
    is running. This reads the same YAML and Hydra overrides without importing them.

    Args:
        yaml_path: Path to the Arena Experiment YAML file.
        field_name: Environment field to read from each Run.
        overrides: Hydra field overrides applied to the Runs declared in YAML.
        default: Value returned for Runs that leave the field unset.

    Returns:
        Run names mapped to their value for the field, in YAML declaration order.
    """
    run_values_by_name = load_experiment_run_definitions_from_yaml(yaml_path)
    override_values_by_run_name = _environment_field_values_from_overrides(field_name, overrides or [])

    field_values_by_run_name: dict[str, Any] = {}
    for run_name, run_values in run_values_by_name.items():
        declared_in_yaml = _environment_field_value(run_values, field_name, default)
        field_values_by_run_name[run_name] = override_values_by_run_name.get(run_name, declared_in_yaml)
    return field_values_by_run_name


def typed_experiment_requires_cameras(yaml_path: str | Path, overrides: list[str] | None = None) -> bool:
    """Return whether any Run in a typed YAML Experiment enables environment cameras.

    Args:
        yaml_path: Path to the Arena Experiment YAML file.
        overrides: Hydra field overrides applied to the Runs declared in YAML.

    Returns:
        Whether any declared Run enables environment cameras.
    """
    camera_values_by_run_name = find_environment_field_values(yaml_path, "enable_cameras", overrides, default=False)
    return any(camera_values_by_run_name.values())


def _environment_field_value(run_values: dict[str, Any], field_name: str, default: Any) -> Any:
    """Return the value a Run declares for one environment field in YAML."""
    environment_values = run_values.get("environment")
    if not isinstance(environment_values, dict):
        return default
    return environment_values.get(field_name, default)


def _environment_field_values_from_overrides(field_name: str, overrides: list[str]) -> dict[str, Any]:
    """Return the value overrides assign to one environment field, keyed by Run name.

    Deletions are ignored because they remove a value rather than assign one.
    """
    override_key_pattern = re.compile(rf"runs\.(?P<run_name>[^.]+)\.environment\.{re.escape(field_name)}")
    field_values_by_run_name: dict[str, Any] = {}
    for override in OverridesParser.create().parse_overrides(overrides):
        key_match = override_key_pattern.fullmatch(override.key_or_group)
        if key_match is None or override.is_delete():
            continue
        field_values_by_run_name[key_match.group("run_name")] = override.value()
    return field_values_by_run_name
