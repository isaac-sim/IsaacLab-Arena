# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read the values of an Arena Experiment from its YAML file and overrides, and merge shared values into each Run.

The values are plain dicts and lists with interpolations unresolved. Reading them needs no Isaac Sim and no registries.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Any

from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from isaaclab_arena.variations.variations_hydra import overrides_from_dict
from isaaclab_arena.video.video_recording import RUN_DIR_TIMESTAMP_FORMAT

DEFAULT_EXPERIMENT_NAME_FORMAT = f"{RUN_DIR_TIMESTAMP_FORMAT}_experiment"
"""strftime format of the Experiment name when neither a file nor an override sets one, for example
``2026-11-27_14-42-54_experiment``. The timestamp keeps the KPI records of separate runs apart."""

RUN_FIELDS_REPLACING_SHARED = frozenset({"rollout_limit"})
"""Run fields that replace the shared value as a whole when a Run declares them, instead of merging into it.
A rollout limit is one choice among exclusive fields, so merging two limits of different kinds would combine both."""

RUN_SECTIONS_SELECTED_BY_TYPE = ("environment", "policy")
"""Run sections whose ``type`` selects a typed config. A Run only inherits the shared section of the same type."""

_EXPERIMENT_FIELDS = frozenset({"name", "kpi_exporters", "runs", "shared"})
"""Top-level fields of an Experiment YAML file."""


def read_experiment_values(
    yaml_path: str | Path | None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Read the Experiment values from a YAML file and overrides, and merge the ``shared`` values into each Run.

    With a file, overrides edit its values as written, like an edit of the file, but cannot add Runs. To switch the
    ``type`` of an ``environment`` or ``policy`` section that writes fields of the old type, assign the whole
    section, for example ``runs.<name>.policy={type:replay,replay_file_path:demo.hdf5}``. Without a file, the
    overrides are the whole Experiment. The result is validated, every Run selects an environment type and a policy
    type, and it has a ``name``: the declared one, else the file stem, else the current time in
    ``DEFAULT_EXPERIMENT_NAME_FORMAT``.

    Args:
        yaml_path: Path to the Arena Experiment YAML file, or None to build the Experiment from overrides alone.
        overrides: Override strings in Hydra syntax, rooted at the Experiment.

    Returns:
        The Experiment values: ``name``, ``kpi_exporters`` when set, and ``runs`` with the ``shared`` values merged
        into each Run. Interpolations stay unresolved.
    """
    file_values = _read_yaml_mapping(yaml_path) if yaml_path is not None else {}
    try:
        experiment_values = _apply_experiment_overrides(file_values, overrides or [])
    except (OmegaConfBaseException, TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not apply overrides to Arena Experiment '{_experiment_source_for_messages(yaml_path)}': {exc}"
        ) from exc

    unknown_fields = sorted(set(experiment_values) - _EXPERIMENT_FIELDS)
    assert not unknown_fields, f"Unknown Experiment fields: {', '.join(unknown_fields)}"
    assert isinstance(experiment_values.get("shared", {}), dict), "Experiment 'shared' must be a mapping"
    assert isinstance(
        experiment_values.get("kpi_exporters", []), list
    ), f"Experiment 'kpi_exporters' must be a list of names, got {experiment_values['kpi_exporters']!r}"
    runs = experiment_values.get("runs")
    if yaml_path is None:
        assert runs, (
            "Without an Experiment file, the overrides must define at least one Run, for example "
            "runs.<name>.environment.type=<type>"
        )
    else:
        assert runs, f"The Experiment must define at least one Run under 'runs' in '{yaml_path}'"
    assert isinstance(runs, dict), "Experiment 'runs' must be a mapping from Run names to Run configurations"
    if yaml_path is not None:
        file_runs = file_values.get("runs")
        declared_run_names = set(file_runs) if isinstance(file_runs, dict) else set()
        undeclared_run_names = sorted(set(runs) - declared_run_names)
        assert not undeclared_run_names, (
            f"Runs not declared in the Experiment file: {', '.join(undeclared_run_names)}. "
            f"Overrides can only change the declared Runs: {', '.join(sorted(declared_run_names)) or '(none)'}"
        )

    experiment_values["name"] = _experiment_name(experiment_values, yaml_path)
    shared_values = experiment_values.pop("shared", None) or {}
    try:
        experiment_values["runs"] = _merge_shared_into_runs(shared_values, runs)
    except ValueError as exc:
        raise ValueError(
            f"Could not read Arena Experiment '{_experiment_source_for_messages(yaml_path)}': {exc}"
        ) from exc
    for run_name, run_values in experiment_values["runs"].items():
        _assert_run_selects_types(run_name, run_values, has_experiment_file=yaml_path is not None)
    return experiment_values


def _apply_experiment_overrides(experiment_values: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply overrides to Experiment values, as if editing the YAML file.

    Overrides use Hydra syntax. ``path=value``, ``+path=value`` and ``++path=value`` all set the value at ``path``
    and create missing keys. ``~path`` deletes the key and takes no value. A mapping value replaces the old mapping.
    Later overrides win. A comma-separated value such as ``path=1,2`` is rejected, because Hydra reads it as a sweep,
    and so are a package (``path@package=value``) and ``extend_list(...)``, which have no meaning here.
    Interpolations such as ``${environment_builder.num_envs}`` are kept as strings.

    Args:
        experiment_values: Experiment values read from the YAML file, or empty without a file. They are not
            modified.
        overrides: Override strings in Hydra syntax.

    Returns:
        New Experiment values with the overrides applied.
    """
    config = OmegaConf.create(experiment_values)
    for override in OverridesParser.create().parse_overrides(overrides):
        key_path = override.key_or_group
        assert (
            not override.is_sweep_override()
        ), f"Ambiguous value for '{key_path}'. Use {key_path}=[value1,value2] for a list, or quote the value."
        assert override.package is None, (
            f"Override '{override.input_line}' names a package with @, which Experiment overrides do not support. "
            f"Remove '@{override.package}' from the override."
        )
        assert not override.is_list_extend(), (
            f"Override '{override.input_line}' uses extend_list, which Experiment overrides do not support. "
            f"Assign the whole list: {key_path}=[...]"
        )
        if override.is_delete():
            # Hydra's ~path=value deletes the key only when it holds that value; Experiment overrides do not support it.
            assert (
                override.value_type is None
            ), f"Override '{override.input_line}' deletes a key, so it cannot take a value. Use ~{key_path}"
            _delete_key(config, key_path)
        else:
            OmegaConf.update(config, key_path, override.value(), merge=False)
    return OmegaConf.to_container(config, resolve=False)


def _merge_shared_into_runs(
    shared_values: dict[str, Any], run_values_by_name: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Return the Run values by name, with the ``shared`` values merged under each Run.

    A Run field listed in ``RUN_FIELDS_REPLACING_SHARED`` replaces the shared value instead of merging into it.
    A Run whose ``environment.type`` or ``policy.type`` differs from the shared one does not inherit that shared
    section, because the shared fields belong to another type. Every merged Run's ``variations`` must set each
    variation path once. Interpolations remain unresolved.

    Args:
        shared_values: Values under ``shared``, inherited by every Run.
        run_values_by_name: Values under ``runs``, keyed by Run name.

    Returns:
        Run names mapped to their merged values, in declaration order.
    """
    merged_run_values_by_name: dict[str, dict[str, Any]] = {}
    for run_name, run_values in run_values_by_name.items():
        assert isinstance(run_name, str) and run_name, "Experiment Run names must be non-empty strings"
        assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", run_name), (
            f"Experiment Run name '{run_name}' must start with a letter or underscore and contain only letters, "
            "numbers, underscores, or hyphens"
        )
        assert isinstance(run_values, dict), f"Run '{run_name}' must be a mapping"

        fields_not_inherited = set(RUN_FIELDS_REPLACING_SHARED & run_values.keys())
        for section_name in RUN_SECTIONS_SELECTED_BY_TYPE:
            if _selects_a_different_type(shared_values.get(section_name), run_values.get(section_name)):
                fields_not_inherited.add(section_name)
        inherited_values = {key: value for key, value in shared_values.items() if key not in fields_not_inherited}
        try:
            merged_values = OmegaConf.merge(inherited_values, run_values)
        except (OmegaConfBaseException, TypeError, ValueError) as exc:
            raise ValueError(f"Could not merge shared values into Run '{run_name}': {exc}") from exc
        merged_run_values = OmegaConf.to_container(merged_values, resolve=False)
        assert (
            "name" not in merged_run_values
        ), f"Run '{run_name}' must not define 'name'; its mapping key is the Run name"
        _assert_run_sets_each_variation_path_once(run_name, merged_run_values)
        merged_run_values_by_name[run_name] = merged_run_values
    return merged_run_values_by_name


def _assert_run_selects_types(run_name: str, run_values: dict[str, Any], *, has_experiment_file: bool) -> None:
    """Check that a Run's merged values select an environment type and a policy type.

    Without an Experiment file, the error for a missing type adds that the overrides are the whole Experiment.
    """
    for section_name in RUN_SECTIONS_SELECTED_BY_TYPE:
        section_values = run_values.get(section_name, {})
        assert isinstance(section_values, dict), f"Run '{run_name}': '{section_name}' must be a mapping"
        section_type = section_values.get("type")
        missing_type_message = (
            f"Run '{run_name}': missing '{section_name}.type'. "
            f"Set it, for example with runs.{run_name}.{section_name}.type=<type>"
        )
        if not has_experiment_file:
            missing_type_message += (
                ". Without an Experiment file, the overrides are the whole Experiment. To change a Run of an"
                " Experiment file, pass the file."
            )
        assert section_type, missing_type_message
        assert isinstance(
            section_type, str
        ), f"Run '{run_name}': '{section_name}.type' must be a string, got {section_type!r}"


def _assert_run_sets_each_variation_path_once(run_name: str, run_values: dict[str, Any]) -> None:
    """Check that a Run's merged ``variations`` give each variation path one value."""
    variations = run_values.get("variations", {})
    assert isinstance(variations, dict), f"Run '{run_name}': 'variations' must be a mapping"
    # The environment build turns the variations into one Hydra override per path, where the last one silently wins.
    variation_paths = set()
    for variation_override in overrides_from_dict(variations):
        variation_path = variation_override.partition("=")[0]
        assert variation_path not in variation_paths, (
            f"Run '{run_name}': variation '{variation_path}' is written both as a dotted key and nested. Write it one"
            " way; an override can only change an entry that the file writes nested."
        )
        variation_paths.add(variation_path)


def _experiment_source_for_messages(yaml_path: str | Path | None) -> str:
    """Return where an Experiment comes from, as named in error messages: its YAML path or the command line."""
    return "command line" if yaml_path is None else str(yaml_path)


def _read_yaml_mapping(yaml_path: str | Path) -> dict[str, Any]:
    """Return the top-level mapping of an Experiment YAML file, with interpolations unresolved."""
    path = Path(yaml_path)
    assert path.suffix.lower() in {".yaml", ".yml"}, (
        f"Experiment config must be a YAML file (.yaml or .yml), got '{path}'. "
        "JSON Experiment configs are no longer supported; write their jobs as Runs in a YAML Experiment."
    )
    assert path.is_file(), f"Experiment config does not exist: '{path}'"
    try:
        raw_experiment_config = OmegaConf.load(path)
    except OmegaConfBaseException as exc:
        raise ValueError(f"Could not load Experiment YAML '{path}': {exc}") from exc
    assert OmegaConf.is_dict(raw_experiment_config), "Experiment config must be a mapping"
    return OmegaConf.to_container(raw_experiment_config, resolve=False)


def _experiment_name(experiment_values: dict[str, Any], yaml_path: str | Path | None) -> str:
    """Return the declared Experiment name, else the file stem, else a name in ``DEFAULT_EXPERIMENT_NAME_FORMAT``."""
    if "name" not in experiment_values:
        if yaml_path is not None:
            return Path(yaml_path).stem
        return datetime.datetime.now().strftime(DEFAULT_EXPERIMENT_NAME_FORMAT)
    experiment_name = experiment_values["name"]
    # YAML and Hydra read a name such as 2026 as a number. bool is an int subclass, but true is not a name.
    if isinstance(experiment_name, (int, float)) and not isinstance(experiment_name, bool):
        experiment_name = str(experiment_name)
    assert isinstance(experiment_name, str), f"Experiment 'name' must be a string, got {experiment_name!r}"
    return experiment_name


def _selects_a_different_type(shared_section: Any, run_section: Any) -> bool:
    """Return whether both sections set a ``type`` and the types differ."""
    if not isinstance(shared_section, dict) or not isinstance(run_section, dict):
        return False
    shared_type = shared_section.get("type")
    run_type = run_section.get("type")
    return shared_type is not None and run_type is not None and shared_type != run_type


def _delete_key(config: Any, key_path: str) -> None:
    """Delete the key at the dotted ``key_path``, asserting that it exists."""
    parent_path, _, key = key_path.rpartition(".")
    parent = OmegaConf.select(config, parent_path) if parent_path else config
    missing_key_message = f"Cannot delete '{key_path}': the key does not exist."
    assert OmegaConf.is_dict(parent), missing_key_message
    existing_keys = ", ".join(sorted(str(existing_key) for existing_key in parent)) or "(none)"
    parent_description = f"'{parent_path}'" if parent_path else "the Experiment"
    assert key in parent, f"{missing_key_message} Keys under {parent_description}: {existing_keys}"
    del parent[key]
