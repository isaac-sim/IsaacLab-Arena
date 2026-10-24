# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Load and validate the list of tracked benchmarks."""

from __future__ import annotations

import re
import yaml
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
"""Repository root; ``experiment_cfg`` paths are relative to it."""

_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_TOP_LEVEL_KEYS = {"benchmarks"}
_ENTRY_KEYS = {"name", "experiment_cfg", "overrides"}
_RESERVED_OVERRIDE_KEYS = ("osmo.workflow_name", "osmo.output_url")


@dataclass
class BenchmarkEntry:
    """One tracked benchmark: an Experiment YAML plus optional Hydra overrides."""

    name: str
    """Stable identifier used in OSMO workflow names and folder paths."""

    experiment_cfg: str
    """Repo-root relative path to an existing Experiment YAML."""

    overrides: list[str]
    """Hydra overrides applied on top of the Experiment YAML."""


def load_benchmark_list(path: Path) -> list[BenchmarkEntry]:
    """Load and validate a benchmark list YAML file.

    Args:
        path: YAML file of the form ``benchmarks: [{name, experiment_cfg, overrides?}]``.

    Returns:
        The validated entries in file order.
    """
    content = yaml.safe_load(Path(path).read_text())
    assert isinstance(content, dict), f"{path}: expected a mapping with a 'benchmarks' key"
    unknown_keys = set(content) - _TOP_LEVEL_KEYS
    assert not unknown_keys, f"{path}: unknown top-level keys {sorted(unknown_keys)}"
    raw_entries = content.get("benchmarks")
    assert isinstance(raw_entries, list) and raw_entries, f"{path}: 'benchmarks' must be a non-empty list"

    entries = []
    seen_names = set()
    for raw_entry in raw_entries:
        entry = _parse_entry(raw_entry, path)
        assert entry.name not in seen_names, f"{path}: duplicate benchmark name '{entry.name}'"
        seen_names.add(entry.name)
        entries.append(entry)
    return entries


def _parse_entry(raw_entry: object, path: Path) -> BenchmarkEntry:
    """Validate one raw list entry and convert it to a ``BenchmarkEntry``."""
    assert isinstance(raw_entry, dict), f"{path}: each benchmark must be a mapping, got {raw_entry!r}"
    name = raw_entry.get("name")
    assert isinstance(name, str) and _NAME_PATTERN.match(
        name
    ), f"{path}: benchmark name {name!r} must match {_NAME_PATTERN.pattern}"
    unknown_keys = set(raw_entry) - _ENTRY_KEYS
    assert not unknown_keys, f"{path}: benchmark '{name}' has unknown keys {sorted(unknown_keys)}"

    experiment_cfg = raw_entry.get("experiment_cfg")
    assert isinstance(experiment_cfg, str), f"{path}: benchmark '{name}' needs a string 'experiment_cfg'"
    assert not Path(
        experiment_cfg
    ).is_absolute(), f"{path}: benchmark '{name}' experiment_cfg '{experiment_cfg}' must be relative to the repo root"
    experiment_cfg_resolved = (REPO_ROOT / experiment_cfg).resolve()
    assert experiment_cfg_resolved.is_relative_to(
        REPO_ROOT
    ), f"{path}: benchmark '{name}' experiment_cfg '{experiment_cfg}' is outside the repo root {REPO_ROOT}"
    assert (
        experiment_cfg_resolved.is_file()
    ), f"{path}: benchmark '{name}' experiment_cfg '{experiment_cfg}' does not exist relative to {REPO_ROOT}"

    overrides = raw_entry.get("overrides", [])
    assert isinstance(overrides, list) and all(
        isinstance(override, str) for override in overrides
    ), f"{path}: benchmark '{name}' overrides must be a list of strings"
    for override in overrides:
        override_key = override.lstrip("+~")
        for reserved_key in _RESERVED_OVERRIDE_KEYS:
            assert override_key != reserved_key and not override_key.startswith(
                reserved_key + "="
            ), f"{path}: benchmark '{name}' override '{override}' sets reserved key '{reserved_key}'"
    return BenchmarkEntry(name=name, experiment_cfg=experiment_cfg, overrides=list(overrides))
