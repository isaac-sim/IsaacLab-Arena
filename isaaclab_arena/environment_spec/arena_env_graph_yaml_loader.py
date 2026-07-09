# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import yaml
from pathlib import Path
from typing import Any

_YAML_INCLUDE_KEY = "external_yamls"


def deep_merge_env_graph_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge two env-graph dicts, with ``override`` winning on key conflicts.

    Nested dict values are merged recursively. Lists and scalars are replaced
    entirely when the override defines the key.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == _YAML_INCLUDE_KEY:
            continue
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_env_graph_dicts(merged[key], value)
        else:
            # TODO(qianl): For ``relations``, extend the base list instead of replacing it wholesale.
            merged[key] = copy.deepcopy(value)
    return merged


def load_env_graph_spec_dict(path: str | Path, *, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load an env-graph YAML file, recursively resolving ``external_yamls`` includes.

    Args:
        path: Path to the root YAML file.
        _stack: Active include chain used to detect cycles.

    Returns:
        A merged mapping ready for :class:`ArenaEnvGraphSpec` validation.
    """
    path = Path(path).resolve()
    assert path.is_file(), f"Env graph spec YAML not found: {path}"
    assert path not in _stack, "Cyclic env graph spec external_yamls include: " + " -> ".join(
        str(p) for p in (*_stack, path)
    )

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    assert isinstance(raw, dict), f"Env graph spec must be a dict, got {type(raw).__name__}"

    includes = raw.pop(_YAML_INCLUDE_KEY, None) or []
    assert isinstance(includes, list), f"'{_YAML_INCLUDE_KEY}' must be a list of file paths"
    for include in includes:
        assert isinstance(include, str), f"'{_YAML_INCLUDE_KEY}' entries must be strings, got {type(include).__name__}"

    merged: dict[str, Any] = {}
    for include in includes:
        include_path = (path.parent / include).resolve()
        included = load_env_graph_spec_dict(include_path, _stack=(*_stack, path))
        merged = deep_merge_env_graph_dicts(merged, included)
    return deep_merge_env_graph_dicts(merged, raw)
