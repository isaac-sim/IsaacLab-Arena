# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import yaml
from pathlib import Path
from typing import Any

_YAML_INCLUDE_KEY = "external_yaml"


def load_env_graph_spec_dict(path: str | Path) -> dict[str, Any]:
    """Load an env-graph YAML file, resolving its top-level ``external_yaml`` include.

    Only the entry YAML may declare ``external_yaml``; the included file must not
    contain a further include (no nesting).

    Args:
        path: Path to the root YAML file.

    Returns:
        A merged mapping ready for :class:`ArenaEnvGraphSpec` validation.
    """
    path = Path(path).resolve()
    raw = _load_yaml_dict(path)

    include = raw.pop(_YAML_INCLUDE_KEY, None)
    if include is None:
        return raw
    assert isinstance(include, str), f"'{_YAML_INCLUDE_KEY}' must be a file path string, got {type(include).__name__}"

    include_path = (path.parent / include).resolve()
    included = _load_yaml_dict(include_path)
    assert _YAML_INCLUDE_KEY not in included, (
        f"Nested '{_YAML_INCLUDE_KEY}' is not allowed; only the top-level env graph spec YAML "
        f"may declare an include: {include_path}"
    )
    return _merge_env_graph_dicts(included, raw)


def _load_yaml_dict(path: str | Path) -> dict[str, Any]:
    """Read a single YAML file and assert it parses to a mapping."""
    path = Path(path).resolve()
    assert path.is_file(), f"Env graph spec YAML not found: {path}"
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    assert isinstance(raw, dict), f"Env graph spec must be a dict, got {type(raw).__name__}"
    return raw


def _merge_env_graph_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge two env-graph dicts, asserting that no top-level key appears in both."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        assert key not in merged, f"Duplicate env graph spec key across includes: '{key}'"
        merged[key] = copy.deepcopy(value)
    return merged
