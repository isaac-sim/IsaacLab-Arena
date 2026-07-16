# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Format composed dataclass fields as Hydra override help."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Collection, Iterator, Mapping
from dataclasses import Field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

_OVERRIDE_HELP_METADATA_KEY = "override_help"
_HYDRA_PATH_SEGMENT_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
_HYDRA_MAPPING_KEY_ESCAPE_CHARACTERS = frozenset("\\()[]{}:=, \t")


def format_config_override_help(
    cfg: object,
    *,
    prefix: str = "",
    excluded_paths: Collection[str] = (),
) -> str:
    """Format the configurable leaves of a composed dataclass as Hydra overrides.

    Dataclass fields with ``metadata={"override_help": False}`` are omitted.

    Args:
        cfg: Composed dataclass instance whose fields can be overridden.
        prefix: Optional Hydra path prepended to every field.
        excluded_paths: Fully qualified fields managed by the caller instead of Hydra.

    Returns:
        One ``path=value`` override per line.
    """
    assert is_dataclass(cfg) and not isinstance(cfg, type), "Override help requires a dataclass instance"
    _assert_hydra_path(prefix)
    excluded_paths = frozenset(excluded_paths)
    return "\n".join(
        f"{path}={_format_override_value(value, path)}"
        for path, value in _iter_override_leaves(cfg, prefix)
        if path not in excluded_paths
    )


def print_config_override_help(
    cfg: object,
    *,
    prefix: str = "",
    excluded_paths: Collection[str] = (),
    file: TextIO | None = None,
) -> None:
    """Print the configurable leaves of a composed dataclass as Hydra overrides.

    Args:
        cfg: Composed dataclass instance whose fields can be overridden.
        prefix: Optional Hydra path prepended to every field.
        excluded_paths: Fully qualified fields managed by the caller instead of Hydra.
        file: Text stream receiving the formatted overrides; defaults to stdout.
    """
    output_stream = sys.stdout if file is None else file
    print(
        format_config_override_help(cfg, prefix=prefix, excluded_paths=excluded_paths),
        file=output_stream,
    )


def _iter_override_leaves(value: Any, path: str) -> Iterator[tuple[str, Any]]:
    if is_dataclass(value) and not isinstance(value, type):
        for config_field in fields(value):
            if _include_in_override_help(config_field):
                yield from _iter_override_leaves(
                    getattr(value, config_field.name),
                    _join_hydra_path(path, config_field.name),
                )
        return

    if isinstance(value, Mapping) and value and all(_is_hydra_path_segment(key) for key in value):
        for key, child in value.items():
            yield from _iter_override_leaves(child, _join_hydra_path(path, key))
        return

    assert path, "Override help cannot render a scalar root"
    yield path, value


def _include_in_override_help(config_field: Field[Any]) -> bool:
    return config_field.init and config_field.metadata.get(_OVERRIDE_HELP_METADATA_KEY, True) is not False


def _join_hydra_path(prefix: str, segment: str) -> str:
    assert _is_hydra_path_segment(segment), f"Cannot render {segment!r} as part of a Hydra override path"
    return f"{prefix}.{segment}" if prefix else segment


def _assert_hydra_path(path: str) -> None:
    assert not path or all(
        _is_hydra_path_segment(segment) for segment in path.split(".")
    ), f"Invalid Hydra override prefix {path!r}"


def _is_hydra_path_segment(value: object) -> bool:
    return isinstance(value, str) and _HYDRA_PATH_SEGMENT_PATTERN.fullmatch(value) is not None


def _format_override_value(value: Any, path: str) -> str:
    if isinstance(value, Enum):
        return _format_string(value.name)
    if isinstance(value, Path):
        return _format_string(str(value))
    if is_dataclass(value) and not isinstance(value, type):
        value = {
            config_field.name: getattr(value, config_field.name) for config_field in fields(value) if config_field.init
        }
    if isinstance(value, Mapping):
        assert all(isinstance(key, str) for key in value), "Hydra override mappings require string keys"
        entries = (f"{_format_mapping_key(key)}:{_format_override_value(child, path)}" for key, child in value.items())
        return "{" + ",".join(entries) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_format_override_value(child, path) for child in value) + "]"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        assert value == value and value not in {
            float("inf"),
            float("-inf"),
        }, f"Cannot render non-finite Hydra override value at '{path}': {value!r}"
        return str(value)
    if isinstance(value, str):
        return _format_string(value)
    raise TypeError(f"Cannot render Hydra override value at '{path}': {value!r}")


def _format_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _format_mapping_key(value: str) -> str:
    assert value and "\n" not in value and "\r" not in value, f"Invalid Hydra mapping key {value!r}"
    return "".join(
        f"\\{character}" if character in _HYDRA_MAPPING_KEY_ESCAPE_CHARACTERS else character for character in value
    )
