# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Strict YAML loading for typed dataclass configurations."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

CfgT = TypeVar("CfgT")


def render_typed_yaml_cfg(cfg: CfgT) -> str:
    """Render a structured dataclass configuration as YAML.

    Args:
        cfg: Dataclass configuration instance to render.

    Returns:
        The YAML representation of the configuration.
    """
    try:
        config = OmegaConf.structured(cfg)
        return OmegaConf.to_yaml(config, resolve=False)
    except (OmegaConfBaseException, TypeError, ValueError) as exc:
        raise ValueError(f"Could not render typed config as YAML: {exc}") from exc


def load_typed_yaml_cfg(
    path: str | Path,
    cfg_type: type[CfgT],
    *,
    config_name: str,
) -> CfgT:
    """Load YAML into a structured dataclass configuration.

    Unknown fields and values incompatible with the dataclass schema are rejected.

    Args:
        path: YAML configuration file to load.
        cfg_type: Dataclass type defining the configuration schema.
        config_name: Descriptive configuration name used in validation errors.

    Returns:
        The loaded dataclass configuration.
    """
    config_path = Path(path).expanduser().resolve()
    assert config_path.is_file(), f"{config_name} config does not exist: '{config_path}'"
    assert config_path.suffix.lower() in {
        ".yaml",
        ".yml",
    }, f"{config_name} config must be YAML, got '{config_path}'"

    try:
        schema = OmegaConf.structured(cfg_type)
        OmegaConf.set_struct(schema, True)
        config = OmegaConf.merge(schema, OmegaConf.load(config_path))
        OmegaConf.resolve(config)
        loaded = OmegaConf.to_object(config)
    except (OmegaConfBaseException, TypeError, ValueError) as exc:
        raise ValueError(f"Could not load {config_name} config from '{config_path}': {exc}") from exc

    assert isinstance(loaded, cfg_type)
    return loaded
