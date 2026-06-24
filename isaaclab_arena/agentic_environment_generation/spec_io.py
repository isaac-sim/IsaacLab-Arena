# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Write initial and linked environment graph specs to YAML."""

from __future__ import annotations

import re
from pathlib import Path

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec, ArenaEnvInitialGraphSpec

DEFAULT_AGENTIC_OUTPUT_DIR = Path("isaaclab_arena_environments/agent_generated")


def safe_filename_stem(name: str) -> str:
    """Return a filesystem-safe stem derived from an env name."""
    stem = re.sub(r"[^\w.-]+", "_", name).strip("._")
    return stem or "unnamed_env"


def initial_spec_path(env_name: str, out_dir: Path) -> Path:
    """Return the default initial-spec YAML path for ``env_name`` under ``out_dir``."""
    return out_dir / f"{safe_filename_stem(env_name)}_initial.yaml"


def linked_spec_path(env_name: str, out_dir: Path) -> Path:
    """Return the default linked-spec YAML path for ``env_name`` under ``out_dir``."""
    return out_dir / f"{safe_filename_stem(env_name)}_linked.yaml"


def write_env_graph_specs(
    initial_env_graph_spec: ArenaEnvInitialGraphSpec, linked_env_graph_spec: ArenaEnvGraphSpec, out_dir: Path
) -> tuple[Path, Path]:
    """Dump both environment graph specs to YAML under ``out_dir``.

    Returns:
        ``(initial_path, linked_path)``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    env_name = initial_env_graph_spec.env_name
    initial_path = initial_spec_path(env_name, out_dir)
    linked_path = linked_spec_path(env_name, out_dir)

    initial_env_graph_spec.write_yaml(initial_path)
    linked_env_graph_spec.write_yaml(linked_path)

    return initial_path, linked_path


def save_initial_graph_spec(initial_env_graph_spec: ArenaEnvInitialGraphSpec, out_dir: Path) -> tuple[Path, Path]:
    """Link ``initial_env_graph_spec`` and write initial/linked YAML files under ``out_dir``."""
    linked_env_graph_spec = initial_env_graph_spec.link()
    return write_env_graph_specs(initial_env_graph_spec, linked_env_graph_spec, out_dir)
