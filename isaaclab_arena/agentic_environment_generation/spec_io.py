# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Write environment graph specs to YAML."""

from __future__ import annotations

import re
from pathlib import Path

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec

DEFAULT_AGENTIC_OUTPUT_DIR = Path("isaaclab_arena_environments/agent_generated")


def safe_filename_stem(name: str) -> str:
    """Return a filesystem-safe stem derived from an env name."""
    stem = re.sub(r"[^\w.-]+", "_", name).strip("._")
    return stem or "unnamed_env"


def env_graph_spec_path(env_name: str, out_dir: Path) -> Path:
    """Return the default graph-spec YAML path for ``env_name`` under ``out_dir``."""
    return out_dir / f"{safe_filename_stem(env_name)}.yaml"


def write_env_graph_spec(graph_spec: ArenaEnvGraphSpec, out_dir: Path) -> Path:
    """Dump an environment graph spec to YAML under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = env_graph_spec_path(graph_spec.env_name, out_dir)
    graph_spec.write_yaml(path)
    return path
