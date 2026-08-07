# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Write environment graph specs to YAML."""

from __future__ import annotations

import json
import re
import yaml
from pathlib import Path
from typing import Any

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

DEFAULT_AGENTIC_OUTPUT_DIR = Path("isaaclab_arena_environments/agent_generated")
INVALID_SPEC_FILENAME_PREFIX = "invalid_"
"""Filename prefix marking a written spec as one that failed validation."""


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


def rejected_env_graph_spec_path(env_name: str, out_dir: Path) -> Path:
    """Return the graph-spec YAML path for a rejected ``env_name`` under ``out_dir``."""
    return out_dir / f"{INVALID_SPEC_FILENAME_PREFIX}{safe_filename_stem(env_name)}.yaml"


def write_rejected_env_graph_spec(graph_spec_data: dict[str, Any], out_dir: Path, traces: tuple[str, ...] = ()) -> Path:
    """Dump a rejected environment graph spec to YAML under ``out_dir``, named as invalid.
    Args:
        graph_spec_data: Rejected spec as the agent returned it.
        out_dir: Directory to write into, created when missing.
        traces: Validation failures, written above the spec as YAML comments.

    Returns:
        The path written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    env_name = graph_spec_data.get("env_name")
    path = rejected_env_graph_spec_path(env_name if isinstance(env_name, str) else "", out_dir)
    header = "".join(f"# {line}\n" for line in ("this spec failed validation:", *traces))
    # A rejected spec can hold whatever the model returned, so render the values YAML cannot.
    plain_data = json.loads(json.dumps(graph_spec_data, default=str))
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(plain_data, f, sort_keys=False, default_flow_style=False)
    return path
