# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Utilities for importing lightweight modules from the Isaac-GR00T submodule."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _find_gr00t_source_root() -> Path:
    """Return the local Isaac-GR00T submodule root."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "submodules" / "Isaac-GR00T"
        if (candidate / "gr00t" / "__init__.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find submodules/Isaac-GR00T. Initialize the GR00T submodule or install gr00t "
        "into the active Python environment."
    )


def ensure_gr00t_importable() -> Path | None:
    """Add the local Isaac-GR00T checkout to sys.path when gr00t is not installed."""
    try:
        gr00t_spec = importlib.util.find_spec("gr00t")
    except ValueError:
        gr00t_spec = None

    if gr00t_spec is not None:
        return None

    gr00t_source_root = _find_gr00t_source_root()
    gr00t_source_root_str = str(gr00t_source_root)
    if gr00t_source_root_str not in sys.path:
        sys.path.insert(0, gr00t_source_root_str)
    return gr00t_source_root
