# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path


def _find_groot_submodule() -> str | None:
    """Locate the Isaac-GR00T submodule relative to the repo root."""
    # Walk up from this file to find the repo root (where submodules/ lives)
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "submodules" / "Isaac-GR00T"
        if candidate.is_dir():
            return str(candidate)
    return None


def ensure_groot_in_path() -> None:
    """Ensure the Isaac-GR00T submodule is importable.

    Adds the submodule to sys.path if ``gr00t`` is not already importable.
    This allows the lightweight client imports (PolicyClient, MsgSerializer)
    without requiring a full ``pip install`` of the GR00T package.

    Also prepends ``GROOT_DEPS_DIR`` to ``PYTHONPATH`` and re-execs the
    process if set, so GR00T's pip dependencies are importable before
    Isaac Sim's bundled packages.
    """
    # 1. Add submodule to sys.path if gr00t is not already importable
    try:
        import gr00t  # noqa: F401
    except ModuleNotFoundError:
        submodule_path = _find_groot_submodule()
        if submodule_path and submodule_path not in sys.path:
            sys.path.insert(0, submodule_path)

    # 2. Handle GROOT_DEPS_DIR re-exec (for heavy deps like transformers)
    deps_dir = os.environ.get("GROOT_DEPS_DIR")
    if not deps_dir or os.environ.get("_GROOT_PYTHONPATH_APPLIED") == "1":
        return

    os.environ["PYTHONPATH"] = deps_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["_GROOT_PYTHONPATH_APPLIED"] = "1"

    os.execv(sys.executable, [sys.executable] + sys.argv)


# Keep old name as alias for backward compatibility
ensure_groot_deps_in_path = ensure_groot_in_path
