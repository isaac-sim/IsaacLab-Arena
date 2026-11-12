# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check that every directory of Python code in the installed packages has an ``__init__.py``.

The ``__init__.py`` is what makes a directory part of the built wheel; setuptools skips directories without one.
"""

import os
import pathlib
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Directories (relative to the repo root) that hold Python files but are deliberately not packages.
# Test directories stay unpackaged so pytest keeps importing their modules by basename.
ALLOWLIST = {
    "isaaclab_arena_curobo/tests",
    "isaaclab_arena_gr00t/tests",
    "isaaclab_arena_gr00t/tests/utils",
}


def _installed_package_roots() -> list[str]:
    """Return the top-level package names that ``pyproject.toml`` includes in the wheel."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    patterns = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
    return [pattern.rstrip("*") for pattern in patterns]


def _directories_without_init_py(package_root: str) -> list[str]:
    """Return directories under ``package_root`` holding ``.py`` files but no ``__init__.py``."""
    missing = []
    for directory, subdirectories, files in os.walk(REPO_ROOT / package_root):
        subdirectories[:] = [name for name in subdirectories if name != "__pycache__" and not name.startswith(".")]
        has_python_files = any(name.endswith(".py") for name in files)
        if has_python_files and "__init__.py" not in files:
            missing.append(pathlib.Path(directory).relative_to(REPO_ROOT).as_posix())
    return missing


def test_python_directories_have_init_py():
    # The wheel is built with ``namespaces = false``, so setuptools drops any directory without an
    # ``__init__.py`` (and everything below it) from a non-editable install.
    missing = []
    for package_root in _installed_package_roots():
        missing.extend(_directories_without_init_py(package_root))
    unexpected = sorted(set(missing) - ALLOWLIST)
    assert not unexpected, f"Directories with Python files but no __init__.py (add one or allowlist): {unexpected}"


def test_allowlist_has_no_stale_entries():
    missing = set()
    for package_root in _installed_package_roots():
        missing.update(_directories_without_init_py(package_root))
    stale = sorted(ALLOWLIST - missing)
    assert not stale, f"ALLOWLIST entries that now have an __init__.py or no Python files: {stale}"
