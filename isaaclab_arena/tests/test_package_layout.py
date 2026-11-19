# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check that every directory of Python code in the installed packages is installed by setuptools.

The ``__init__.py`` is what makes a directory part of the built wheel; setuptools skips directories without one.
"""

import os
import pathlib
import tomllib

from setuptools import find_packages

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Directories (relative to the repo root) that hold Python files but are deliberately not packages.
# Test directories stay unpackaged so pytest keeps importing their modules by basename.
ALLOWLIST = {
    "isaaclab_arena_curobo/tests",
    "isaaclab_arena_policies/isaaclab_arena_gr00t/tests",
    "isaaclab_arena_policies/isaaclab_arena_gr00t/tests/utils",
}


def _setuptools_find_config() -> dict:
    """Return the ``[tool.setuptools.packages.find]`` table of ``pyproject.toml``."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    return pyproject["tool"]["setuptools"]["packages"]["find"]


def _directories_with_python_files(source_root: pathlib.Path, include: list[str]) -> list[pathlib.Path]:
    """Return the directories under ``source_root`` that hold ``.py`` files and match an ``include`` pattern."""
    directories = []
    for pattern in include:
        for directory, subdirectories, files in os.walk(source_root / pattern.rstrip("*")):
            subdirectories[:] = [name for name in subdirectories if name != "__pycache__" and not name.startswith(".")]
            if any(name.endswith(".py") for name in files):
                directories.append(pathlib.Path(directory))
    return directories


def _python_directories_not_installed() -> set[str]:
    """Return directories holding ``.py`` files that setuptools does not install.

    A directory is installed only if it and every parent up to the top-level package have an ``__init__.py``,
    so a missing parent is caught even when the parent has no ``.py`` files of its own.
    """
    config = _setuptools_find_config()
    include = config["include"]
    exclude = config.get("exclude", [])
    not_installed = set()
    for where in config["where"]:
        source_root = REPO_ROOT / where
        installed = set(find_packages(where=str(source_root), include=include, exclude=exclude))
        for directory in _directories_with_python_files(source_root, include):
            package_name = ".".join(directory.relative_to(source_root).parts)
            if package_name not in installed:
                not_installed.add(directory.relative_to(REPO_ROOT).as_posix())
    return not_installed


def test_python_directories_are_installed():
    # The wheel is built with ``namespaces = false``, so setuptools drops any directory without an
    # ``__init__.py`` (and everything below it) from a non-editable install.
    unexpected = sorted(_python_directories_not_installed() - ALLOWLIST)
    assert not unexpected, f"Directories with Python files not installed by setuptools (add __init__.py): {unexpected}"


def test_allowlist_has_no_stale_entries():
    stale = sorted(ALLOWLIST - _python_directories_not_installed())
    assert not stale, f"ALLOWLIST entries that are now installed or have no Python files: {stale}"
