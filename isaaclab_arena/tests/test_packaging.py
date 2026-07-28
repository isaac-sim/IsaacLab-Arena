# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the editable-install packaging of the Arena packages.

These tests need no simulator and no third-party runtime dependencies; they
validate that an installed ``isaaclab_arena`` exposes every package root and
that modules resolve through the installed mapping. CI runs them against a
bare ``--no-deps --editable`` install in the native uv job.
"""

import importlib.util
import tomllib
from importlib import metadata
from pathlib import Path

PACKAGE_ROOTS = (
    "isaaclab_arena",
    "isaaclab_arena_curobo",
    "isaaclab_arena_dreamzero",
    "isaaclab_arena_environments",
    "isaaclab_arena_examples",
    "isaaclab_arena_g1",
    "isaaclab_arena_gr00t",
    "isaaclab_arena_openpi",
)

DISCOVERABLE_MODULES = (
    "isaaclab_arena.evaluation.policy_runner",
    "isaaclab_arena.tasks.sequential_composite_tasks.franka_put_and_close_door_task",
)

REPOSITORY_ROOT = Path(__file__).parents[2]


def test_package_roots_importable():
    for package_root in PACKAGE_ROOTS:
        assert importlib.util.find_spec(package_root) is not None, package_root


def test_modules_discoverable():
    for module in DISCOVERABLE_MODULES:
        assert importlib.util.find_spec(module) is not None, module


def test_package_version_resolves():
    assert metadata.version("isaaclab_arena")


def test_native_openpi_client_matches_server_commit():
    """Keep the native OpenPI client wire-compatible with the separately built server."""
    pyproject_config = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    openpi_client_source = pyproject_config["tool"]["uv"]["sources"]["openpi-client"]
    openpi_server_commit = (
        (REPOSITORY_ROOT / "isaaclab_arena_openpi" / "docker" / "OPENPI_COMMIT").read_text(encoding="utf-8").strip()
    )

    assert pyproject_config["dependency-groups"]["openpi"] == ["openpi-client"]
    assert "openpi" in pyproject_config["tool"]["uv"]["default-groups"]
    assert openpi_client_source["rev"] == openpi_server_commit
