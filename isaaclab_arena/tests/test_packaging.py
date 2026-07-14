# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the editable-install packaging of the Arena packages.

These tests need no simulator and no third-party runtime dependencies; they
validate that an installed ``isaaclab_arena`` exposes every package root,
that modules resolve through the installed mapping, and that non-Python
runtime resources are reachable relative to the package. CI runs them against
a bare ``--no-deps --editable`` install in the ``uv_package`` job.
"""

import importlib.util
from importlib import metadata
from pathlib import Path

PACKAGE_ROOTS = (
    "isaaclab_arena",
    "isaaclab_arena_curobo",
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

RUNTIME_RESOURCES = (
    "isaaclab_arena/visualization/report_template.html",
    "isaaclab_arena_environments/eval_jobs_configs/zero_action_jobs_config.json",
    "isaaclab_arena_g1/g1_env/config/loco_manip_g1_joints_order_43dof.yaml",
    "isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml",
)


def _get_repo_root() -> Path:
    """Locate the repo root from the installed ``isaaclab_arena`` package."""
    arena_spec = importlib.util.find_spec("isaaclab_arena")
    assert arena_spec is not None and arena_spec.origin is not None
    return Path(arena_spec.origin).resolve().parent.parent


def test_package_roots_importable():
    for package_root in PACKAGE_ROOTS:
        assert importlib.util.find_spec(package_root) is not None, package_root


def test_modules_discoverable():
    for module in DISCOVERABLE_MODULES:
        assert importlib.util.find_spec(module) is not None, module


def test_runtime_resources_present():
    repo_root = _get_repo_root()
    for runtime_resource in RUNTIME_RESOURCES:
        assert (repo_root / runtime_resource).is_file(), runtime_resource


def test_package_version_resolves():
    assert metadata.version("isaaclab_arena")
