# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke test that the cuRobo-enabled image carries the deps the IK-reachability code needs.

Runs only in the cuRobo image (build with ``./docker/run_docker.sh -c``); the base image has no
cuRobo. These are pure import checks — they need no SimulationApp — and guard against the gated
cuRobo / isaaclab_mimic motion-planner deps (and the ``isaaclab_arena_curobo`` package itself)
silently going missing from that image.
"""

import importlib

import pytest

pytestmark = pytest.mark.curobo_deps

# cuRobo core + the isaaclab_mimic motion-planner layer the IK oracle and planner builder rely on.
CUROBO_DEPENDENCY_MODULES = [
    "curobo",
    "curobo.types.math",
    "curobo.wrap.reacher.ik_solver",
    "curobo.wrap.reacher.motion_gen",
    "isaaclab_mimic.motion_planners.curobo.curobo_planner",
    "isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg",
]

# The cuRobo-gated package modules, which must import without a running SimulationApp.
ARENA_CUROBO_MODULES = [
    "isaaclab_arena_curobo.curobo_planner_utils",
    "isaaclab_arena_curobo.ik_utils",
    "isaaclab_arena_curobo.placement_pool_ik_validation",
]


@pytest.mark.parametrize("module_name", CUROBO_DEPENDENCY_MODULES + ARENA_CUROBO_MODULES)
def test_module_importable(module_name):
    """Each cuRobo dependency and gated arena module imports cleanly in the cuRobo image."""
    importlib.import_module(module_name)


def test_curobo_reports_version():
    """cuRobo exposes a version string, i.e. it is a real install rather than a stub on the path."""
    import curobo

    assert getattr(curobo, "__version__", None), "curobo is importable but reports no __version__"
