# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for relation solver example scripts.

These tests simply verify the examples run without crashing.
"""

import matplotlib

# Use non-interactive backend so plt.show() is a no-op
matplotlib.use("Agg")


def test_relation_solver_visualization_notebook_runs():
    """Smoke test: verify the visualization notebook runs without errors."""
    from isaaclab_arena_examples.relations.relation_solver_visualization_notebook import run_visualization_demo

    run_visualization_demo()


def test_dummy_object_placer_notebook_runs():
    """Smoke test: verify the dummy object placer notebook runs without errors."""
    from isaaclab_arena_examples.relations.dummy_object_placer_notebook import run_dummy_object_placer_demo

    run_dummy_object_placer_demo()


def test_isaac_sim_object_placer_smoke():
    """Smoke test: verify the Isaac Sim object placer notebook runs without errors."""
    from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
    from isaaclab_arena_examples.relations.isaac_sim_object_placer_notebook import smoke_test_isaac_sim_object_placer

    result = run_simulation_app_function(smoke_test_isaac_sim_object_placer)
    assert result, "Isaac Sim object placer smoke test failed"
