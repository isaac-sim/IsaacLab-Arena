# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the relation solver visualization example."""

import matplotlib

# Use a non-interactive backend so plt.show() is a no-op.
matplotlib.use("Agg")


def test_relation_solver_visualization_notebook_runs():
    """Verify the relation solver visualization example runs without errors."""
    from isaaclab_arena_examples.relations.relation_solver_visualization_notebook import run_visualization_demo

    run_visualization_demo()
