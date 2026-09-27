# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check tool-hanging goal geometry."""

import torch

import pytest

from isaaclab_arena_environments.isaac_cap.tool_hanging.geometry import goal_geometry_from_dict

pytestmark = pytest.mark.isaac_cap

# Tool at the origin; fixture translated by one meter along X with a quarter turn about Z.
T_W_T = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
T_W_X = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 2**-0.5, 2**-0.5]])


def test_loop_on_rod_uses_closest_segment_point():
    # The fixture's local X rod becomes a world Y segment through (1, 0, 0).
    goal = goal_geometry_from_dict({
        "loop": {"center_xyz": [1.0, 0.3, 0.005], "radius_m": 0.01},
        "rod": {"start_xyz": [0, 0, 0], "end_xyz": [0.5, 0, 0]},
    })
    assert goal.evaluate(T_W_T, T_W_X).tolist() == [True]
    goal.loops[0].center_xyz = (1.0, 0.6, 0.0)  # beyond the rod end
    assert goal.evaluate(T_W_T, T_W_X).tolist() == [False]


def test_point_in_box_ignores_fixture_rotation():
    goal = goal_geometry_from_dict({
        "containment": {
            "minimum_xyz": [-1.1, -0.1, -0.1],
            "maximum_xyz": [-0.9, 0.1, 0.1],
            "point_xyz": [0.0, 0.0, 0.05],
        }
    })
    assert goal.evaluate(T_W_T, T_W_X).tolist() == [True]
    goal.point_xyz = (0.0, 0.0, 0.2)
    assert goal.evaluate(T_W_T, T_W_X).tolist() == [False]
