# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""Companion layout schema and correlated reset selection."""

import pytest

from isaaclab_arena.relations.placement_layouts import PlacementLayouts
from isaaclab_arena.utils.pose import Pose


def test_layout_cache_round_trip_and_no_overwrite(tmp_path):
    cache = PlacementLayouts({"cup": [Pose((1, 0, 0)), Pose((2, 0, 0))], "plate": [Pose((3, 0, 0)), Pose((4, 0, 0))]})
    path = tmp_path / "poses.yaml"
    cache.write_yaml(path)
    assert PlacementLayouts.from_yaml(path) == cache
    with pytest.raises(FileExistsError):
        cache.write_yaml(path)
    assert PlacementLayouts.from_yaml(path) == cache


@pytest.mark.parametrize("poses", [{}, {"cup": []}, {"cup": [Pose()], "plate": [Pose(), Pose()]}])
def test_layout_cache_rejects_incomplete_layouts(poses):
    with pytest.raises(AssertionError):
        PlacementLayouts(poses)


@pytest.mark.parametrize("pose", [Pose((float("nan"), 0, 0)), Pose(rotation_xyzw=(0, 0, 0, 0))])
def test_layout_cache_rejects_invalid_poses(pose):
    with pytest.raises(AssertionError):
        PlacementLayouts({"cup": [pose]})


def test_layout_cache_rejects_a_missing_quaternion(tmp_path):
    path = tmp_path / "poses.yaml"
    path.write_text("cup:\n- position_xyz: [0, 0, 0]\n")
    with pytest.raises(AssertionError, match="requires"):
        PlacementLayouts.from_yaml(path)
