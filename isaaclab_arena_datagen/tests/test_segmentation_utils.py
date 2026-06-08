# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Pure-logic tests for segmentation metadata helpers."""

from isaaclab_arena_datagen.segmentation_utils import (
    find_body_index_for_prim,
    get_label_for_instance_id,
    label_to_tracking_candidates,
)


def test_get_label_accepts_string_or_int_keys():
    labels = {"7": {"class": "robot"}, 8: {"class": "cube"}}

    assert get_label_for_instance_id(labels, 7) == {"class": "robot"}
    assert get_label_for_instance_id(labels, 8) == {"class": "cube"}


def test_label_candidates_prefer_paths_over_coarse_class():
    label = {
        "class": "robot",
        "primPath": "/World/envs/env_0/Robot/panda_link4/visuals",
    }

    assert label_to_tracking_candidates(label) == [
        "/World/envs/env_0/Robot/panda_link4/visuals",
        "robot",
    ]


def test_body_index_matches_nested_visual_meshes():
    body_names = ["panda_link0", "panda_link1", "panda_link4", "base_link", "right_inner_finger"]

    assert find_body_index_for_prim("/World/envs/env_0/Robot/panda_link4/visuals/mesh", body_names) == 2
    assert (
        find_body_index_for_prim(
            "/World/envs/env_0/Robot/Gripper/Robotiq_2F_85/right_inner_finger/collision_mesh",
            body_names,
        )
        == 4
    )


def test_body_index_does_not_guess_root_body_from_coarse_robot_label():
    body_names = ["panda_link0", "panda_link1", "panda_link2"]

    assert find_body_index_for_prim("robot", body_names) is None
