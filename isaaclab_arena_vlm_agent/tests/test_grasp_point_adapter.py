# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check authored contact geometry across objects, robot frames, and parallel environments."""

import torch
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from isaaclab_arena_vlm_agent.grasp_point_adapter import (
    GraspPointCfg,
    GraspPointObservationAdapter,
    RevolutePointMotionCfg,
)
from isaaclab_arena_vlm_agent.observation_adapter import VLMObservationAdapter


def _inputs():
    q_W_O = [0.0, 0.0, 2**-0.5, 2**-0.5]
    poses = {
        "left_arm": torch.tensor([[0, 0, 0, 0, 0, 0, 1.0], [10.0, 0, 0, *q_W_O]]),
        "right_arm": torch.tensor([[0, 0, 0, 0, 0, 0, 1.0], [10.0, 0, 0, 0, 0, 0, 1.0]]),
        "handle": torch.tensor([[0, 0, 0, 0, 0, 0, 1.0], [10.0, 2, 1, *q_W_O]]),
        "hinge": torch.tensor([[0, 0, 0, 0, 0, 0, 1.0], [10.0, 2, 1, *q_W_O]]),
    }
    world = SimpleNamespace(
        get_pose_w=Mock(side_effect=lambda name: poses[name]),
        get_body_pose_w=Mock(return_value=poses["hinge"]),
        get_aabb_in_local_frame=Mock(
            return_value=SimpleNamespace(
                min_point=torch.tensor([[-2.0, -2.0, -2.0], [-0.5, -0.25, -0.1]]),
                max_point=torch.tensor([[2.0, 2.0, 2.0], [0.5, 0.25, 0.1]]),
            )
        ),
    )
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            arena_world=world,
            action_manager=SimpleNamespace(
                active_terms=["left_arm", "right_arm"],
                get_term=lambda name: SimpleNamespace(cfg=SimpleNamespace(asset_name=name)),
            ),
        )
    )
    inner = Mock(spec=VLMObservationAdapter)
    inner.extract_vlm_input_from_observation.return_value = (
        {"camera": object()},
        {"simulator_ground_truth_assistance": {"existing": [1, 2]}, "other": True},
    )
    return env, world, poses, inner


def _points(adapter, env, env_id):
    _, values = adapter.extract_vlm_input_from_observation(env, {}, env_id)
    return values["simulator_ground_truth_assistance"]["grasp_points"]


def test_points_preserve_inputs_and_follow_live_poses_in_each_robot_root():
    env, world, poses, inner = _inputs()
    adapter = GraspPointObservationAdapter(
        inner,
        {
            "tip": GraspPointCfg("handle", offset_O_m=(1, 0, 0), normal_O=(0, 2, 0)),
            "origin": GraspPointCfg("handle"),
        },
    )
    images, values = adapter.extract_vlm_input_from_observation(env, {}, 1)
    original_images, original_values = inner.extract_vlm_input_from_observation.return_value
    assert images is original_images
    assert values["other"] is True
    assert values["simulator_ground_truth_assistance"]["existing"] == [1, 2]
    assert "grasp_points" not in original_values["simulator_ground_truth_assistance"]
    points = values["simulator_ground_truth_assistance"]["grasp_points"]
    left = points["tip"]["points_in_robot_roots"]["left_arm"]
    right = points["tip"]["points_in_robot_roots"]["right_arm"]
    assert left["position_root_m"] == pytest.approx([3, 0, 1])
    assert right["position_root_m"] == pytest.approx([0, 3, 1])
    assert left["outward_normal_root"] == pytest.approx([0, 1, 0])
    assert right["outward_normal_root"] == pytest.approx([-1, 0, 0])
    assert set(points["origin"]["points_in_robot_roots"]["left_arm"]) == {"position_root_m"}
    world.get_aabb_in_local_frame.assert_not_called()
    poses["handle"][1, 1] += 2
    updated = _points(adapter, env, 1)["tip"]["points_in_robot_roots"]["left_arm"]
    assert updated["position_root_m"] == pytest.approx([5, 0, 1])
    assert _points(adapter, env, 0)["tip"]["points_in_robot_roots"]["left_arm"]["position_root_m"] == [1, 0, 0]


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("body_name", [None, "fixed_body"])
def test_bounding_box_point_and_revolute_tangent_use_selected_environment(direction, body_name):
    env, world, _, inner = _inputs()
    point = GraspPointCfg(
        "handle",
        aabb_fraction=(1, 0.5, 0.5),
        offset_O_m=(0.1, 0, 0),
        revolute_motion=RevolutePointMotionCfg("hinge", (0, 0, 0), (0, 0, 2), body_name, direction),
    )
    adapter = GraspPointObservationAdapter(inner, {"edge": point})
    values = _points(adapter, env, 1)["edge"]["points_in_robot_roots"]["left_arm"]
    assert values["position_root_m"] == pytest.approx([2.6, 0, 1])
    assert values["motion_tangent_root"] == pytest.approx([0, direction, 0], abs=1e-7)
    world.get_aabb_in_local_frame.assert_called_once_with("handle")
    if body_name:
        world.get_body_pose_w.assert_called_once_with("hinge", body_name)
    else:
        world.get_body_pose_w.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"offset_O_m": (0, 0)},
        {"offset_O_m": (0, float("nan"), 0)},
        {"aabb_fraction": (0, 0, 1.1)},
        {"normal_O": (0, 0, 0)},
        {"normal_O": (0, float("inf"), 0)},
    ],
)
def test_invalid_contact_geometry_is_rejected(kwargs):
    with pytest.raises(AssertionError):
        GraspPointCfg("handle", **kwargs)


@pytest.mark.parametrize("axis,direction", [((0, 0, 0), 1), ((0, 0, 1), 0)])
def test_invalid_revolute_motion_is_rejected(axis, direction):
    with pytest.raises(AssertionError):
        RevolutePointMotionCfg("hinge", (0, 0, 0), axis, direction=direction)


def test_contact_on_rotation_axis_does_not_emit_a_zero_tangent():
    env, _, _, inner = _inputs()
    point = GraspPointCfg("handle", revolute_motion=RevolutePointMotionCfg("hinge", (0, 0, 0), (0, 0, 1)))
    with pytest.raises(AssertionError, match="nonzero direction"):
        _points(GraspPointObservationAdapter(inner, {"pivot": point}), env, 1)
