# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Simulator-free tests for the spatial-memory box geometry helpers."""

import math
import torch

from isaaclab_arena_environments.spatial_memory.geometry import (
    FACE_PZ,
    compose_poses,
    inverse_transform_points,
    points_in_pinhole_frustum,
    pose_from_position_yaw,
    sample_box_face_points,
    segments_blocked_by_boxes,
    transform_points,
)


def test_face_points_lie_just_outside_the_requested_face():
    points = sample_box_face_points((0.2, 0.1, 0.04), faces=(FACE_PZ,), spacing_m=0.05, outward_offset_m=0.002)
    assert torch.allclose(points[:, 2], torch.full((points.shape[0],), 0.022))
    assert points[:, 0].abs().max() <= 0.1 + 1e-6
    assert points[:, 1].abs().max() <= 0.05 + 1e-6
    # 0.2 / 0.05 -> 5 points along x, 0.1 / 0.05 -> 3 points along y.
    assert points.shape == (15, 3)


def test_transform_round_trip():
    T_W_B = pose_from_position_yaw(torch.tensor([[1.0, 2.0, 0.5]]), torch.tensor([0.7]))
    points_B = torch.tensor([[[0.1, 0.0, 0.0], [0.0, 0.2, 0.3]]])
    points_W = transform_points(T_W_B, points_B)
    assert torch.allclose(inverse_transform_points(T_W_B, points_W), points_B, atol=1e-6)
    # A yaw of 0.7 rad moves +x of B to (cos, sin) in W.
    assert torch.allclose(points_W[0, 0], torch.tensor([1.0 + 0.1 * math.cos(0.7), 2.0 + 0.1 * math.sin(0.7), 0.5]))


def test_compose_matches_sequential_transform():
    T_A_B = pose_from_position_yaw(torch.tensor([[0.3, -0.1, 0.2]]), torch.tensor([1.1]))
    T_B_C = pose_from_position_yaw(torch.tensor([[0.0, 0.5, 0.0]]), torch.tensor([-0.4]))
    points_C = torch.tensor([[[0.2, 0.1, -0.3]]])
    direct = transform_points(compose_poses(T_A_B, T_B_C), points_C)
    sequential = transform_points(T_A_B, transform_points(T_B_C, points_C))
    assert torch.allclose(direct, sequential, atol=1e-6)


def test_frustum_accepts_centre_and_rejects_behind_and_outside():
    points_C = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [2.0, 0.0, 1.0], [0.0, 0.0, 0.01]])
    inside = points_in_pinhole_frustum(points_C, 50.0, 50.0, 100, 100, near_m=0.05, far_m=10.0)
    assert inside.tolist() == [True, False, False, False]


def test_box_blocks_segment_only_when_between_origin_and_point():
    origins = torch.zeros(1, 3)
    points = torch.tensor([[[2.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 1.0, 0.0]]])
    box_pose = pose_from_position_yaw(torch.tensor([[[1.0, 0.0, 0.0]]]))
    half = torch.tensor([[0.1, 0.1, 0.1]])
    blocked = segments_blocked_by_boxes(origins, points, box_pose, half)
    assert blocked.tolist() == [[True, False, False]]


def test_point_on_front_face_is_not_blocked_by_its_own_box():
    points_B = sample_box_face_points((0.2, 0.2, 0.2), faces=(FACE_PZ,), spacing_m=0.1)
    box_pose = pose_from_position_yaw(torch.tensor([[[0.0, 0.0, 0.0]]]))
    points_W = transform_points(box_pose[:, 0], points_B.unsqueeze(0))
    camera_above = torch.tensor([[0.0, 0.0, 1.0]])
    camera_below = torch.tensor([[0.0, 0.0, -1.0]])
    half = torch.tensor([[0.1, 0.1, 0.1]])
    assert not segments_blocked_by_boxes(camera_above, points_W, box_pose, half).any()
    assert segments_blocked_by_boxes(camera_below, points_W, box_pose, half).all()


def test_disabled_box_never_blocks():
    origins = torch.zeros(2, 3)
    points = torch.tensor([[[2.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])
    box_pose = pose_from_position_yaw(torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]))
    half = torch.tensor([[0.1, 0.1, 0.1]])
    enabled = torch.tensor([[True], [False]])
    blocked = segments_blocked_by_boxes(origins, points, box_pose, half, box_enabled=enabled)
    assert blocked.tolist() == [[True], [False]]
