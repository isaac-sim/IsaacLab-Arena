# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Unproject/reproject round-trip tests for the analytic optical-flow path.

These run WITHOUT a SimulationApp / Isaac Sim: they exercise only the pure-torch
``_unproject_depth_to_world_points`` and ``SceneFlowComputer.compute_true_optical_flow``
(plus the torch-only SE(3) geometry layer).

Regression guard: the depth unprojection used to be built with
``isaaclab.utils.math.unproject_depth`` plus a ``(W, H)`` reshape/permute that did
*not* invert the hand-rolled pinhole re-projection in ``compute_true_optical_flow``.
The round trip was therefore not the identity, so a perfectly static camera with
zero 3-D scene flow produced ~100+ px of spurious radial optical flow. The fix
builds world points with the same explicit row-major ``(H, W)`` pinhole as the
re-projection, making the round trip exact (zero scene flow -> zero optical flow).
"""

import torch

from isaaclab_arena_datagen.geometry.rotation import Rotation, axis_angle_to_matrix
from isaaclab_arena_datagen.geometry.transform_se3 import TransformSE3
from isaaclab_arena_datagen.geometry.translation import Translation
from isaaclab_arena_datagen.scene_flow import SceneFlowComputer, _CachedFrame, _unproject_depth_to_world_points

CPU = torch.device("cpu")

# Sub-pixel tolerance: 1e-2 px is ~4 orders of magnitude below the ~194 px the
# regression produced for static pixels, while staying robust to float32 noise.
_FLOW_TOL_PX = 1e-2


def _intrinsics(fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=CPU)


def _depth_map(h: int, w: int, seed: int) -> torch.Tensor:
    """Strictly-positive, finite depth so every pixel is valid and in front of the camera."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return 0.5 + 2.5 * torch.rand(h, w, generator=gen, dtype=torch.float32)


def _identity_T() -> TransformSE3:
    return TransformSE3.create_identity(device=CPU)


def _nonidentity_T() -> TransformSE3:
    """A generic camera-to-world transform (rotation + off-origin translation)."""
    rotation = Rotation(R=axis_angle_to_matrix(torch.tensor([0.15, -0.30, 0.20], dtype=torch.float32)))
    translation = Translation(t=torch.tensor([0.30, -0.20, 1.20], dtype=torch.float32))
    return TransformSE3.from_rotation_translation(rotation, translation)


def _make_prev_frame(points_W_hw3: torch.Tensor, T: TransformSE3, valid_mask_hw: torch.Tensor) -> _CachedFrame:
    h, w = points_W_hw3.shape[:2]
    return _CachedFrame(
        points_W_hw3=points_W_hw3,
        T_W_from_C=T,
        track_type_hw=torch.zeros(h, w, dtype=torch.uint8),  # STATIC
        valid_mask_hw=valid_mask_hw,
        points_localbody_hw3=torch.zeros(h, w, 3, dtype=torch.float32),
        rigid_keys_hw=torch.full((h, w), -1, dtype=torch.int64),
        articulation_keys_hw=torch.full((h, w), -1, dtype=torch.int64),
        articulation_body_idx_hw=torch.full((h, w), -1, dtype=torch.int64),
    )


def _max_flow_over(flow_hw2: torch.Tensor, mask_hw: torch.Tensor) -> float:
    return torch.linalg.norm(flow_hw2, dim=-1)[mask_hw].max().item()


def test_unproject_uses_col_for_x_and_row_for_y():
    """Pixel ``(row, col)`` maps to camera ``x`` from the *column* and ``y`` from the *row*.

    This is the crux of the original bug: a ``(W, H)`` reshape/permute scrambled
    the row/column ordering of the unprojected points.
    """
    h, w = 24, 32
    fx, fy, cx, cy = 40.0, 38.0, 16.0, 12.0
    K = _intrinsics(fx, fy, cx, cy)
    depth = _depth_map(h, w, seed=0)

    pts = _unproject_depth_to_world_points(depth, K, _identity_T())  # identity -> world == camera

    assert pts.shape == (h, w, 3)
    row, col = 7, 19
    d = depth[row, col]
    expected = torch.tensor([(col - cx) / fx * d, (row - cy) / fy * d, d])
    assert torch.allclose(pts[row, col], expected, atol=1e-5)
    # Depth channel preserved exactly across the whole map.
    assert torch.allclose(pts[..., 2], depth, atol=1e-6)


def test_round_trip_zero_flow_identity_extrinsic():
    """Unproject -> reproject is the identity for an identity camera (zero scene flow)."""
    h, w = 24, 32
    K = _intrinsics(40.0, 40.0, 16.0, 12.0)
    depth = _depth_map(h, w, seed=1)
    T = _identity_T()
    valid = torch.isfinite(depth)

    comp = SceneFlowComputer()
    comp._prev = _make_prev_frame(_unproject_depth_to_world_points(depth, K, T), T, valid)
    flow = comp.compute_true_optical_flow(K, T, scene_flow_W_hw3=None)

    assert flow is not None and flow.shape == (h, w, 2)
    assert _max_flow_over(flow, valid) < _FLOW_TOL_PX


def test_round_trip_zero_flow_nonidentity_extrinsic():
    """Round trip is exact for a rotated, translated camera too."""
    h, w = 24, 32
    K = _intrinsics(45.0, 47.0, 15.0, 13.0)
    depth = _depth_map(h, w, seed=2)
    T = _nonidentity_T()
    valid = torch.isfinite(depth)

    comp = SceneFlowComputer()
    comp._prev = _make_prev_frame(_unproject_depth_to_world_points(depth, K, T), T, valid)
    flow = comp.compute_true_optical_flow(K, T, scene_flow_W_hw3=None)

    assert flow is not None
    assert _max_flow_over(flow, valid) < _FLOW_TOL_PX


def test_zero_scene_flow_gives_zero_optical_flow():
    """An explicit zero 3-D scene-flow field yields ~0 2-D optical flow."""
    h, w = 16, 20
    K = _intrinsics(50.0, 50.0, 10.0, 8.0)
    depth = _depth_map(h, w, seed=3)
    T = _nonidentity_T()
    valid = torch.isfinite(depth)

    comp = SceneFlowComputer()
    comp._prev = _make_prev_frame(_unproject_depth_to_world_points(depth, K, T), T, valid)
    flow = comp.compute_true_optical_flow(K, T, scene_flow_W_hw3=torch.zeros(h, w, 3, dtype=torch.float32))

    assert _max_flow_over(flow, valid) < _FLOW_TOL_PX


def test_localized_scene_flow_moves_only_that_pixel():
    """A single moving point produces real pixel motion while static pixels stay ~0."""
    h, w = 16, 20
    K = _intrinsics(100.0, 100.0, 10.0, 8.0)
    depth = _depth_map(h, w, seed=4)
    T = _identity_T()  # world == camera, so a world-x shift is a pure horizontal pixel shift
    valid = torch.isfinite(depth)

    comp = SceneFlowComputer()
    comp._prev = _make_prev_frame(_unproject_depth_to_world_points(depth, K, T), T, valid)
    scene_flow = torch.zeros(h, w, 3, dtype=torch.float32)
    scene_flow[5, 9, 0] = 0.05  # 5 cm sideways
    flow = comp.compute_true_optical_flow(K, T, scene_flow_W_hw3=scene_flow)

    assert torch.linalg.norm(flow[5, 9]).item() > 1.0  # clearly non-zero motion at the moved pixel
    others = valid.clone()
    others[5, 9] = False
    assert _max_flow_over(flow, others) < _FLOW_TOL_PX


def test_invalid_depth_is_masked_and_does_not_corrupt_valid_pixels():
    """Non-finite depth is masked to zero flow and never leaks into valid pixels."""
    h, w = 12, 16
    K = _intrinsics(35.0, 35.0, 8.0, 6.0)
    depth = _depth_map(h, w, seed=5)
    depth[3, 4] = float("inf")
    depth[9, 1] = float("nan")
    T = _nonidentity_T()
    valid = torch.isfinite(depth)

    comp = SceneFlowComputer()
    comp._prev = _make_prev_frame(_unproject_depth_to_world_points(depth, K, T), T, valid)
    flow = comp.compute_true_optical_flow(K, T, scene_flow_W_hw3=None)

    assert torch.all(flow[~valid] == 0.0)
    assert _max_flow_over(flow, valid) < _FLOW_TOL_PX
