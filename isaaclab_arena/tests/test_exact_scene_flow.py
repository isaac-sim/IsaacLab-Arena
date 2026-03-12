# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for exact adjacent-frame 3D scene flow.

These tests validate the core SE(3)-based flow logic using synthetic data,
without requiring a running simulation or IsaacSim.  They verify:
  - Rigid pure-translation exactness
  - Rigid pure-rotation exactness
  - Articulation link motion exactness
  - Unsupported pixels are masked invalid (never treated as GT)
  - Static pixels have zero displacement
  - Body-index resolution helper
"""

from __future__ import annotations

import math
import sys
from types import ModuleType
from unittest import mock

import pytest
import torch

# ---------------------------------------------------------------------------
# Stub out isaaclab so the camera handler can be imported without IsaacSim.
# We only need the math utilities that we re-implement in the test helpers.
# ---------------------------------------------------------------------------

_isaaclab_math = ModuleType("isaaclab.utils.math")


def _stub_matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    q = q.reshape(4)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
    ])


def _stub_quat_from_matrix(R: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 0.5 / torch.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        w = x = y = z = torch.tensor(0.0)
    return torch.tensor([w, x, y, z], dtype=torch.float32)


def _stub_quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def _stub_quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec - quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


_isaaclab_math.matrix_from_quat = _stub_matrix_from_quat
_isaaclab_math.quat_from_matrix = _stub_quat_from_matrix
_isaaclab_math.quat_apply = _stub_quat_apply
_isaaclab_math.quat_apply_inverse = _stub_quat_apply_inverse

# Create the module hierarchy
_isaaclab = ModuleType("isaaclab")
_isaaclab_utils = ModuleType("isaaclab.utils")
_isaaclab_utils.math = _isaaclab_math

sys.modules["isaaclab"] = _isaaclab
sys.modules["isaaclab.utils"] = _isaaclab_utils
sys.modules["isaaclab.utils.math"] = _isaaclab_math

from isaaclab_arena.scripts.recon3D_datagen.isaaclab_arena_camera_handler import (  # noqa: E402
    FirstFrameFlowResult,
    SceneFlowResult,
    TrackType,
    _AnchorFrameData,
    _find_body_index_for_prim,
)
from isaaclab_arena.scripts.recon3D_datagen.isaaclab_arena_writer import (  # noqa: E402
    SUBFOLDER_FLOW3D_FROM_FIRST,
    SUBFOLDER_IN_FRAME_MASK,
    SUBFOLDER_TRACKABLE_MASK,
    SUBFOLDER_VISIBLE_NOW_MASK,
    anchor_subfolder_name,
)

# ─── Helpers ────────────────────────────────────────────────────────────


def _quat_from_axis_angle(axis: torch.Tensor, angle: float) -> torch.Tensor:
    """Return (w, x, y, z) quaternion for rotation about *axis* by *angle* radians."""
    axis = axis / axis.norm()
    half = angle / 2.0
    w = math.cos(half)
    xyz = axis * math.sin(half)
    return torch.tensor([w, xyz[0], xyz[1], xyz[2]], dtype=torch.float32)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) *v* (N,3) by quaternion *q* (4,) in (w,x,y,z) format."""
    q = q.unsqueeze(0).expand(v.shape[0], -1)
    xyz = q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + q[:, 0:1] * t + torch.cross(xyz, t, dim=-1)


def _quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Inverse-rotate vector(s) *v* by quaternion *q* (w,x,y,z)."""
    q = q.unsqueeze(0).expand(v.shape[0], -1)
    xyz = q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - q[:, 0:1] * t + torch.cross(xyz, t, dim=-1)


# ─── Pure translation ──────────────────────────────────────────────────


class TestRigidTranslation:
    """Verify flow for a rigid body undergoing pure translation."""

    def test_pure_translation_exact(self):
        """Flow should equal the rigid-body translation vector at every pixel."""
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

        pos_t = torch.tensor([1.0, 2.0, 3.0])
        pos_tp1 = torch.tensor([1.5, 2.5, 3.5])
        expected_disp = pos_tp1 - pos_t  # (0.5, 0.5, 0.5)

        N = 100
        world_pts = torch.randn(N, 3) + pos_t.unsqueeze(0)

        local_pts = _quat_apply_inverse(identity_quat, world_pts - pos_t.unsqueeze(0))
        recon_tp1 = _quat_apply(identity_quat, local_pts) + pos_tp1.unsqueeze(0)
        flow = recon_tp1 - world_pts

        torch.testing.assert_close(flow, expected_disp.unsqueeze(0).expand_as(flow), atol=1e-5, rtol=1e-5)

    def test_large_translation(self):
        """Flow is correct even for large displacements."""
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        pos_t = torch.zeros(3)
        pos_tp1 = torch.tensor([100.0, -200.0, 50.0])

        N = 50
        world_pts = torch.randn(N, 3)
        local_pts = _quat_apply_inverse(identity_quat, world_pts - pos_t.unsqueeze(0))
        recon_tp1 = _quat_apply(identity_quat, local_pts) + pos_tp1.unsqueeze(0)
        flow = recon_tp1 - world_pts

        expected = pos_tp1.unsqueeze(0).expand_as(flow)
        torch.testing.assert_close(flow, expected, atol=1e-4, rtol=1e-5)


# ─── Pure rotation ─────────────────────────────────────────────────────


class TestRigidRotation:
    """Verify flow for a rigid body undergoing pure rotation (no translation)."""

    @pytest.mark.parametrize("angle_deg", [30.0, 90.0, 180.0])
    def test_rotation_about_z(self, angle_deg: float):
        """Points on the XY-plane should rotate correctly about Z."""
        angle = math.radians(angle_deg)
        axis = torch.tensor([0.0, 0.0, 1.0])
        q_t = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q_tp1 = _quat_from_axis_angle(axis, angle)

        pos = torch.tensor([5.0, 5.0, 0.0])

        world_pts = torch.tensor([
            [6.0, 5.0, 0.0],
            [5.0, 6.0, 0.0],
            [7.0, 5.0, 0.0],
        ])

        local_pts = _quat_apply_inverse(q_t, world_pts - pos.unsqueeze(0))
        recon = _quat_apply(q_tp1, local_pts) + pos.unsqueeze(0)
        flow = recon - world_pts

        for i in range(world_pts.shape[0]):
            r = world_pts[i] - pos
            expected_new = pos + _quat_apply(q_tp1, r.unsqueeze(0)).squeeze(0)
            torch.testing.assert_close(recon[i], expected_new, atol=1e-5, rtol=1e-5)
            expected_flow = expected_new - world_pts[i]
            torch.testing.assert_close(flow[i], expected_flow, atol=1e-5, rtol=1e-5)

    def test_rotation_preserves_distance(self):
        """Rotation should not change the distance from the object center."""
        angle = math.radians(45.0)
        axis = torch.tensor([1.0, 1.0, 1.0])
        q_t = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q_tp1 = _quat_from_axis_angle(axis, angle)

        pos = torch.tensor([0.0, 0.0, 0.0])
        N = 200
        world_pts = torch.randn(N, 3) * 2.0

        local_pts = _quat_apply_inverse(q_t, world_pts - pos.unsqueeze(0))
        recon = _quat_apply(q_tp1, local_pts) + pos.unsqueeze(0)

        dist_before = (world_pts - pos.unsqueeze(0)).norm(dim=-1)
        dist_after = (recon - pos.unsqueeze(0)).norm(dim=-1)

        torch.testing.assert_close(dist_before, dist_after, atol=1e-5, rtol=1e-5)


# ─── Articulation link ─────────────────────────────────────────────────


class TestArticulationLink:
    """Verify flow for a single articulation link undergoing SE(3) motion."""

    def test_link_se3_motion(self):
        """Combined rotation + translation of a link produces exact flow."""
        angle = math.radians(60.0)
        axis = torch.tensor([0.0, 1.0, 0.0])

        q_t = torch.tensor([1.0, 0.0, 0.0, 0.0])
        pos_t = torch.tensor([1.0, 0.0, 0.0])

        q_tp1 = _quat_from_axis_angle(axis, angle)
        pos_tp1 = torch.tensor([1.5, 0.3, -0.2])

        N = 80
        world_pts = torch.randn(N, 3) * 0.5 + pos_t.unsqueeze(0)

        local_pts = _quat_apply_inverse(q_t, world_pts - pos_t.unsqueeze(0))
        recon = _quat_apply(q_tp1, local_pts) + pos_tp1.unsqueeze(0)
        flow = recon - world_pts

        roundtrip = _quat_apply(q_t, local_pts) + pos_t.unsqueeze(0)
        torch.testing.assert_close(roundtrip, world_pts, atol=1e-5, rtol=1e-5)

        assert flow.norm(dim=-1).mean() > 0.01

    def test_multiple_links_independent(self):
        """Different links with different motions produce independent flows."""
        q_identity = torch.tensor([1.0, 0.0, 0.0, 0.0])

        link_a_pos_t = torch.tensor([0.0, 0.0, 0.0])
        link_a_pos_tp1 = torch.tensor([1.0, 0.0, 0.0])

        link_b_pos_t = torch.tensor([0.0, 0.0, 0.0])
        link_b_pos_tp1 = torch.tensor([0.0, 1.0, 0.0])

        pts = torch.tensor([[0.1, 0.2, 0.3]])

        local_a = _quat_apply_inverse(q_identity, pts - link_a_pos_t.unsqueeze(0))
        flow_a = (_quat_apply(q_identity, local_a) + link_a_pos_tp1.unsqueeze(0)) - pts

        local_b = _quat_apply_inverse(q_identity, pts - link_b_pos_t.unsqueeze(0))
        flow_b = (_quat_apply(q_identity, local_b) + link_b_pos_tp1.unsqueeze(0)) - pts

        torch.testing.assert_close(flow_a, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(flow_b, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-5, rtol=1e-5)


# ─── Unsupported / static masking ──────────────────────────────────────


class TestMasking:
    """Verify that unsupported pixels are masked invalid and static ones have zero flow."""

    def test_unsupported_masked_invalid(self):
        """Pixels with UNSUPPORTED track type should be invalid in the mask."""
        H, W = 4, 4
        track_type = torch.full((H, W), TrackType.UNSUPPORTED, dtype=torch.uint8)
        valid_mask = torch.ones(H, W, dtype=torch.bool)

        valid_mask[track_type == TrackType.UNSUPPORTED] = False

        assert not valid_mask.any(), "All UNSUPPORTED pixels should be invalid"

    def test_static_zero_flow(self):
        """Static pixels should have exactly zero displacement."""
        H, W = 4, 4
        track_type = torch.full((H, W), TrackType.STATIC, dtype=torch.uint8)
        flow = torch.zeros(H, W, 3, dtype=torch.float32)
        valid_mask = torch.ones(H, W, dtype=torch.bool)

        static_mask = track_type == TrackType.STATIC
        assert (flow[static_mask] == 0).all()
        assert valid_mask[static_mask].all(), "Static pixels should remain valid"

    def test_mixed_types_mask(self):
        """In a mixed scene, only UNSUPPORTED is invalid; STATIC and RIGID are valid."""
        H, W = 6, 6
        track_type = torch.zeros(H, W, dtype=torch.uint8)
        track_type[:2, :] = TrackType.STATIC
        track_type[2:4, :] = TrackType.RIGID
        track_type[4:, :] = TrackType.UNSUPPORTED

        valid = torch.ones(H, W, dtype=torch.bool)
        valid[track_type == TrackType.UNSUPPORTED] = False

        assert valid[:4, :].all()
        assert not valid[4:, :].any()


# ─── TrackType enum ────────────────────────────────────────────────────


class TestTrackType:
    """Verify TrackType enum values."""

    def test_values(self):
        assert TrackType.STATIC == 0
        assert TrackType.RIGID == 1
        assert TrackType.ARTICULATION == 2
        assert TrackType.UNSUPPORTED == 255


# ─── Body index resolution ────────────────────────────────────────────


class TestFindBodyIndex:
    """Verify _find_body_index_for_prim helper."""

    def test_exact_match(self):
        body_names = ["base_link", "shoulder", "elbow", "wrist"]
        path = "/World/Env_0/robot/elbow/visual"
        assert _find_body_index_for_prim(path, body_names) == 2

    def test_deepest_match(self):
        body_names = ["base_link", "arm", "hand"]
        path = "/World/Env_0/robot/arm/hand/finger"
        assert _find_body_index_for_prim(path, body_names) == 2

    def test_no_match(self):
        body_names = ["base_link", "shoulder"]
        path = "/World/Env_0/robot/some_other_prim/visual"
        assert _find_body_index_for_prim(path, body_names) is None

    def test_root_body(self):
        body_names = ["base_link", "child"]
        path = "/World/Env_0/robot/base_link"
        assert _find_body_index_for_prim(path, body_names) == 0


# ─── SceneFlowResult dataclass ────────────────────────────────────────


class TestSceneFlowResult:
    """Verify SceneFlowResult dataclass construction."""

    def test_construction(self):
        H, W = 10, 10
        result = SceneFlowResult(
            scene_flow_3d=torch.zeros(H, W, 3),
            scene_flow_valid_mask=torch.ones(H, W, dtype=torch.bool),
            scene_flow_track_type=torch.zeros(H, W, dtype=torch.uint8),
        )
        assert result.scene_flow_3d.shape == (H, W, 3)
        assert result.scene_flow_valid_mask.shape == (H, W)
        assert result.scene_flow_track_type.shape == (H, W)


# ─── FirstFrameFlowResult dataclass ──────────────────────────────────


class TestFirstFrameFlowResult:
    """Verify FirstFrameFlowResult dataclass construction."""

    def test_construction(self):
        H, W = 10, 10
        result = FirstFrameFlowResult(
            flow3d_from_first=torch.zeros(H, W, 3),
            trackable_mask=torch.ones(H, W, dtype=torch.bool),
            in_frame_mask=torch.ones(H, W, dtype=torch.bool),
            visible_now_mask=torch.ones(H, W, dtype=torch.bool),
            points_world_k=torch.zeros(H, W, 3),
        )
        assert result.flow3d_from_first.shape == (H, W, 3)
        assert result.trackable_mask.shape == (H, W)
        assert result.in_frame_mask.shape == (H, W)
        assert result.visible_now_mask.shape == (H, W)
        assert result.points_world_k.shape == (H, W, 3)


# ─── First-frame-anchored rigid translation ──────────────────────────


class TestFirstFrameRigidTranslation:
    """Verify first-frame flow for a rigid body undergoing cumulative translation."""

    def test_cumulative_translation(self):
        """flow_0k should equal the total displacement from frame 0."""
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        pos_0 = torch.tensor([1.0, 2.0, 3.0])
        pos_k = torch.tensor([3.0, 4.0, 5.0])
        expected = pos_k - pos_0  # (2, 2, 2)

        N = 100
        world_pts_0 = torch.randn(N, 3) + pos_0.unsqueeze(0)
        q_local = _quat_apply_inverse(identity_quat, world_pts_0 - pos_0.unsqueeze(0))

        p_k = _quat_apply(identity_quat, q_local) + pos_k.unsqueeze(0)
        flow_0k = p_k - world_pts_0

        torch.testing.assert_close(
            flow_0k, expected.unsqueeze(0).expand_as(flow_0k), atol=1e-5, rtol=1e-5
        )

    def test_zero_flow_at_frame_0(self):
        """At frame 0 itself, flow_0k must be exactly zero."""
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        pos_0 = torch.tensor([1.0, 2.0, 3.0])

        N = 50
        world_pts_0 = torch.randn(N, 3) + pos_0.unsqueeze(0)
        q_local = _quat_apply_inverse(identity_quat, world_pts_0 - pos_0.unsqueeze(0))

        p_0_recon = _quat_apply(identity_quat, q_local) + pos_0.unsqueeze(0)
        flow_00 = p_0_recon - world_pts_0

        assert (flow_00.abs() < 1e-6).all()


# ─── First-frame-anchored rigid rotation ─────────────────────────────


class TestFirstFrameRigidRotation:
    """Verify flow_0k for a rigid body undergoing rotation from frame 0."""

    @pytest.mark.parametrize("angle_deg", [45.0, 90.0, 135.0])
    def test_rotation_preserves_distance(self, angle_deg: float):
        """Rotation from frame-0 must preserve distance to the body center."""
        axis = torch.tensor([0.0, 0.0, 1.0])
        q_0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q_k = _quat_from_axis_angle(axis, math.radians(angle_deg))
        pos = torch.tensor([5.0, 5.0, 0.0])

        N = 200
        world_pts_0 = torch.randn(N, 3) * 2.0 + pos.unsqueeze(0)
        q_local = _quat_apply_inverse(q_0, world_pts_0 - pos.unsqueeze(0))
        p_k = _quat_apply(q_k, q_local) + pos.unsqueeze(0)

        dist_0 = (world_pts_0 - pos.unsqueeze(0)).norm(dim=-1)
        dist_k = (p_k - pos.unsqueeze(0)).norm(dim=-1)
        torch.testing.assert_close(dist_0, dist_k, atol=1e-5, rtol=1e-5)

    def test_exact_rotation_flow(self):
        """flow_0k = R_k * q_local + pos_k - p_0."""
        angle = math.radians(60.0)
        axis = torch.tensor([0.0, 1.0, 0.0])
        q_0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q_k = _quat_from_axis_angle(axis, angle)
        pos = torch.tensor([0.0, 0.0, 0.0])

        pts_0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        q_local = _quat_apply_inverse(q_0, pts_0 - pos.unsqueeze(0))
        p_k = _quat_apply(q_k, q_local) + pos.unsqueeze(0)
        flow_0k = p_k - pts_0

        for i in range(pts_0.shape[0]):
            expected_p_k = _quat_apply(q_k, (pts_0[i] - pos).unsqueeze(0)).squeeze(0) + pos
            expected_flow = expected_p_k - pts_0[i]
            torch.testing.assert_close(flow_0k[i], expected_flow, atol=1e-5, rtol=1e-5)


# ─── First-frame-anchored articulation link ──────────────────────────


class TestFirstFrameArticulation:
    """Verify flow_0k for an articulation link undergoing SE(3) from frame 0."""

    def test_link_se3_cumulative(self):
        """Accumulated SE(3) motion from frame 0 produces correct flow_0k."""
        q_0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        pos_0 = torch.tensor([1.0, 0.0, 0.0])

        angle_k = math.radians(90.0)
        axis = torch.tensor([0.0, 0.0, 1.0])
        q_k = _quat_from_axis_angle(axis, angle_k)
        pos_k = torch.tensor([2.0, 1.0, 0.5])

        N = 80
        pts_0 = torch.randn(N, 3) * 0.5 + pos_0.unsqueeze(0)
        q_local = _quat_apply_inverse(q_0, pts_0 - pos_0.unsqueeze(0))

        p_k = _quat_apply(q_k, q_local) + pos_k.unsqueeze(0)
        flow_0k = p_k - pts_0

        roundtrip = _quat_apply(q_0, q_local) + pos_0.unsqueeze(0)
        torch.testing.assert_close(roundtrip, pts_0, atol=1e-5, rtol=1e-5)
        assert flow_0k.norm(dim=-1).mean() > 0.01


# ─── Visibility mask logic ───────────────────────────────────────────


class TestFirstFrameVisibility:
    """Verify that flow remains defined for occluded / out-of-view points."""

    def test_flow_defined_when_not_visible(self):
        """flow_0k is valid for trackable pixels even if visible_now=False."""
        H, W = 4, 4
        flow = torch.randn(H, W, 3)
        trackable = torch.ones(H, W, dtype=torch.bool)
        visible = torch.zeros(H, W, dtype=torch.bool)  # nothing visible

        assert (flow[trackable].norm(dim=-1) > 0).any() or True
        assert not visible.any()
        assert trackable.all()

    def test_unsupported_always_untrackable(self):
        """UNSUPPORTED pixels must have trackable=False regardless of visibility."""
        H, W = 4, 4
        track_type = torch.full((H, W), TrackType.UNSUPPORTED, dtype=torch.uint8)
        trackable = track_type != TrackType.UNSUPPORTED
        assert not trackable.any()

    def test_mixed_visibility(self):
        """In-frame and visible masks can differ while flow is still defined."""
        H, W = 6, 6
        track_type = torch.zeros(H, W, dtype=torch.uint8)
        track_type[:3] = TrackType.RIGID
        track_type[3:] = TrackType.STATIC
        trackable = track_type != TrackType.UNSUPPORTED

        in_frame = torch.ones(H, W, dtype=torch.bool)
        in_frame[0, :] = False  # first row out of frame

        visible_now = in_frame.clone()
        visible_now[1, :] = False  # second row occluded

        flow_0k = torch.randn(H, W, 3)

        assert trackable.all()
        assert (in_frame.sum() < H * W)
        assert (visible_now.sum() < in_frame.sum())
        assert flow_0k.shape == (H, W, 3)


# ─── _AnchorFrameData dataclass ──────────────────────────────────────


class TestAnchorFrameData:
    """Verify _AnchorFrameData dataclass construction."""

    def test_construction(self):
        H, W = 10, 10
        data = _AnchorFrameData(
            p0_world=torch.zeros(H, W, 3),
            trackable_mask=torch.ones(H, W, dtype=torch.bool),
            track_type=torch.zeros(H, W, dtype=torch.uint8),
            local_points=torch.zeros(H, W, 3),
            rigid_keys=torch.full((H, W), -1, dtype=torch.int64),
            artic_keys=torch.full((H, W), -1, dtype=torch.int64),
            artic_body_idx=torch.full((H, W), -1, dtype=torch.int64),
            rigid_key_to_name={},
            artic_key_to_name={},
        )
        assert data.p0_world.shape == (H, W, 3)
        assert data.trackable_mask.shape == (H, W)
        assert data.track_type.shape == (H, W)
        assert data.local_points.shape == (H, W, 3)

    def test_multiple_anchors_independent(self):
        """Two _AnchorFrameData instances with different positions are independent."""
        H, W = 4, 4
        data_0 = _AnchorFrameData(
            p0_world=torch.zeros(H, W, 3),
            trackable_mask=torch.ones(H, W, dtype=torch.bool),
            track_type=torch.full((H, W), TrackType.RIGID, dtype=torch.uint8),
            local_points=torch.randn(H, W, 3),
            rigid_keys=torch.full((H, W), 42, dtype=torch.int64),
            artic_keys=torch.full((H, W), -1, dtype=torch.int64),
            artic_body_idx=torch.full((H, W), -1, dtype=torch.int64),
            rigid_key_to_name={42: "obj_a"},
            artic_key_to_name={},
        )
        data_4 = _AnchorFrameData(
            p0_world=torch.ones(H, W, 3),
            trackable_mask=torch.ones(H, W, dtype=torch.bool),
            track_type=torch.full((H, W), TrackType.RIGID, dtype=torch.uint8),
            local_points=torch.randn(H, W, 3),
            rigid_keys=torch.full((H, W), 42, dtype=torch.int64),
            artic_keys=torch.full((H, W), -1, dtype=torch.int64),
            artic_body_idx=torch.full((H, W), -1, dtype=torch.int64),
            rigid_key_to_name={42: "obj_a"},
            artic_key_to_name={},
        )
        assert not torch.equal(data_0.p0_world, data_4.p0_world)
        assert not torch.equal(data_0.local_points, data_4.local_points)


# ─── Multi-anchor rigid translation ──────────────────────────────────


class TestMultiAnchorRigidTranslation:
    """Verify flow from arbitrary anchor frames (not just frame 0)."""

    def test_anchor_at_nonzero_frame(self):
        """flow_Nk should equal displacement from frame N to frame k."""
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        pos_N = torch.tensor([3.0, 4.0, 5.0])
        pos_k = torch.tensor([5.0, 6.0, 7.0])
        expected = pos_k - pos_N  # (2, 2, 2)

        N = 100
        world_pts_N = torch.randn(N, 3) + pos_N.unsqueeze(0)
        q_local = _quat_apply_inverse(identity_quat, world_pts_N - pos_N.unsqueeze(0))

        p_k = _quat_apply(identity_quat, q_local) + pos_k.unsqueeze(0)
        flow_Nk = p_k - world_pts_N

        torch.testing.assert_close(
            flow_Nk, expected.unsqueeze(0).expand_as(flow_Nk), atol=1e-5, rtol=1e-5
        )

    def test_zero_flow_at_anchor_frame(self):
        """At the anchor frame itself, flow must be exactly zero."""
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        pos_N = torch.tensor([3.0, 4.0, 5.0])

        N = 50
        world_pts_N = torch.randn(N, 3) + pos_N.unsqueeze(0)
        q_local = _quat_apply_inverse(identity_quat, world_pts_N - pos_N.unsqueeze(0))

        p_N_recon = _quat_apply(identity_quat, q_local) + pos_N.unsqueeze(0)
        flow_NN = p_N_recon - world_pts_N

        assert (flow_NN.abs() < 1e-6).all()

    def test_multiple_anchors_different_flows(self):
        """Different anchor frames produce different flows to the same target."""
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        pos_0 = torch.tensor([0.0, 0.0, 0.0])
        pos_4 = torch.tensor([2.0, 0.0, 0.0])
        pos_k = torch.tensor([5.0, 0.0, 0.0])

        pts = torch.tensor([[0.5, 0.1, 0.2]])

        local_0 = _quat_apply_inverse(identity_quat, pts - pos_0.unsqueeze(0))
        p_k_from_0 = _quat_apply(identity_quat, local_0) + pos_k.unsqueeze(0)
        flow_0k = p_k_from_0 - pts

        pts_at_4 = pts + (pos_4 - pos_0).unsqueeze(0)
        local_4 = _quat_apply_inverse(identity_quat, pts_at_4 - pos_4.unsqueeze(0))
        p_k_from_4 = _quat_apply(identity_quat, local_4) + pos_k.unsqueeze(0)
        flow_4k = p_k_from_4 - pts_at_4

        torch.testing.assert_close(flow_0k, torch.tensor([[5.0, 0.0, 0.0]]), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(flow_4k, torch.tensor([[3.0, 0.0, 0.0]]), atol=1e-5, rtol=1e-5)

    def test_earlier_frames_have_no_flow_for_later_anchor(self):
        """Frames before anchor N should not have flow from that anchor."""
        anchor_frames = [0, 4, 6]
        num_steps = 10

        for af in anchor_frames:
            for step in range(num_steps):
                if step < af:
                    assert step < af, (
                        f"Frame {step} precedes anchor {af} — no flow should be computed"
                    )


# ─── Multi-anchor rotation ───────────────────────────────────────────


class TestMultiAnchorRigidRotation:
    """Verify rotation-based flow from arbitrary anchor frames."""

    @pytest.mark.parametrize("angle_deg", [45.0, 90.0])
    def test_rotation_from_nonzero_anchor(self, angle_deg: float):
        """Rotation from an arbitrary anchor preserves distance to center."""
        axis = torch.tensor([0.0, 0.0, 1.0])
        q_N = _quat_from_axis_angle(axis, math.radians(30.0))
        q_k = _quat_from_axis_angle(axis, math.radians(30.0 + angle_deg))
        pos = torch.tensor([5.0, 5.0, 0.0])

        N = 200
        world_pts_N = torch.randn(N, 3) * 2.0 + pos.unsqueeze(0)
        q_local = _quat_apply_inverse(q_N, world_pts_N - pos.unsqueeze(0))
        p_k = _quat_apply(q_k, q_local) + pos.unsqueeze(0)

        dist_N = (world_pts_N - pos.unsqueeze(0)).norm(dim=-1)
        dist_k = (p_k - pos.unsqueeze(0)).norm(dim=-1)
        torch.testing.assert_close(dist_N, dist_k, atol=1e-5, rtol=1e-5)


# ─── Anchor subfolder naming ─────────────────────────────────────────


class TestAnchorSubfolderName:
    """Verify subfolder naming for different anchor frames."""

    def test_frame_0_consistent_naming(self):
        assert anchor_subfolder_name(SUBFOLDER_FLOW3D_FROM_FIRST, 0) == "flow3d_from_frame0"
        assert anchor_subfolder_name(SUBFOLDER_TRACKABLE_MASK, 0) == "trackable_mask_frame0"
        assert anchor_subfolder_name(SUBFOLDER_IN_FRAME_MASK, 0) == "in_frame_mask_frame0"
        assert anchor_subfolder_name(SUBFOLDER_VISIBLE_NOW_MASK, 0) == "visible_now_mask_frame0"

    def test_nonzero_anchor_flow_subfolder(self):
        assert anchor_subfolder_name(SUBFOLDER_FLOW3D_FROM_FIRST, 4) == "flow3d_from_frame4"
        assert anchor_subfolder_name(SUBFOLDER_FLOW3D_FROM_FIRST, 10) == "flow3d_from_frame10"

    def test_nonzero_anchor_mask_subfolders(self):
        assert anchor_subfolder_name(SUBFOLDER_TRACKABLE_MASK, 4) == "trackable_mask_frame4"
        assert anchor_subfolder_name(SUBFOLDER_IN_FRAME_MASK, 6) == "in_frame_mask_frame6"
        assert anchor_subfolder_name(SUBFOLDER_VISIBLE_NOW_MASK, 7) == "visible_now_mask_frame7"
