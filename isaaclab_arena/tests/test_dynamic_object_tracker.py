# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DynamicObjectTracker.

Validates pose recording, visibility tracking, motion filtering, and hybrid
JSON + .npz output without requiring a running simulation or IsaacSim.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from types import ModuleType

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Stub out isaaclab (same pattern as test_exact_scene_flow.py)
# ---------------------------------------------------------------------------

_isaaclab_math = ModuleType("isaaclab.utils.math")


def _stub_matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    q = q.reshape(4)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
    ])


def _stub_quat_from_matrix(R: torch.Tensor) -> torch.Tensor:
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

_isaaclab = ModuleType("isaaclab")
_isaaclab_utils = ModuleType("isaaclab.utils")
_isaaclab_utils.math = _isaaclab_math

sys.modules["isaaclab"] = _isaaclab
sys.modules["isaaclab.utils"] = _isaaclab_utils
sys.modules["isaaclab.utils.math"] = _isaaclab_math

from isaaclab_arena.scripts.recon3D_datagen.isaaclab_arena_camera_handler import (  # noqa: E402
    DynamicObjectTracker,
    MeshSamplesResult,
    ObjectInstanceRegistry,
    _rotation_from_normal,
    reconstruct_mesh_points_at_step,
)
from isaaclab_arena.scripts.recon3D_datagen.isaaclab_arena_writer import (  # noqa: E402
    IsaacLabArenaWriter,
)

NUM_STEPS = 5


# ---------------------------------------------------------------------------
# Helpers: fake scene / env
# ---------------------------------------------------------------------------

def _identity_quat():
    return torch.tensor([1.0, 0.0, 0.0, 0.0])


def _quat_from_axis_angle(axis, angle_rad):
    """Return (w, x, y, z) quaternion for rotation about *axis* by *angle_rad*."""
    axis = torch.tensor(axis, dtype=torch.float32)
    axis = axis / axis.norm()
    half = angle_rad / 2.0
    w = math.cos(half)
    xyz = axis * math.sin(half)
    return torch.tensor([w, xyz[0].item(), xyz[1].item(), xyz[2].item()])


class _FakeRigidData:
    def __init__(self, pos, quat):
        self.root_link_pos_w = pos.unsqueeze(0)
        self.root_link_quat_w = quat.unsqueeze(0)


class _FakeRigidObject:
    def __init__(self, pos, quat):
        self.data = _FakeRigidData(pos, quat)


class _FakeArticData:
    def __init__(self, body_positions, body_quats, body_names):
        self.body_link_pos_w = body_positions.unsqueeze(0)
        self.body_link_quat_w = body_quats.unsqueeze(0)
        self.body_names = body_names


class _FakeArticulation:
    def __init__(self, body_positions, body_quats, body_names):
        self.data = _FakeArticData(body_positions, body_quats, body_names)


class _FakeScene:
    def __init__(self, rigid_dict, artic_dict):
        self._rigid = rigid_dict
        self._artic = artic_dict

    @property
    def rigid_objects(self):
        return self._rigid

    @property
    def articulations(self):
        return self._artic

    def __getitem__(self, key):
        if key in self._rigid:
            return self._rigid[key]
        if key in self._artic:
            return self._artic[key]
        raise KeyError(key)


class _FakeEnv:
    def __init__(self, scene):
        self._scene = scene

    @property
    def unwrapped(self):
        return self

    @property
    def scene(self):
        return self._scene


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegisterVisibleObjects:
    def test_collects_rigid_and_articulation(self):
        registry = ObjectInstanceRegistry()
        tracker = DynamicObjectTracker(registry, num_steps=NUM_STEPS)

        semantic_info = [
            {"track_type": "STATIC", "asset_name": "ground"},
            {"track_type": "RIGID", "asset_name": "box_a"},
            {"track_type": "ARTICULATION", "asset_name": "robot"},
        ]
        tracker.register_visible_objects(semantic_info)

        assert ("RIGID", "box_a") in tracker._seen_assets
        assert ("ARTICULATION", "robot") in tracker._seen_assets
        assert ("STATIC", "ground") not in tracker._seen_assets

    def test_deduplicates_across_calls(self):
        registry = ObjectInstanceRegistry()
        tracker = DynamicObjectTracker(registry, num_steps=NUM_STEPS)

        info = [{"track_type": "RIGID", "asset_name": "box_a"}]
        tracker.register_visible_objects(info)
        tracker.register_visible_objects(info)

        assert len(tracker._seen_assets) == 1


class TestRecordStepPoses:
    def _make_env(self, box_pos, box_quat):
        rigid = {"box_a": _FakeRigidObject(box_pos, box_quat)}
        return _FakeEnv(_FakeScene(rigid, {}))

    def test_records_rigid_pose(self):
        registry = ObjectInstanceRegistry()
        tracker = DynamicObjectTracker(registry, num_steps=NUM_STEPS)

        pos = torch.tensor([1.0, 2.0, 3.0])
        quat = _identity_quat()
        env = self._make_env(pos, quat)

        tracker.record_step_poses(env, 0)

        assert "box_a" in tracker._rigid_poses
        T = tracker._rigid_poses["box_a"][0]
        assert T.shape == (4, 4)
        assert torch.allclose(T[:3, 3], pos, atol=1e-6)
        assert torch.allclose(T[:3, :3], torch.eye(3), atol=1e-6)

    def test_preallocated_tensor_shape(self):
        registry = ObjectInstanceRegistry()
        tracker = DynamicObjectTracker(registry, num_steps=NUM_STEPS)

        pos = torch.tensor([1.0, 2.0, 3.0])
        env = self._make_env(pos, _identity_quat())
        tracker.record_step_poses(env, 0)

        assert tracker._rigid_poses["box_a"].shape == (NUM_STEPS, 4, 4)

    def test_records_articulation_parts(self):
        registry = ObjectInstanceRegistry()
        tracker = DynamicObjectTracker(registry, num_steps=NUM_STEPS)

        body_pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        body_quat = torch.stack([_identity_quat(), _identity_quat()])
        artic = {"robot": _FakeArticulation(body_pos, body_quat, ["base", "link1"])}
        env = _FakeEnv(_FakeScene({}, artic))

        tracker.record_step_poses(env, 0)

        assert "robot" in tracker._artic_poses
        assert tracker._artic_poses["robot"].shape == (NUM_STEPS, 2, 4, 4)
        assert tracker._artic_body_names["robot"] == ["base", "link1"]
        T_link1 = tracker._artic_poses["robot"][0, 1]
        assert torch.allclose(T_link1[:3, 3], torch.tensor([1.0, 0.0, 0.0]))


class TestMotionDetection:
    def test_static_object_is_filtered(self):
        """An object that never moves should not appear in output."""
        registry = ObjectInstanceRegistry()
        tracker = DynamicObjectTracker(registry, num_steps=NUM_STEPS)

        pos = torch.tensor([1.0, 2.0, 3.0])
        quat = _identity_quat()
        tracker._seen_assets.add(("RIGID", "box_a"))

        for step in range(NUM_STEPS):
            rigid = {"box_a": _FakeRigidObject(pos, quat)}
            env = _FakeEnv(_FakeScene(rigid, {}))
            tracker.record_step_poses(env, step)

        result = tracker.get_dynamic_object_data(motion_eps=1e-4)
        assert len(result.objects_metadata) == 0
        assert len(result.pose_arrays) == 0

    def test_translating_object_is_detected(self):
        """An object that translates should appear in output."""
        registry = ObjectInstanceRegistry()
        tracker = DynamicObjectTracker(registry, num_steps=NUM_STEPS)
        tracker._seen_assets.add(("RIGID", "box_a"))

        for step in range(NUM_STEPS):
            pos = torch.tensor([float(step), 0.0, 0.0])
            quat = _identity_quat()
            rigid = {"box_a": _FakeRigidObject(pos, quat)}
            env = _FakeEnv(_FakeScene(rigid, {}))
            tracker.record_step_poses(env, step)

        result = tracker.get_dynamic_object_data(motion_eps=1e-4)
        assert len(result.objects_metadata) == 1
        obj_key = list(result.objects_metadata.keys())[0]
        assert result.objects_metadata[obj_key]["type"] == "rigid"
        assert obj_key in result.pose_arrays
        assert result.pose_arrays[obj_key].shape == (NUM_STEPS, 3, 4)

    def test_rotating_object_is_detected(self):
        """An object with pure rotation should be detected as dynamic."""
        registry = ObjectInstanceRegistry()
        tracker = DynamicObjectTracker(registry, num_steps=NUM_STEPS)
        tracker._seen_assets.add(("RIGID", "box_a"))

        pos = torch.tensor([0.0, 0.0, 0.0])
        for step in range(NUM_STEPS):
            angle = step * 0.1
            quat = _quat_from_axis_angle([0.0, 0.0, 1.0], angle)
            rigid = {"box_a": _FakeRigidObject(pos, quat)}
            env = _FakeEnv(_FakeScene(rigid, {}))
            tracker.record_step_poses(env, step)

        result = tracker.get_dynamic_object_data(motion_eps=1e-4)
        assert len(result.objects_metadata) == 1

    def test_articulation_with_moving_link(self):
        """An articulation where one link moves should appear."""
        registry = ObjectInstanceRegistry()
        tracker = DynamicObjectTracker(registry, num_steps=NUM_STEPS)
        tracker._seen_assets.add(("ARTICULATION", "robot"))

        for step in range(NUM_STEPS):
            base_pos = torch.tensor([0.0, 0.0, 0.0])
            link_pos = torch.tensor([float(step) * 0.1, 0.0, 0.0])
            body_pos = torch.stack([base_pos, link_pos])
            body_quat = torch.stack([_identity_quat(), _identity_quat()])
            artic = {"robot": _FakeArticulation(body_pos, body_quat, ["base", "link1"])}
            env = _FakeEnv(_FakeScene({}, artic))
            tracker.record_step_poses(env, step)

        result = tracker.get_dynamic_object_data(motion_eps=1e-4)
        assert len(result.objects_metadata) == 1
        obj_key = list(result.objects_metadata.keys())[0]
        obj = result.objects_metadata[obj_key]
        assert obj["type"] == "articulation"
        assert "parts" in obj
        assert "base" in obj["parts"]
        assert "link1" in obj["parts"]
        base_key = obj["parts"]["base"]["pose_array_key"]
        link1_key = obj["parts"]["link1"]["pose_array_key"]
        assert base_key in result.pose_arrays
        assert link1_key in result.pose_arrays
        assert result.pose_arrays[link1_key].shape == (NUM_STEPS, 3, 4)

    def test_oscillating_object_is_detected(self):
        """An object that moves back to its start is still dynamic (adjacent-frame check)."""
        registry = ObjectInstanceRegistry()
        tracker = DynamicObjectTracker(registry, num_steps=NUM_STEPS)
        tracker._seen_assets.add(("RIGID", "box_a"))

        positions = [0.0, 0.5, 0.0, 0.5, 0.0]
        for step, x in enumerate(positions):
            pos = torch.tensor([x, 0.0, 0.0])
            quat = _identity_quat()
            rigid = {"box_a": _FakeRigidObject(pos, quat)}
            env = _FakeEnv(_FakeScene(rigid, {}))
            tracker.record_step_poses(env, step)

        result = tracker.get_dynamic_object_data(motion_eps=1e-4)
        assert len(result.objects_metadata) == 1

    def test_unseen_object_is_excluded(self):
        """An object that moved but was never seen in any camera should be excluded."""
        registry = ObjectInstanceRegistry()
        tracker = DynamicObjectTracker(registry, num_steps=NUM_STEPS)

        for step in range(NUM_STEPS):
            pos = torch.tensor([float(step), 0.0, 0.0])
            quat = _identity_quat()
            rigid = {"box_a": _FakeRigidObject(pos, quat)}
            env = _FakeEnv(_FakeScene(rigid, {}))
            tracker.record_step_poses(env, step)

        result = tracker.get_dynamic_object_data(motion_eps=1e-4)
        assert len(result.objects_metadata) == 0


class TestOutputFormat:
    def test_rigid_pose_array_shape(self):
        registry = ObjectInstanceRegistry()
        n = 3
        tracker = DynamicObjectTracker(registry, num_steps=n)
        tracker._seen_assets.add(("RIGID", "box_a"))

        for step in range(n):
            pos = torch.tensor([float(step), 0.0, 0.0])
            quat = _identity_quat()
            rigid = {"box_a": _FakeRigidObject(pos, quat)}
            env = _FakeEnv(_FakeScene(rigid, {}))
            tracker.record_step_poses(env, step)

        result = tracker.get_dynamic_object_data(motion_eps=1e-4)
        arr = list(result.pose_arrays.values())[0]
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.shape == (n, 3, 4)

    def test_metadata_fields(self):
        registry = ObjectInstanceRegistry()
        tracker = DynamicObjectTracker(registry, num_steps=10)
        result = tracker.get_dynamic_object_data(motion_eps=0.001)
        assert result.metadata["num_steps"] == 10
        assert result.metadata["motion_threshold"] == 0.001
        assert "coordinate_frame" in result.metadata
        assert "pose_format" in result.metadata

    def test_pose_values_correct(self):
        """Verify the stored translation is correct in the pose array."""
        registry = ObjectInstanceRegistry()
        n = 3
        tracker = DynamicObjectTracker(registry, num_steps=n)
        tracker._seen_assets.add(("RIGID", "box_a"))

        for step in range(n):
            pos = torch.tensor([float(step), 0.0, 0.0])
            quat = _identity_quat()
            rigid = {"box_a": _FakeRigidObject(pos, quat)}
            env = _FakeEnv(_FakeScene(rigid, {}))
            tracker.record_step_poses(env, step)

        result = tracker.get_dynamic_object_data(motion_eps=1e-4)
        arr = list(result.pose_arrays.values())[0]
        for step in range(n):
            np.testing.assert_allclose(arr[step, :, 3], [float(step), 0.0, 0.0], atol=1e-6)
            np.testing.assert_allclose(arr[step, :, :3], np.eye(3), atol=1e-6)


class TestWriterIntegration:
    def test_writes_json_and_npz(self):
        registry = ObjectInstanceRegistry()
        n = 3
        tracker = DynamicObjectTracker(registry, num_steps=n)
        tracker._seen_assets.add(("RIGID", "box_a"))

        for step in range(n):
            pos = torch.tensor([float(step), 0.0, 0.0])
            quat = _identity_quat()
            rigid = {"box_a": _FakeRigidObject(pos, quat)}
            env = _FakeEnv(_FakeScene(rigid, {}))
            tracker.record_step_poses(env, step)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = IsaacLabArenaWriter(tmpdir)
            result = tracker.get_dynamic_object_data(motion_eps=1e-4)
            writer.write_dynamic_object_poses(result)

            dyn_dir = os.path.join(tmpdir, "dynamic_objects")
            json_path = os.path.join(dyn_dir, "dynamic_objects.json")
            npz_path = os.path.join(dyn_dir, "dynamic_objects_poses.npz")
            assert os.path.isfile(json_path)
            assert os.path.isfile(npz_path)

            with open(json_path) as f:
                meta = json.load(f)

            assert "metadata" in meta
            assert "objects" in meta
            assert len(meta["objects"]) == 1

            obj = list(meta["objects"].values())[0]
            assert obj["type"] == "rigid"
            array_key = obj["pose_array_key"]

            poses = np.load(npz_path)
            assert array_key in poses
            assert poses[array_key].shape == (n, 3, 4)

    def test_writes_articulation_parts_as_separate_arrays(self):
        registry = ObjectInstanceRegistry()
        n = 3
        tracker = DynamicObjectTracker(registry, num_steps=n)
        tracker._seen_assets.add(("ARTICULATION", "robot"))

        for step in range(n):
            base_pos = torch.tensor([0.0, 0.0, 0.0])
            link_pos = torch.tensor([float(step) * 0.1, 0.0, 0.0])
            body_pos = torch.stack([base_pos, link_pos])
            body_quat = torch.stack([_identity_quat(), _identity_quat()])
            artic = {"robot": _FakeArticulation(body_pos, body_quat, ["base", "link1"])}
            env = _FakeEnv(_FakeScene({}, artic))
            tracker.record_step_poses(env, step)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = IsaacLabArenaWriter(tmpdir)
            result = tracker.get_dynamic_object_data(motion_eps=1e-4)
            writer.write_dynamic_object_poses(result)

            dyn_dir = os.path.join(tmpdir, "dynamic_objects")
            npz_path = os.path.join(dyn_dir, "dynamic_objects_poses.npz")
            poses = np.load(npz_path)

            with open(os.path.join(dyn_dir, "dynamic_objects.json")) as f:
                meta = json.load(f)

            obj = list(meta["objects"].values())[0]
            for part_name, part_meta in obj["parts"].items():
                key = part_meta["pose_array_key"]
                assert key in poses
                assert poses[key].shape == (n, 3, 4)


# ---------------------------------------------------------------------------
# Mesh sampling & reconstruction tests
# ---------------------------------------------------------------------------


class TestRotationFromNormal:
    def test_z_axis_is_normal(self):
        n = np.array([0.0, 0.0, 1.0])
        R = _rotation_from_normal(n)
        assert R.shape == (3, 3)
        np.testing.assert_allclose(R[:, 2], n, atol=1e-10)

    def test_arbitrary_normal(self):
        n = np.array([1.0, 1.0, 1.0])
        n = n / np.linalg.norm(n)
        R = _rotation_from_normal(n)
        np.testing.assert_allclose(R[:, 2], n, atol=1e-10)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_negative_normal(self):
        n = np.array([0.0, -1.0, 0.0])
        R = _rotation_from_normal(n)
        np.testing.assert_allclose(R[:, 2], n, atol=1e-10)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)

    def test_near_y_axis(self):
        """Normal close to y-axis triggers the alternate reference vector."""
        n = np.array([0.0, 0.99, 0.01])
        n = n / np.linalg.norm(n)
        R = _rotation_from_normal(n)
        np.testing.assert_allclose(R[:, 2], n, atol=1e-10)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)


class TestComputeRelativeSE3:
    def _identity_4x4(self):
        return np.eye(4, dtype=np.float64)

    def test_identity_object_pose(self):
        """When object pose is identity, relative SE(3) equals point SE(3)."""
        pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        T_obj = self._identity_4x4()

        rel = DynamicObjectTracker._compute_relative_se3(pts, normals, T_obj)
        assert rel.shape == (1, 3, 4)
        np.testing.assert_allclose(rel[0, :, 3], [1.0, 2.0, 3.0], atol=1e-5)
        np.testing.assert_allclose(rel[0, :, 2], [0.0, 0.0, 1.0], atol=1e-5)

    def test_translated_object(self):
        """Point at (3,0,0) with object at (1,0,0) => relative translation (2,0,0)."""
        pts = np.array([[3.0, 0.0, 0.0]], dtype=np.float32)
        normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        T_obj = self._identity_4x4()
        T_obj[0, 3] = 1.0  # object at x=1

        rel = DynamicObjectTracker._compute_relative_se3(pts, normals, T_obj)
        np.testing.assert_allclose(rel[0, :, 3], [2.0, 0.0, 0.0], atol=1e-5)

    def test_multiple_points(self):
        pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        T_obj = self._identity_4x4()

        rel = DynamicObjectTracker._compute_relative_se3(pts, normals, T_obj)
        assert rel.shape == (2, 3, 4)


class TestReconstructMeshPoints:
    def test_identity_roundtrip(self):
        """With identity object pose, reconstructed points equal originals."""
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        T_obj_0 = np.eye(4, dtype=np.float64)

        rel = DynamicObjectTracker._compute_relative_se3(pts, normals, T_obj_0)

        pose_3x4 = np.eye(4, dtype=np.float32)[:3, :]
        pose_arrays = {"obj": np.stack([pose_3x4])}
        mesh_samples = MeshSamplesResult(relative_se3_arrays={"obj": rel})

        result = reconstruct_mesh_points_at_step(mesh_samples, pose_arrays, 0)
        points_out, normals_out = result["obj"]
        np.testing.assert_allclose(points_out, pts, atol=1e-5)
        np.testing.assert_allclose(normals_out, normals, atol=1e-5)

    def test_translated_object_reconstruction(self):
        """Points should follow object translation."""
        pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        T_obj_0 = np.eye(4, dtype=np.float64)

        rel = DynamicObjectTracker._compute_relative_se3(pts, normals, T_obj_0)

        pose_step0 = np.eye(4, dtype=np.float32)[:3, :]
        pose_step1 = np.eye(4, dtype=np.float32)[:3, :]
        pose_step1[:, 3] = [5.0, 0.0, 0.0]  # object moved to (5,0,0)

        pose_arrays = {"obj": np.stack([pose_step0, pose_step1])}
        mesh_samples = MeshSamplesResult(relative_se3_arrays={"obj": rel})

        result = reconstruct_mesh_points_at_step(mesh_samples, pose_arrays, 1)
        points_out, _ = result["obj"]
        np.testing.assert_allclose(points_out, [[6.0, 0.0, 0.0]], atol=1e-5)

    def test_rotated_object_reconstruction(self):
        """Points and normals should rotate with the object."""
        pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        T_obj_0 = np.eye(4, dtype=np.float64)

        rel = DynamicObjectTracker._compute_relative_se3(pts, normals, T_obj_0)

        # 90-degree rotation about z-axis
        pose_step0 = np.eye(4, dtype=np.float32)[:3, :]
        R90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        pose_step1 = np.zeros((3, 4), dtype=np.float32)
        pose_step1[:3, :3] = R90
        pose_step1[:, 3] = [0.0, 0.0, 0.0]

        pose_arrays = {"obj": np.stack([pose_step0, pose_step1])}
        mesh_samples = MeshSamplesResult(relative_se3_arrays={"obj": rel})

        result = reconstruct_mesh_points_at_step(mesh_samples, pose_arrays, 1)
        points_out, normals_out = result["obj"]
        # (1,0,0) rotated 90deg about z => (0,1,0)
        np.testing.assert_allclose(points_out, [[0.0, 1.0, 0.0]], atol=1e-5)
        # normal (0,0,1) stays (0,0,1) under z-rotation
        np.testing.assert_allclose(normals_out, [[0.0, 0.0, 1.0]], atol=1e-5)

    def test_missing_key_skipped(self):
        """Keys present in mesh_samples but absent from pose_arrays are skipped."""
        rel = np.zeros((1, 3, 4), dtype=np.float32)
        mesh_samples = MeshSamplesResult(relative_se3_arrays={"missing_obj": rel})

        result = reconstruct_mesh_points_at_step(mesh_samples, {}, 0)
        assert len(result) == 0


class TestMeshWriterIntegration:
    def test_writes_and_loads_mesh_samples(self):
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        normals = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        T_obj_0 = np.eye(4, dtype=np.float64)
        rel = DynamicObjectTracker._compute_relative_se3(pts, normals, T_obj_0)

        mesh_samples = MeshSamplesResult(relative_se3_arrays={"my_obj": rel})

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = IsaacLabArenaWriter(tmpdir)
            writer.write_mesh_samples(mesh_samples)

            npz_path = os.path.join(tmpdir, "dynamic_objects", "dynamic_objects_mesh_samples.npz")
            assert os.path.isfile(npz_path)

            loaded = np.load(npz_path)
            assert "my_obj" in loaded
            assert loaded["my_obj"].shape == (2, 3, 4)

            loaded_samples = MeshSamplesResult(
                relative_se3_arrays={k: loaded[k] for k in loaded.files},
            )
            pose_3x4 = np.eye(4, dtype=np.float32)[:3, :]
            pose_arrays = {"my_obj": np.stack([pose_3x4])}
            result = reconstruct_mesh_points_at_step(loaded_samples, pose_arrays, 0)
            points_out, normals_out = result["my_obj"]
            np.testing.assert_allclose(points_out, pts, atol=1e-5)
            np.testing.assert_allclose(np.abs(normals_out), np.abs(normals), atol=1e-5)
