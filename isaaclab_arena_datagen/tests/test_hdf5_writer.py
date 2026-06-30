# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Schema round-trip test for the self-contained SyntheticScene HDF5 writer.

Guards against format drift from the schema documented in
``isaaclab_arena_datagen/README.md`` (and consumed by nvblox_next's loaders).
Requires h5py + hdf5plugin but **not** Isaac Sim.
"""

import json
import numpy as np
import os
import torch
import types

import pytest

from isaaclab_arena_datagen.geometry.transform_se3 import TransformSE3
from isaaclab_arena_datagen.io import hdf5_keys as Keys
from isaaclab_arena_datagen.io.hdf5_writer import (
    DatagenHDF5Writer,
    StoredFloatType,
    episode_output_dir,
    read_float32,
    stored_float_type,
)

H, W, N = 4, 6, 3
CAM = "cam0"


@pytest.fixture()
def written_dataset(tmp_path):
    """Write a tiny single-camera dataset and return its file path."""
    writer = DatagenHDF5Writer(str(tmp_path), sequence_index=0, cameras=[(CAM, H, W)], num_frames=N)
    T = TransformSE3.from_matrix(torch.eye(4))
    for i in range(N):
        writer.write_rgb(torch.zeros(H, W, 3, dtype=torch.uint8), CAM, i)
        writer.write_depth(torch.ones(H, W), CAM, i)
        writer.write_intrinsics(torch.eye(3), CAM, i)
        writer.write_extrinsics(T, CAM, i)
        writer.write_normals(torch.zeros(H, W, 3), CAM, i)
        writer.write_semantic_segmentation(
            torch.zeros(H, W, 4, dtype=torch.uint8),
            [{"name": "obj", "rgba": [10, 20, 30, 255]}],
            CAM,
            i,
        )
    for i in range(N - 1):
        writer.write_optical_flow(torch.zeros(H, W, 2), CAM, i)
        writer.write_scene_flow_3d(torch.zeros(H, W, 3), CAM, i)
        writer.write_scene_flow_track_type(torch.zeros(H, W, dtype=torch.uint8), CAM, i)

    writer.write_dynamic_object_poses(
        types.SimpleNamespace(
            T_W_from_localbody_arrays={"cracker_box": np.zeros((N, 3, 4), np.float32)},
            metadata={"foo": 1},
            objects_metadata={"cracker_box": {"type": "rigid"}},
        )
    )
    writer.write_mesh_samples(
        types.SimpleNamespace(se3_localbody_from_point_arrays={"cracker_box": np.zeros((5, 3, 4), np.float32)})
    )
    writer.close()
    return os.path.join(str(tmp_path), "dataset.h5")


def test_camera_dataset_shapes_and_dtypes(written_dataset):
    h5py = pytest.importorskip("h5py")
    expected = {
        Keys.COLOR: ((N, H, W, 3), np.uint8),
        Keys.DEPTH: ((N, H, W), np.float32),
        Keys.INTRINSIC: ((N, 3, 3), np.float32),
        Keys.EXTRINSIC: ((N, 3, 4), np.float64),
        Keys.NORMAL: ((N, H, W, 3), np.float32),
        Keys.SEMANTIC: ((N, H, W), np.int32),
        Keys.FLOW2D: ((N - 1, H, W, 2), np.float32),
        Keys.FLOW3D: ((N - 1, H, W, 3), np.float32),
        Keys.FLOW3D_TRACK_TYPE: ((N - 1, H, W), np.uint8),
    }
    with h5py.File(written_dataset, "r") as f:
        g = f["sequence_000000"][CAM]
        for key, (shape, dtype) in expected.items():
            assert g[key].shape == shape, f"{key} shape {g[key].shape} != {shape}"
            assert g[key].dtype == dtype, f"{key} dtype {g[key].dtype} != {dtype}"
        assert g[Keys.SEMANTIC_JSON].shape == (N,)


def test_sequence_and_camera_attrs(written_dataset):
    h5py = pytest.importorskip("h5py")
    with h5py.File(written_dataset, "r") as f:
        seq = f["sequence_000000"]
        assert seq.attrs[Keys.ATTR_NUM_FRAMES] == N
        assert seq.attrs[Keys.ATTR_SEQUENCE_ID] == "sequence_000000"
        assert json.loads(seq.attrs[Keys.ATTR_CAMERA_IDS]) == [CAM]
        g = seq[CAM]
        assert g.attrs[Keys.ATTR_HEIGHT] == H and g.attrs[Keys.ATTR_WIDTH] == W


def test_semantic_json_roundtrip(written_dataset):
    h5py = pytest.importorskip("h5py")
    with h5py.File(written_dataset, "r") as f:
        blob = f["sequence_000000"][CAM][Keys.SEMANTIC_JSON][0]
        assert json.loads(blob)["objects"][0]["name"] == "obj"


def test_dynamic_objects_group(written_dataset):
    h5py = pytest.importorskip("h5py")
    with h5py.File(written_dataset, "r") as f:
        dgrp = f["sequence_000000"][Keys.DYNAMIC_OBJECTS]
        assert dgrp[Keys.POSES]["cracker_box"].shape == (N, 3, 4)
        assert dgrp[Keys.MESH_SAMPLES]["cracker_box"].shape == (5, 3, 4)
        assert json.loads(dgrp.attrs[Keys.ATTR_OBJECT_IDS]) == {"cracker_box": 1}
        assert json.loads(dgrp.attrs[Keys.ATTR_METADATA_JSON])["metadata"] == {"foo": 1}


def test_requires_more_than_one_frame(tmp_path):
    with pytest.raises(AssertionError):
        DatagenHDF5Writer(str(tmp_path), sequence_index=0, cameras=[(CAM, H, W)], num_frames=1)


def test_episode_output_dir_uses_nested_layout(tmp_path):
    assert episode_output_dir(str(tmp_path), 7) == os.path.join(str(tmp_path), "episode_0007")


def test_store_flags_off_omit_optional_datasets(tmp_path):
    """Normals and 3D flow are absent when their store flags are off; flow2d stays."""
    h5py = pytest.importorskip("h5py")
    writer = DatagenHDF5Writer(
        str(tmp_path), sequence_index=0, cameras=[(CAM, H, W)], num_frames=N, store_normals=False, store_flow3d=False
    )
    # The corresponding write_* calls are no-ops with the flags off.
    writer.write_normals(torch.zeros(H, W, 3), CAM, 0)
    writer.write_scene_flow_3d(torch.zeros(H, W, 3), CAM, 0)
    writer.close()
    with h5py.File(os.path.join(str(tmp_path), "dataset.h5"), "r") as f:
        g = f["sequence_000000"][CAM]
        assert Keys.NORMAL not in g
        assert Keys.FLOW3D not in g
        assert Keys.FLOW3D_TRACK_TYPE not in g
        assert Keys.FLOW2D in g


def test_float16_storage_and_dtype_agnostic_read(tmp_path):
    """depth/flow2d store as float16 when requested, and read back as float32 either way."""
    h5py = pytest.importorskip("h5py")
    writer = DatagenHDF5Writer(
        str(tmp_path), sequence_index=0, cameras=[(CAM, H, W)], num_frames=N, store_float_type=StoredFloatType.FLOAT16
    )
    writer.write_depth(torch.ones(H, W), CAM, 0)
    writer.write_optical_flow(torch.ones(H, W, 2), CAM, 0)
    writer.close()
    with h5py.File(os.path.join(str(tmp_path), "dataset.h5"), "r") as f:
        g = f["sequence_000000"][CAM]
        assert g[Keys.DEPTH].dtype == np.float16
        assert g[Keys.FLOW2D].dtype == np.float16
        assert stored_float_type(g[Keys.DEPTH]) == StoredFloatType.FLOAT16
        depth = read_float32(g[Keys.DEPTH], 0)
        assert depth.dtype == np.float32
        assert np.allclose(depth, 1.0)


def test_default_float32_read_back(written_dataset):
    """The decode helper reads a legacy float32 depth dataset back as float32."""
    h5py = pytest.importorskip("h5py")
    with h5py.File(written_dataset, "r") as f:
        g = f["sequence_000000"][CAM]
        assert g[Keys.DEPTH].dtype == np.float32
        assert stored_float_type(g[Keys.DEPTH]) == StoredFloatType.FLOAT32
        assert read_float32(g[Keys.DEPTH]).dtype == np.float32


def test_downsample_resolution_shapes_attrs_and_intrinsics(tmp_path):
    """Stored modalities, camera attrs, and intrinsics all follow the downsampled resolution."""
    h5py = pytest.importorskip("h5py")
    render_h, render_w = 8, 12
    target = (6, 4)  # (width, height) = half of render
    writer = DatagenHDF5Writer(
        str(tmp_path),
        sequence_index=0,
        cameras=[(CAM, render_h, render_w)],
        num_frames=N,
        color_resolution=target,
        depth_resolution=target,
        flow2d_resolution=target,
        semantic_resolution=target,
    )
    K = torch.tensor([[100.0, 0.0, 6.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]])
    writer.write_rgb(torch.zeros(render_h, render_w, 3, dtype=torch.uint8), CAM, 0)
    writer.write_depth(torch.ones(render_h, render_w), CAM, 0)
    writer.write_intrinsics(K, CAM, 0)
    writer.write_optical_flow(torch.zeros(render_h, render_w, 2), CAM, 0)
    writer.write_semantic_segmentation(torch.zeros(render_h, render_w, 4, dtype=torch.uint8), [], CAM, 0)
    writer.close()
    with h5py.File(os.path.join(str(tmp_path), "dataset.h5"), "r") as f:
        g = f["sequence_000000"][CAM]
        assert g[Keys.COLOR].shape == (N, 4, 6, 3)
        assert g[Keys.DEPTH].shape == (N, 4, 6)
        assert g[Keys.FLOW2D].shape == (N - 1, 4, 6, 2)
        assert g[Keys.SEMANTIC].shape == (N, 4, 6)
        assert g.attrs[Keys.ATTR_HEIGHT] == 4 and g.attrs[Keys.ATTR_WIDTH] == 6
        # Intrinsics scaled by 0.5 in both axes.
        assert np.allclose(g[Keys.INTRINSIC][0], [[50.0, 0.0, 3.0], [0.0, 50.0, 2.0], [0.0, 0.0, 1.0]])


def test_target_resolution_must_not_exceed_render(tmp_path):
    with pytest.raises(AssertionError):
        DatagenHDF5Writer(
            str(tmp_path), sequence_index=0, cameras=[(CAM, H, W)], num_frames=N, color_resolution=(W + 2, H)
        )


def test_color_and_depth_resolution_must_match(tmp_path):
    with pytest.raises(AssertionError):
        DatagenHDF5Writer(
            str(tmp_path),
            sequence_index=0,
            cameras=[(CAM, H, W)],
            num_frames=N,
            color_resolution=(W, H),
            depth_resolution=(W - 2, H),
        )


def test_trim_shrinks_datasets_to_actual_length(tmp_path):
    """Writer pre-allocated to a max capacity is trimmed to the actual frame count."""
    h5py = pytest.importorskip("h5py")
    capacity, actual = 10, 4
    episode_dir = episode_output_dir(str(tmp_path), 0)
    writer = DatagenHDF5Writer(episode_dir, sequence_index=0, cameras=[(CAM, H, W)], num_frames=capacity)
    T = TransformSE3.from_matrix(torch.eye(4))
    for i in range(actual):
        writer.write_rgb(torch.zeros(H, W, 3, dtype=torch.uint8), CAM, i)
        writer.write_depth(torch.ones(H, W), CAM, i)
        writer.write_intrinsics(torch.eye(3), CAM, i)
        writer.write_extrinsics(T, CAM, i)
        writer.write_normals(torch.zeros(H, W, 3), CAM, i)
        writer.write_semantic_segmentation(torch.zeros(H, W, 4, dtype=torch.uint8), [], CAM, i)
    for i in range(actual - 1):
        writer.write_optical_flow(torch.zeros(H, W, 2), CAM, i)
        writer.write_scene_flow_3d(torch.zeros(H, W, 3), CAM, i)
        writer.write_scene_flow_track_type(torch.zeros(H, W, dtype=torch.uint8), CAM, i)
    writer.trim(actual)
    writer.close()

    with h5py.File(os.path.join(episode_dir, "dataset.h5"), "r") as f:
        seq = f["sequence_000000"]
        assert seq.attrs[Keys.ATTR_NUM_FRAMES] == actual
        g = seq[CAM]
        assert g[Keys.COLOR].shape == (actual, H, W, 3)
        assert g[Keys.SEMANTIC_JSON].shape == (actual,)
        assert g[Keys.FLOW2D].shape == (actual - 1, H, W, 2)
        assert g[Keys.FLOW3D_TRACK_TYPE].shape == (actual - 1, H, W)
