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
from isaaclab_arena_datagen.io.hdf5_writer import DatagenHDF5Writer

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
