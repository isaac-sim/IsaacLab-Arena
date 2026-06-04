# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Self-contained HDF5 writer for the SyntheticScene dataset format.

This module merges two layers that used to live in ``nvblox_next``:

* the low-level h5py mechanics of ``SyntheticSceneDatasetWriter`` (group/dataset
  layout, pre-allocation, Zstandard compression), and
* the IsaacLab/PyTorch conversion layer of ``IsaacLabSyntheticDatasetWriter``
  (tensor -> numpy, ``TransformSE3`` -> ``(3, 4)`` extrinsic, RGBA segmentation
  -> integer ID map).

The resulting ``dataset.h5`` follows the exact same schema as the original
pipeline, so it remains loadable by ``nvblox_next``'s ``SyntheticSceneFlowDataset``
and ``SyntheticSceneRGBDDataset``. Crucially, this writer depends only on
``h5py`` + ``hdf5plugin`` (plus numpy/torch) -- no ``nvblox_next`` and no
``pytorch3d``.

HDF5 layout (one sequence per file)::

    sequence_000000/                     (group; attrs: sequence_id, num_frames, camera_ids, ...)
      cam0/                              (group; attrs: height, width)
        color            (N, H, W, 3)  uint8
        depth            (N, H, W)      float32
        intrinsic        (N, 3, 3)      float32
        extrinsic        (N, 3, 4)      float64
        normal           (N, H, W, 3)  float32
        semantic         (N, H, W)      int32
        semantic_json    (N,)           str
        flow2d           (N-1, H, W, 2) float32
        flow3d           (N-1, H, W, 3) float32
        flow3d_track_type(N-1, H, W)    uint8
      dynamic_objects/                   (group; attrs: metadata_json, object_ids)
        poses/<name>     (N, 3, 4)       float32
        mesh_samples/<name> (P, 3, 4)    float32
"""

from __future__ import annotations

import h5py
import json
import numpy as np
import os
import torch
from typing import TYPE_CHECKING, Any

import hdf5plugin

from isaaclab_arena_datagen.io import hdf5_keys as Keys

if TYPE_CHECKING:
    from isaaclab_arena_datagen.dynamic_object_tracker import DynamicObjectResult, MeshSamplesResult
    from isaaclab_arena_datagen.geometry.transform_se3 import TransformSE3

# Zstandard level 3 -- good speed/ratio balance for image and float data.
_ZSTD: dict[str, Any] = hdf5plugin.Zstd(clevel=3)
_STR_DTYPE = h5py.string_dtype()


def camera_id_from_index(index: int) -> str:
    r"""Return the camera identifier for a given index (e.g. 0 -> ``"cam0"``)."""
    return f"cam{index}"


def _rgba_to_semantic_ids(
    seg_rgba_hw4: np.ndarray,
    semantic_info: list[dict[str, Any]],
) -> np.ndarray:
    """Convert an RGBA segmentation image to an int32 ID map.

    Each object in *semantic_info* is assigned a 1-based integer ID in list
    order. Pixels whose RGBA value does not match any object receive ID 0
    (background).

    Args:
        seg_rgba_hw4: ``(H, W, 4)`` uint8 RGBA segmentation image.
        semantic_info: Per-object metadata dicts, each containing an
            ``"rgba"`` (or ``"color"``) key with an ``[R, G, B, A]`` list.

    Returns:
        ``(H, W)`` int32 array of per-pixel semantic IDs.
    """
    H, W, _ = seg_rgba_hw4.shape
    semantic_ids = np.zeros((H, W), dtype=np.int32)
    for obj_idx, obj in enumerate(semantic_info, start=1):
        rgba = obj.get("rgba") or obj.get("color")
        if rgba is None:
            continue
        color = np.array(rgba, dtype=np.uint8)
        mask = np.all(seg_rgba_hw4 == color, axis=-1)
        semantic_ids[mask] = obj_idx
    return semantic_ids


class DatagenHDF5Writer:
    """Writes one SyntheticScene sequence into ``{output_dir}/dataset.h5``.

    All per-frame datasets are pre-allocated at construction based on
    *num_frames* and the per-camera ``(camera_id, height, width)`` specs. The
    public ``write_*`` methods accept IsaacLab/PyTorch data and persist it in
    the schema described in the module docstring.

    Args:
        output_dir: Root directory; the HDF5 file is written as
            ``{output_dir}/dataset.h5``.
        sequence_index: Integer index for the sequence group (typically ``0``).
        cameras: List of ``(camera_id, height, width)`` tuples, one per camera.
        num_frames: Total number of simulation frames to pre-allocate (> 1).
        anchor_frame_indices: Optional anchor (source) frames for long-range
            flow tracking. Empty by default (adjacent-frame flow only).
    """

    def __init__(
        self,
        output_dir: str,
        sequence_index: int,
        cameras: list[tuple[str, int, int]],
        num_frames: int,
        anchor_frame_indices: list[int] | None = None,
        filename: str = "dataset.h5",
    ) -> None:
        """Open the HDF5 file and pre-allocate all datasets.

        *num_frames* is the maximum capacity; datasets are created resizable so
        :meth:`trim` can shrink them to the actual frame count (used when the
        sequence length -- e.g. an episode -- is only known at the end).
        """
        assert num_frames > 1, f"num_frames must be > 1 to store adjacent-frame flow, got {num_frames}"
        heights = {h for _, h, _ in cameras}
        widths = {w for _, _, w in cameras}
        if len(heights) > 1 or len(widths) > 1:
            raise ValueError(
                f"All cameras must share the same spatial dimensions. Got heights={heights}, widths={widths}."
            )

        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir
        self._anchors = sorted(anchor_frame_indices or [])
        self._num_frames = num_frames

        self._file = h5py.File(os.path.join(output_dir, filename), "w")

        sequence_id = Keys.sequence_group_name(sequence_index)
        self._seq_group = self._file.require_group(sequence_id)
        self._seq_group.attrs[Keys.ATTR_SEQUENCE_ID] = sequence_id
        self._seq_group.attrs[Keys.ATTR_NUM_FRAMES] = num_frames
        self._seq_group.attrs[Keys.ATTR_ANCHOR_FRAME_INDICES] = json.dumps(self._anchors)
        self._seq_group.attrs[Keys.ATTR_CAMERA_IDS] = json.dumps([cid for cid, _, _ in cameras])

        self._cam_groups: dict[str, h5py.Group] = {}
        for cam_id, height, width in cameras:
            cam_group = self._seq_group.require_group(cam_id)
            cam_group.attrs[Keys.ATTR_HEIGHT] = height
            cam_group.attrs[Keys.ATTR_WIDTH] = width
            self._cam_groups[cam_id] = cam_group
            self._preallocate_camera(cam_id, height, width, num_frames)

    # ------------------------------------------------------------------
    # Pre-allocation
    # ------------------------------------------------------------------

    def _preallocate_camera(self, camera_id: str, H: int, W: int, n: int) -> None:
        """Pre-allocate all per-frame datasets for one camera.

        Datasets are resizable along the frame axis (``maxshape`` set) so
        :meth:`trim` can shrink them to the actual frame count at close.
        """
        g = self._cam_groups[camera_id]

        def make(name: str, shape: tuple[int, ...], dtype: Any) -> None:
            g.create_dataset(name, shape=shape, dtype=dtype, chunks=(1,) + shape[1:], maxshape=shape, **_ZSTD)

        make(Keys.COLOR, (n, H, W, 3), np.uint8)
        make(Keys.DEPTH, (n, H, W), np.float32)
        make(Keys.INTRINSIC, (n, 3, 3), np.float32)
        make(Keys.EXTRINSIC, (n, 3, 4), np.float64)
        make(Keys.NORMAL, (n, H, W, 3), np.float32)
        make(Keys.SEMANTIC, (n, H, W), np.int32)
        g.create_dataset(Keys.SEMANTIC_JSON, shape=(n,), dtype=_STR_DTYPE, maxshape=(n,), **_ZSTD)

        make(Keys.FLOW2D, (n - 1, H, W, 2), np.float32)
        make(Keys.FLOW3D, (n - 1, H, W, 3), np.float32)
        make(Keys.FLOW3D_TRACK_TYPE, (n - 1, H, W), np.uint8)

        for anchor_idx in self._anchors:
            rows = n - anchor_idx
            if rows <= 0:
                continue
            anchor_key = str(anchor_idx)
            for group_name, suffix, dtype in [
                (Keys.FLOW3D_FROM_FRAME, (H, W, 3), np.float32),
                (Keys.TRACKABLE_MASK_FRAME, (H, W), np.uint8),
                (Keys.IN_FRAME_MASK_FRAME, (H, W), np.uint8),
                (Keys.VISIBLE_NOW_MASK_FRAME, (H, W), np.uint8),
            ]:
                sub = g.require_group(group_name)
                sub.create_dataset(
                    anchor_key,
                    shape=(rows,) + suffix,
                    dtype=dtype,
                    chunks=(1,) + suffix,
                    maxshape=(rows,) + suffix,
                    **_ZSTD,
                )

    def trim(self, num_valid_frames: int) -> None:
        """Shrink all per-frame datasets to *num_valid_frames* actual frames.

        Used when the sequence length is only known after recording (e.g. an
        episode that terminates early). Per-frame datasets are resized to
        ``num_valid_frames`` and adjacent-flow datasets to
        ``num_valid_frames - 1``. Updates the ``num_frames`` attribute.
        """
        assert (
            0 < num_valid_frames <= self._num_frames
        ), f"num_valid_frames={num_valid_frames} out of range (1, {self._num_frames}]"
        per_frame = (Keys.COLOR, Keys.DEPTH, Keys.INTRINSIC, Keys.EXTRINSIC, Keys.NORMAL, Keys.SEMANTIC)
        flow = (Keys.FLOW2D, Keys.FLOW3D, Keys.FLOW3D_TRACK_TYPE)
        for g in self._cam_groups.values():
            for key in per_frame:
                g[key].resize(num_valid_frames, axis=0)
            g[Keys.SEMANTIC_JSON].resize((num_valid_frames,))
            for key in flow:
                g[key].resize(max(num_valid_frames - 1, 0), axis=0)
        self._num_frames = num_valid_frames
        self._seq_group.attrs[Keys.ATTR_NUM_FRAMES] = num_valid_frames

    # ------------------------------------------------------------------
    # Core per-frame outputs
    # ------------------------------------------------------------------

    def write_rgb(self, rgb_hw3: torch.Tensor, camera_id: str, frame_index: int) -> None:
        """Save an ``(H, W, 3)`` uint8 RGB image."""
        self._cam_groups[camera_id][Keys.COLOR][frame_index] = rgb_hw3.cpu().numpy()

    def write_depth(self, depth_hw: torch.Tensor, camera_id: str, frame_index: int) -> None:
        """Save an ``(H, W)`` float32 depth map (metres)."""
        self._cam_groups[camera_id][Keys.DEPTH][frame_index] = depth_hw.cpu().numpy().astype(np.float32)

    def write_intrinsics(self, intrinsics_33: torch.Tensor, camera_id: str, frame_index: int) -> None:
        """Save the ``(3, 3)`` float32 intrinsic matrix."""
        self._cam_groups[camera_id][Keys.INTRINSIC][frame_index] = intrinsics_33.cpu().numpy().astype(np.float32)

    def write_extrinsics(self, T_W_from_C: TransformSE3, camera_id: str, frame_index: int) -> None:
        """Save the camera-to-world transform as a ``(3, 4)`` float64 ``[R|t]`` matrix."""
        R = T_W_from_C.rotation.R  # (B, N, 3, 3)
        t = T_W_from_C.translation.t  # (B, N, 3)
        extrinsic_34 = torch.cat([R[0, 0], t[0, 0].unsqueeze(-1)], dim=-1).double().cpu().numpy()
        self._cam_groups[camera_id][Keys.EXTRINSIC][frame_index] = extrinsic_34

    def write_normals(self, normals_hw3: torch.Tensor, camera_id: str, frame_index: int) -> None:
        """Save an ``(H, W, 3)`` float32 world-space surface-normal map."""
        self._cam_groups[camera_id][Keys.NORMAL][frame_index] = normals_hw3.cpu().numpy().astype(np.float32)

    def write_optical_flow(self, flow_hw2: torch.Tensor, camera_id: str, frame_index: int) -> None:
        """Save an ``(H, W, 2)`` float32 optical-flow map (dx, dy in pixels)."""
        self._cam_groups[camera_id][Keys.FLOW2D][frame_index] = flow_hw2.cpu().numpy().astype(np.float32)

    def write_scene_flow_3d(self, flow3d_hw3: torch.Tensor, camera_id: str, frame_index: int) -> None:
        """Save an ``(H, W, 3)`` float32 3-D scene-flow map (metres)."""
        self._cam_groups[camera_id][Keys.FLOW3D][frame_index] = flow3d_hw3.cpu().numpy().astype(np.float32)

    def write_scene_flow_track_type(self, track_type_hw: torch.Tensor, camera_id: str, frame_index: int) -> None:
        """Save an ``(H, W)`` uint8 per-pixel tracking-type map."""
        self._cam_groups[camera_id][Keys.FLOW3D_TRACK_TYPE][frame_index] = track_type_hw.cpu().numpy()

    def write_semantic_segmentation(
        self,
        seg_rgba_hw4: torch.Tensor,
        semantic_info: list[dict[str, Any]],
        camera_id: str,
        frame_index: int,
    ) -> None:
        """Save semantic segmentation: an int32 ID map plus a per-frame JSON blob.

        The RGBA segmentation image is converted to an int32 ID map (1-based,
        0 = background) and the per-object metadata is stored as a JSON string.
        """
        semantic_ids = _rgba_to_semantic_ids(seg_rgba_hw4.cpu().numpy(), semantic_info)
        self._cam_groups[camera_id][Keys.SEMANTIC][frame_index] = semantic_ids
        self._cam_groups[camera_id][Keys.SEMANTIC_JSON][frame_index] = json.dumps({"objects": semantic_info})

    # ------------------------------------------------------------------
    # Dynamic objects
    # ------------------------------------------------------------------

    def write_dynamic_object_poses(self, result: DynamicObjectResult) -> None:
        """Write per-object world-frame pose trajectories and metadata."""
        poses_group = self._seq_group.require_group(f"{Keys.DYNAMIC_OBJECTS}/{Keys.POSES}")
        for key, poses in result.T_W_from_localbody_arrays.items():
            poses_group.create_dataset(key, data=poses, **_ZSTD)

        dynamic_group = self._seq_group.require_group(Keys.DYNAMIC_OBJECTS)
        dynamic_group.attrs[Keys.ATTR_METADATA_JSON] = json.dumps(
            {"metadata": result.metadata, "objects": result.objects_metadata}
        )
        object_ids = {name: idx for idx, name in enumerate(result.objects_metadata.keys(), start=1)}
        encoded = json.dumps(object_ids)
        self._seq_group.attrs[Keys.ATTR_OBJECT_IDS] = encoded
        dynamic_group.attrs[Keys.ATTR_OBJECT_IDS] = encoded

    def write_mesh_samples(self, mesh_result: MeshSamplesResult) -> None:
        """Write sampled mesh surface points (relative SE(3)) per object."""
        mesh_group = self._seq_group.require_group(f"{Keys.DYNAMIC_OBJECTS}/{Keys.MESH_SAMPLES}")
        for key, points in mesh_result.se3_localbody_from_point_arrays.items():
            mesh_group.create_dataset(key, data=points, **_ZSTD)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the HDF5 file."""
        self._file.close()

    def __enter__(self) -> DatagenHDF5Writer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
