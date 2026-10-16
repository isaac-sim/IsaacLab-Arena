# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Write camera observations with parallel compression in Isaac Lab's native HDF5 format."""

from __future__ import annotations

import h5py
import numpy as np
import os
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import copy

from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler


class ParallelCameraHDF5DatasetFileHandler(HDF5DatasetFileHandler):
    """Compress camera chunks concurrently while keeping all HDF5 access on the caller thread.

    Images retain their resolution, dtype, values, and dataset paths. Files use the same gzip
    filter as Isaac Lab's writer and need no additional reader or compression plugin.
    """

    def __init__(self, compression_workers: int = 8):
        """Initialize the writer with a bounded number of compression workers.

        Args:
            compression_workers: Maximum number of CPU workers used during an episode export.
        """
        super().__init__()
        assert compression_workers > 0, "compression_workers must be positive"
        self._compression_workers = min(compression_workers, os.cpu_count() or 1)

    def write_episode(self, episode: EpisodeData, demo_id: int | None = None, dataset_compression: bool = True) -> None:
        """Write an episode, compressing its camera observations in parallel.

        Args:
            episode: Episode with tensor data prepared for export.
            demo_id: Explicit episode index, or None for the next sequential index.
            dataset_compression: Whether to compress datasets; False uses Isaac Lab's plain writer.
        """
        cameras = episode.data.get("camera_obs")
        if not dataset_compression or not isinstance(cameras, dict) or not cameras:
            super().write_episode(episode, demo_id, dataset_compression)
            return

        episode_index = self.get_num_episodes() if demo_id is None else demo_id
        self._raise_if_not_initialized()
        episode_name = f"demo_{episode_index}"
        if episode_name in self._hdf5_data_group:
            raise ValueError(f"Episode group '{episode_name}' already exists in the dataset")
        previous_count = self._demo_count
        previous_total = self._hdf5_data_group.attrs["total"]
        # Delegate metadata, naming, counters, and non-camera datasets to Isaac Lab. The shallow
        # copy avoids copying tensors or modifying the caller's episode, including on failure.
        metadata_episode = copy(episode)
        metadata_episode.data = {**episode.data, "camera_obs": {}}
        try:
            super().write_episode(metadata_episode, demo_id, dataset_compression)
            camera_group = self._hdf5_data_group[f"{episode_name}/camera_obs"]
            with ThreadPoolExecutor(max_workers=self._compression_workers) as executor:
                self._write_cameras(camera_group, cameras, executor)
        except Exception:
            # A failed export must not appear as a successful, partially filled demo.
            if episode_name in self._hdf5_data_group:
                del self._hdf5_data_group[episode_name]
            self._demo_count = previous_count
            self._hdf5_data_group.attrs["total"] = previous_total
            raise

    def _write_cameras(self, group: h5py.Group, cameras: dict, executor: ThreadPoolExecutor) -> None:
        """Write nested camera tensors, keeping at most two queued chunks per worker."""
        for name, tensor in cameras.items():
            if isinstance(tensor, dict):
                self._write_cameras(group.create_group(name), tensor, executor)
                continue

            values = tensor.cpu().numpy()
            if values.size == 0:
                group.create_dataset(name, data=values, compression="gzip", compression_opts=2)
                continue
            dataset = group.create_dataset(
                name, shape=values.shape, dtype=values.dtype, compression="gzip", compression_opts=2
            )
            pending = deque()
            for selection in dataset.iter_chunks():
                future = executor.submit(_compress_chunk, values, selection, dataset.chunks)
                pending.append((selection, future))
                if len(pending) >= 2 * self._compression_workers:
                    chunk_selection, compressed = pending.popleft()
                    dataset.id.write_direct_chunk(tuple(axis.start for axis in chunk_selection), compressed.result())
            while pending:
                chunk_selection, compressed = pending.popleft()
                dataset.id.write_direct_chunk(tuple(axis.start for axis in chunk_selection), compressed.result())


def _compress_chunk(values: np.ndarray, selection: tuple[slice, ...], chunk_shape: tuple[int, ...]) -> bytes:
    """Encode one gzip-filter chunk, including zero fill outside the dataset's boundary."""
    chunk = values[selection]
    if chunk.shape != chunk_shape:
        padded = np.zeros(chunk_shape, dtype=values.dtype)
        padded[tuple(slice(0, size) for size in chunk.shape)] = chunk
        chunk = padded
    # HDF5's gzip filter uses a zlib stream, not a gzip file header. Compression is the only
    # dataset filter, so these bytes can be written directly without running it a second time.
    return zlib.compress(chunk.tobytes(), level=2)
