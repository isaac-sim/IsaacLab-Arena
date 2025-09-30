# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import h5py
import numpy as np
import pathlib

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.utils.datasets import HDF5DatasetFileHandler


def compute_metrics(env: ManagerBasedRLEnv) -> dict[str, float]:
    """Computes the metrics registered in the environment.

    Args:
        env: The environment to compute the metrics for.

    Returns:
        A dictionary of metrics. Maps metric name to metric value.
    """
    assert hasattr(env.cfg, "metrics")
    # Get the path where the recorded data is stored
    dataset_path = get_metric_recorder_dataset_path(env)
    # For each registered metric
    metrics_data = {}
    for metric in env.cfg.metrics:
        # Load the recorded data from disk for this metric
        recorded_metric_data = get_recorded_metric_data(dataset_path, metric.recorder_term_name)
        # Compute the metric value from the recorded data
        metrics_data[metric.name] = metric.compute_metric_from_recording(recorded_metric_data)
    # Also add the number of episodes as a metric
    metrics_data["num_episodes"] = get_num_episodes(dataset_path)
    return metrics_data


def get_recorded_metric_data(dataset_path: pathlib.Path, recorder_term_name: str) -> list[np.ndarray]:
    """Gets the recorded metric data for a given metric name.

    Each metric records data to a dataset at a path. This function gets the recorded data
    for a given metric name.

    Args:
        dataset_path(pathlib.Path): The path to the dataset.
        recorder_term_name(str): The name of the recorder term to get the data for.

    Returns:
        A list of recorded metric data for each simulated episode.
    """
    recorded_metric_data_per_demo: list[np.ndarray] = []
    with h5py.File(dataset_path, "r") as f:
        demos = f["data"]
        for demo in demos:
            recorded_metric_data_per_demo.append(demos[demo][recorder_term_name][:])
    return recorded_metric_data_per_demo


def get_num_episodes(dataset_path: pathlib.Path) -> int:
    """Gets the number of episodes in the dataset.

    Args:
        dataset_path(pathlib.Path): The path to the dataset.

    Returns:
        The number of episodes in the dataset.
    """
    with h5py.File(dataset_path, "r") as f:
        return len(f["data"])


def get_metric_recorder_dataset_path(env: ManagerBasedRLEnv) -> pathlib.Path:
    """Gets the path to the dataset for the metric recorder.

    Args:
        env(ManagerBasedRLEnv): The environment to get the dataset path for.

    Returns:
        The path to the dataset for the metric recorder.
    """
    assert env.cfg.recorders.dataset_file_handler_class_type == HDF5DatasetFileHandler
    return pathlib.Path(env.cfg.recorders.dataset_export_dir_path) / pathlib.Path(
        env.cfg.recorders.dataset_filename + ".hdf5"
    )
