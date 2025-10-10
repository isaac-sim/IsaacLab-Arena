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

import shutil

import pandas as pd

from isaaclab_arena.policy.config.dataset_config import Gr00tDatasetConfig
from isaaclab_arena.policy.data_utils.convert_hdf5_to_lerobot import convert_hdf5_to_lerobot
from isaaclab_arena.policy.data_utils.io_utils import create_config_from_yaml
from isaaclab_arena.tests.utils.constants import TestConstants


def test_g1_convert_hdf5_to_lerobot():
    # Load expected data for comparison
    expected_g1_parquet = pd.read_parquet(
        TestConstants.test_data_dir + "/test_g1_locomanip_lerobot/data/chunk-000/episode_000000.parquet"
    )
    g1_ds_config = create_config_from_yaml(
        TestConstants.test_data_dir + "/test_g1_locomanip_lerobot/test_g1_locomanip_config.yaml", Gr00tDatasetConfig
    )

    # Clean up any existing output directory
    if g1_ds_config.lerobot_data_dir.exists():

        shutil.rmtree(g1_ds_config.lerobot_data_dir)

    # Run conversion
    convert_hdf5_to_lerobot(g1_ds_config)

    # assert it has episodes.jsonl file
    assert (g1_ds_config.lerobot_data_dir / "meta" / "episodes.jsonl").exists()

    # assert it has tasks.jsonl file
    assert (g1_ds_config.lerobot_data_dir / "meta" / "tasks.jsonl").exists()

    # assert it has info.json file
    assert (g1_ds_config.lerobot_data_dir / "meta" / "info.json").exists()

    # assert it has modality.json file
    assert (g1_ds_config.lerobot_data_dir / "meta" / "modality.json").exists()

    # assert it has data/ folder has parquet files
    parquet_files = list((g1_ds_config.lerobot_data_dir / "data").glob("**/*.parquet"))
    assert len(parquet_files) == 1

    # assert it has videos/ folder has mp4 files
    mp4_files = list((g1_ds_config.lerobot_data_dir / "videos").glob("**/*.mp4"))
    assert len(mp4_files) == 1
    # check parquet file contains expected columns
    actual_df = pd.read_parquet(parquet_files[0])
    expected_columns = set(expected_g1_parquet.columns)
    actual_columns = set(actual_df.columns)
    assert expected_columns.issubset(actual_columns), f"Missing columns: {expected_columns - actual_columns}"
    # check parquet file data is the same as expected
    assert actual_df.equals(expected_g1_parquet)

    # remove lerobot_data_dir
    shutil.rmtree(g1_ds_config.lerobot_data_dir.parent)


if __name__ == "__main__":
    test_g1_convert_hdf5_to_lerobot()
