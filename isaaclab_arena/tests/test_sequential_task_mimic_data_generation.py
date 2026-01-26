# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess

HEADLESS = True
ENABLE_CAMERAS = True
GENERATION_NUM_TRIALS = 1

def test_franka_put_and_close_door_mimic_data_generation_single_env():
    """Test mimic data generation for franka_put_and_close_door sequential task on a single env."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "dataset_generated.hdf5")

        args = [TestConstants.python_path, f"{TestConstants.scripts_dir}/imitation_learning/generate_dataset.py"]
        args.append("--device")
        args.append("cpu")
        args.append("--generation_num_trials")
        args.append(str(GENERATION_NUM_TRIALS))
        args.append("--num_envs")
        args.append("1")
        args.append("--input_file")
        args.append(TestConstants.test_data_dir + "/test_sequential_task_mimic_data_generation.hdf5")
        args.append("--output_file")
        args.append(output_file)
        if HEADLESS:
            args.append("--headless")
        if ENABLE_CAMERAS:
            args.append("--enable_cameras")
        args.append("--mimic")
        # example env
        args.append("franka_put_and_close_door")
        args.append("--embodiment")
        args.append("franka")
        run_subprocess(args)


def test_franka_put_and_close_door_mimic_data_generation_multi_env():
    """Test mimic data generation for franka_put_and_close_door sequential task on multiple envs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "dataset_generated.hdf5")

        args = [TestConstants.python_path, f"{TestConstants.scripts_dir}/imitation_learning/generate_dataset.py"]
        args.append("--device")
        args.append("cpu")
        args.append("--generation_num_trials")
        args.append(str(GENERATION_NUM_TRIALS))
        args.append("--num_envs")
        args.append("4")
        args.append("--input_file")
        args.append(TestConstants.test_data_dir + "/test_sequential_task_mimic_data_generation.hdf5")
        args.append("--output_file")
        args.append(output_file)
        if HEADLESS:
            args.append("--headless")
        if ENABLE_CAMERAS:
            args.append("--enable_cameras")
        args.append("--mimic")
        # example env
        args.append("franka_put_and_close_door")
        args.append("--embodiment")
        args.append("franka")
        run_subprocess(args)


if __name__ == "__main__":
    test_franka_put_and_close_door_mimic_data_generation_single_env()
    test_franka_put_and_close_door_mimic_data_generation_multi_env()
