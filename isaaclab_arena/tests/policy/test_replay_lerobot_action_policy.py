# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess

HEADLESS = True
ENABLE_CAMERAS = True
NUM_STEPS = 10
# Only 1 traj in test data
TRAJECTORY_INDEX = 0


def test_g1_locomanip_replay_lerobot_policy_runner_single_env():
    args = [TestConstants.python_path, f"{TestConstants.examples_dir}/policy_runner.py"]
    args.append("--policy_type")
    args.append("replay_lerobot")
    args.append("--config_yaml_path")
    args.append(TestConstants.test_data_dir + "/test_g1_locomanip_lerobot/test_g1_locomanip_replay_action_config.yaml")
    args.append("--max_steps")
    args.append(str(NUM_STEPS))
    args.append("--trajectory_index")
    args.append(str(TRAJECTORY_INDEX))
    if HEADLESS:
        args.append("--headless")
    if ENABLE_CAMERAS:
        args.append("--enable_cameras")
    # example env
    args.append("galileo_g1_locomanip_pick_and_place")
    args.append("--object")
    args.append("brown_box")
    args.append("--embodiment")
    args.append("g1_wbc_joint")
    run_subprocess(args)


def test_gr1_manip_replay_lerobot_policy_runner_single_env():
    args = [TestConstants.python_path, f"{TestConstants.examples_dir}/policy_runner.py"]
    args.append("--policy_type")
    args.append("replay_lerobot")
    args.append("--config_yaml_path")
    args.append(TestConstants.test_data_dir + "/test_gr1_manip_lerobot/test_gr1_manip_replay_action_config.yaml")
    args.append("--max_steps")
    args.append(str(NUM_STEPS))
    args.append("--trajectory_index")
    args.append(str(TRAJECTORY_INDEX))
    if HEADLESS:
        args.append("--headless")
    if ENABLE_CAMERAS:
        args.append("--enable_cameras")
    # example env
    args.append("gr1_open_microwave")
    args.append("--object")
    args.append("microwave")
    args.append("--embodiment")
    args.append("gr1_joint")
    run_subprocess(args)


if __name__ == "__main__":
    test_g1_locomanip_replay_lerobot_policy_runner_single_env()
