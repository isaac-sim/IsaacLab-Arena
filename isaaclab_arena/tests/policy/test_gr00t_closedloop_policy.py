# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess

HEADLESS = True
ENABLE_CAMERAS = True
NUM_STEPS = 10
NUM_ENVS = 3


def test_g1_locomanip_gr00t_closedloop_policy_runner_single_env():
    args = [TestConstants.python_path, f"{TestConstants.examples_dir}/policy_runner.py"]
    args.append("--policy_type")
    args.append("gr00t_closedloop")
    args.append("--policy_config_yaml_path")
    args.append(
        TestConstants.test_data_dir + "/test_g1_locomanip_lerobot/test_g1_locomanip_gr00t_closedloop_config.yaml"
    )
    args.append("--num_steps")
    args.append(str(NUM_STEPS))
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


# NOTE(alexmillane, 2025-10-31): We've have to disable multi-env evaluation for now as it's not supported by the policy runner.
# TODO(alexmillane, 2025-10-31): Un-skip this test once the policy runner supports multi-env evaluation.
@pytest.mark.skip(reason="Skipping multi-env test for now")
def test_g1_locomanip_gr00t_closedloop_policy_runner_multi_envs():
    args = [TestConstants.python_path, f"{TestConstants.examples_dir}/policy_runner.py"]
    args.append("--policy_type")
    args.append("gr00t_closedloop")
    args.append("--policy_config_yaml_path")
    args.append(
        TestConstants.test_data_dir + "/test_g1_locomanip_lerobot/test_g1_locomanip_gr00t_closedloop_config.yaml"
    )
    args.append("--num_steps")
    args.append(str(NUM_STEPS))
    args.append("--num_envs")
    args.append(str(NUM_ENVS))
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


if __name__ == "__main__":
    test_g1_locomanip_gr00t_closedloop_policy_runner_single_env()
    test_g1_locomanip_gr00t_closedloop_policy_runner_multi_envs()
