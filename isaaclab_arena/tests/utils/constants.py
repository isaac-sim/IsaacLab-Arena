# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os


class _TestConstants:
    """Class for storing test data paths"""

    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # The root directory of the repo
        self.repo_root = os.path.realpath(os.path.join(script_dir, *([".."] * 3)))

        self.examples_dir = f"{self.repo_root}/isaaclab_arena/examples"

        self.test_dir = f"{self.repo_root}/isaaclab_arena/tests"

        self.python_path = f"{self.repo_root}/submodules/IsaacLab/_isaac_sim/python.sh"

        self.test_data_dir = f"{self.test_dir}/test_data"


TestConstants = _TestConstants()
