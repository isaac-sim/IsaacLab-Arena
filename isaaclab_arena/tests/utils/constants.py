# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys


def _get_python_path(repo_root: str) -> str:
    """Return the Isaac Sim python interpreter.

    Docker bundles Isaac Sim's ``python.sh`` under the IsaacLab submodule; the
    native uv environment uses uv-managed python.
    """
    docker_python = f"{repo_root}/submodules/IsaacLab/_isaac_sim/python.sh"
    return docker_python if os.path.exists(docker_python) else sys.executable


class _TestConstants:
    """Class for storing test data paths"""

    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # The root directory of the repo
        self.repo_root = os.path.realpath(os.path.join(script_dir, *([".."] * 3)))

        self.examples_dir = f"{self.repo_root}/isaaclab_arena/examples"

        self.test_dir = f"{self.repo_root}/isaaclab_arena/tests"

        self.evaluation_dir = f"{self.repo_root}/isaaclab_arena/evaluation"

        self.arena_environments_dir = f"{self.repo_root}/isaaclab_arena_environments"

        self.scripts_dir = f"{self.repo_root}/isaaclab_arena/scripts"

        self.python_path = _get_python_path(self.repo_root)

        self.test_data_dir = f"{self.test_dir}/test_data"

        self.submodules_dir = f"{self.repo_root}/submodules"


TestConstants = _TestConstants()
