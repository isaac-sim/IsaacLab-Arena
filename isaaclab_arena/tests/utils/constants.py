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

        self.policy_dir = f"{self.repo_root}/isaaclab_arena/policy"


TestConstants = _TestConstants()
