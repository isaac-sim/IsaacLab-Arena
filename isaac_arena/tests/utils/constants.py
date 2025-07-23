# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

import os


class _TestConstants:
    """Class for storing test data paths"""

    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # The root directory of the repo
        self.repo_root = os.path.realpath(os.path.join(script_dir, *([".."] * 3)))

        self.examples_dir = f"{self.repo_root}/isaac_arena/examples/"

        self.test_dir = f"{self.repo_root}/isaac_arena/tests/"

        self.python_path = f"{self.repo_root}/submodules/IsaacLab-Internal/_isaac_sim/python.sh"


TestConstants = _TestConstants()
