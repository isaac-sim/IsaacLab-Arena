# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Installation script for the 'isaaclab_arena' python package."""

from setuptools import find_packages, setup

ISAACLAB_ARENA_VERSION_NUMBER = "1.0.0"

setup(
    name="isaaclab_arena",
    version=ISAACLAB_ARENA_VERSION_NUMBER,
    description="Isaac Lab - Arena. An Isaac Lab extension for robotic policy evaluation. ",
    packages=find_packages(include=["isaaclab_arena*", "isaaclab_arena_g1*", "isaaclab_arena_gr00t*"]),
    python_requires=">=3.10",
    zip_safe=False,
)
