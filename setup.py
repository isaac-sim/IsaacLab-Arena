#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""Installation script for the 'isaaclab_arena' python package."""

from setuptools import find_packages, setup

ISAACLAB_ARENA_VERSION_NUMBER = "1.0.0"

setup(
    name="isaaclab_arena",
    version=ISAACLAB_ARENA_VERSION_NUMBER,
    description="Isaac Lab - Arena. An Isaac Lab extension for robotic policy evaluation. ",
    packages=find_packages(),
    python_requires=">=3.10",
    zip_safe=False,
)
