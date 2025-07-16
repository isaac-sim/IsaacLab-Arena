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
"""Installation script for the 'isaac_arena' python package."""

from setuptools import find_packages, setup

setup(
    name="isaac_arena",
    version="0.1.0",
    description="Isaac Arena environment for Robotics Simulations. ",
    packages=find_packages(),
    python_requires=">=3.10",
    zip_safe=False,
)
