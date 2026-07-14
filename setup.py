# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Installation script for the 'isaaclab_arena' python package."""

from setuptools import find_packages, setup

ISAACLAB_ARENA_VERSION_NUMBER = "1.0.0"

RUNTIME_DEPS = [
    "typing_extensions",
    "onnxruntime",
    "vuer[all]",
    "lightwheel-sdk",
    "pytest",
    "pydantic>=2.0",
    "openai>=2.0",
    # Sensitivity analysis (isaaclab_arena.analysis.sensitivity), imported at module level.
    "sbi",
    "scipy",
    "matplotlib",
    # HDF5 -> LeRobot conversion (isaaclab_arena_gr00t.lerobot.convert_hdf5_to_lerobot), imported at module level.
    # Pinned to 2.2.3 to match the version used in GR00T.
    "pandas==2.2.3",
]

DEV_DEPS = [
    "jupyter",
    "debugpy",
    "tenacity",
    "streamlit>=1.30",
    "streamlit-ace>=0.1.1",
]

setup(
    name="isaaclab_arena",
    version=ISAACLAB_ARENA_VERSION_NUMBER,
    description="Isaac Lab - Arena. An Isaac Lab extension for robotic policy evaluation. ",
    packages=find_packages(
        include=[
            "isaaclab_arena*",
            "isaaclab_arena_curobo*",
            "isaaclab_arena_environments*",
            "isaaclab_arena_examples*",
            "isaaclab_arena_dreamzero*",
            "isaaclab_arena_g1*",
            "isaaclab_arena_gr00t*",
            "isaaclab_arena_openpi*",
        ]
    ),
    python_requires=">=3.10",
    install_requires=RUNTIME_DEPS,
    extras_require={
        "dev": DEV_DEPS,
    },
    zip_safe=False,
)
