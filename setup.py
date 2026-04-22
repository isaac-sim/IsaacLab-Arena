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
    # Used lazily by isaaclab_arena/llm_env_gen/* for NV_API_KEY-based LLM calls.
    "openai",
]

DEV_DEPS = [
    "jupyter",
    "debugpy",
    "tenacity",
]

setup(
    name="isaaclab_arena",
    version=ISAACLAB_ARENA_VERSION_NUMBER,
    description="Isaac Lab - Arena. An Isaac Lab extension for robotic policy evaluation. ",
    packages=find_packages(
        include=[
            "isaaclab_arena*",
            "isaaclab_arena_environments*",
            "isaaclab_arena_examples*",
            "isaaclab_arena_g1*",
            "isaaclab_arena_gr00t*",
        ]
    ),
    python_requires=">=3.10",
    install_requires=RUNTIME_DEPS,
    extras_require={
        "dev": DEV_DEPS,
    },
    zip_safe=False,
)
