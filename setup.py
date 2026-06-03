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
    # isaaclab_arena_datagen: SyntheticScene HDF5 writer (Zstd-compressed).
    "h5py",
    "hdf5plugin",
]

DEV_DEPS = [
    "jupyter",
    "debugpy",
    "tenacity",
]

# isaaclab_arena_datagen: optional dependencies for per-scene visualizations
# (PNG/MP4/HTML) and mesh-surface sampling. Imported lazily so the core data
# pipeline runs without them.
DATAGEN_VIZ_DEPS = [
    "plotly",
    "scipy",
    "imageio",
    "trimesh",
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
            "isaaclab_arena_openpi*",
            "isaaclab_arena_datagen*",
        ]
    ),
    python_requires=">=3.10",
    install_requires=RUNTIME_DEPS,
    extras_require={
        "dev": DEV_DEPS,
        "datagen-viz": DATAGEN_VIZ_DEPS,
    },
    zip_safe=False,
)
