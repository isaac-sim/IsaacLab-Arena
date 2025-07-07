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
"""Installation script for the 'omni.isaac.lab_tasks' python package."""

import itertools
import os
import platform
import toml    # type: ignore[import-untyped]
from typing import Dict, List

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, 'extension.toml'))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    'numpy',
    'h5py',
]

# PYTORCH_INDEX_URL = ['https://downkload.pytorch.org/whl/cu118']

# Extra dependencies for RL agents
EXTRAS_REQUIRE: Dict[str, List[str]] = {
}

# Cumulation of all extra-requires
all_requirements = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))
# Remove duplicates in the all list to avoid double installations
EXTRAS_REQUIRE['all'] = list(set(all_requirements))

# Installation operation
setup(
    name='isaac_arena',
    author='Isaac Lab Project Developers',
    maintainer='Isaac Lab Project Developers',
    url=EXTENSION_TOML_DATA['package']['repository'],
    version=EXTENSION_TOML_DATA['package']['version'],
    description=EXTENSION_TOML_DATA['package']['description'],
    keywords=EXTENSION_TOML_DATA['package']['keywords'],
    include_package_data=True,
    python_requires='>=3.10',
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    packages=['isaac_arena'],
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.10',
        'Isaac Sim :: 4.2.0',
        'Isaac Sim :: 4.1.0',
    ],
    zip_safe=False,
)
