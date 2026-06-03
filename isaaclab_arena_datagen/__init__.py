# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab Arena data-generation package.

Generates SyntheticScene-format HDF5 data (RGB, depth, normals, semantics,
optical/scene flow, dynamic-object poses + mesh samples) from Isaac Lab Arena
simulations, both standalone (:mod:`isaaclab_arena_datagen.run_datagen`) and
during a policy rollout (:mod:`isaaclab_arena_datagen.collection.collector`).

Self-contained: no ``nvblox_next`` or ``pytorch3d`` dependency. Output remains
loadable by nvblox_next's SyntheticScene data loaders.
"""
