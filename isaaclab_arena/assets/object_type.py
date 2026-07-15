# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lightweight enum module for ObjectType.

Kept dependency-free so it can be imported by pure-Python spec / schema modules
(e.g. arena_env_graph_spec.py) without dragging in isaaclab/omni/pxr. This is
important because importing pxr before SimulationApp starts breaks Kit
extensions like omni.kit.usd.mdl during pytest collection.

`isaaclab_arena.assets.object_base` re-exports `ObjectType` from here so existing
`from isaaclab_arena.assets.object_base import ObjectType` consumers keep
working with a single source of truth.
"""

from enum import Enum


class ObjectType(str, Enum):
    BASE = "base"
    RIGID = "rigid"
    ARTICULATION = "articulation"
