# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal runtime geometry helpers for object predicates."""

from __future__ import annotations

from isaaclab_arena.scene.object_state import get_env, object_state
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


def object_bounds_w(env, name: str) -> AxisAlignedBoundingBox:
    """Return an object's authored local bounds transformed to its runtime pose."""
    env = get_env(env)
    assert name in env.arena_object_bounds, f"Arena local bounds metadata is missing for {name!r}"
    state = object_state(env, name).aggregate
    assert state.position_w is not None, f"Object {name!r} has no aggregate position"
    assert state.orientation_w is not None, f"Object {name!r} has no aggregate orientation"
    local_bounds = env.arena_object_bounds[name].to(state.position_w.device)
    return local_bounds.rotated_by_quat(state.orientation_w).translated(state.position_w)
