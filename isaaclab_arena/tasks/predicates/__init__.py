# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Reusable per-env predicate functions for fine-grained subtask tracking.

Predicates return a ``(num_envs,)`` bool tensor by default, or a 0-D bool tensor
when called with ``env_id=<int>``. They read scene state directly from the
manager-based env (``env.scene[name]``) so they can be used in any Isaac Lab
context that exposes that surface: termination/event terms, recorders, tests
that mock the env, or stand-alone debugging.

Use them in a ``FineGrainedSubtask`` via ``functools.partial`` to bind the
object-specific arguments::

    from functools import partial
    from isaaclab_arena.tasks.predicates import object_grabbed, object_picked_up
    from isaaclab_arena.tasks.fine_grained_subtask import FineGrainedSubtask

    FineGrainedSubtask(
        name="lift_cracker_box",
        conditions=[
            partial(object_grabbed, object_name="cracker_box"),
            partial(object_picked_up, object_name="cracker_box",
                    surface_object_name="table", distance=0.1),
        ],
    )
"""

from isaaclab_arena.tasks.predicates.decorators import atomic, composite
from isaaclab_arena.tasks.predicates.geometry import (
    get_root_ang_vel_w,
    get_root_lin_vel_w,
    get_root_pos_w,
    get_root_quat_w,
    logical_and,
    logical_not,
    logical_or,
)
from isaaclab_arena.tasks.predicates.grasp import (
    object_dropped,
    object_grabbed,
    object_in_contact,
)
from isaaclab_arena.tasks.predicates.pose import object_stationary, object_upright
from isaaclab_arena.tasks.predicates.spatial import (
    object_above,
    object_at,
    object_near,
    object_picked_up,
    objects_in_proximity_predicate,
)

__all__ = [
    # decorators
    "atomic",
    "composite",
    # geometry helpers
    "get_root_pos_w",
    "get_root_quat_w",
    "get_root_lin_vel_w",
    "get_root_ang_vel_w",
    "logical_and",
    "logical_or",
    "logical_not",
    # spatial
    "object_above",
    "object_at",
    "object_near",
    "object_picked_up",
    "objects_in_proximity_predicate",
    # pose
    "object_upright",
    "object_stationary",
    # grasp / contact
    "object_in_contact",
    "object_grabbed",
    "object_dropped",
]
