# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import get_args

from isaaclab_arena.agentic_environment_generation.environment_intent_spec import RelationKind
from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpatialConstraintType


def test_relation_kind_is_subset_of_spatial_constraint_type():
    """``RelationKind`` must be a subset of the engine's spatial-constraint enum."""
    wire_kinds = set(get_args(RelationKind))
    engine_kinds = {member.value for member in ArenaEnvGraphSpatialConstraintType}
    extras = wire_kinds - engine_kinds
    assert not extras, (
        "RelationKind contains values absent from ArenaEnvGraphSpatialConstraintType: "
        f"{sorted(extras)}. Engine supports: {sorted(engine_kinds)}. "
        "Either add the missing primitive to the engine enum, or remove it from RelationKind."
    )
