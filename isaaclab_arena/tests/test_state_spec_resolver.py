# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environments.arena_env_graph_types import ArenaEnvGraphStateSpec

_TEST_DATA = Path(__file__).parent / "test_data"
_INIT_GRAPH = _TEST_DATA / "pick_and_place_maple_table_init_env_graph.yaml"
_FULL_GRAPH = _TEST_DATA / "pick_and_place_maple_table_env_graph.yaml"


def _spatial_by_id(state: ArenaEnvGraphStateSpec) -> dict[str, tuple]:
    """Project a state's spatial constraints to id -> (type, parent, child, params), order-insensitive."""
    return {c.id: (c.type, c.parent, c.child, c.params) for c in state.spatial_constraints}


def _task_constraints_by_id(state: ArenaEnvGraphStateSpec) -> dict[str, tuple]:
    return {c.id: (c.type, c.parent, c.child, c.params) for c in state.task_constraints}


def test_populate_reproduces_groundtruth_full_graph():
    """Resolving the partial init graph yields the hand-authored full graph (structurally)."""
    populated = ArenaEnvGraphSpec.from_yaml(_INIT_GRAPH, is_task_wiring_enabled=False).resolve_constraints()
    groundtruth = ArenaEnvGraphSpec.from_yaml(_FULL_GRAPH)

    # Same set of states, chained 0..N.
    assert set(populated.state_specs_by_id) == set(groundtruth.state_specs_by_id)

    # Each state's spatial + task constraints match (keyed by id, so order is irrelevant).
    for state_id, groundtruth_state in groundtruth.state_specs_by_id.items():
        got = populated.state_specs_by_id[state_id]
        assert _spatial_by_id(got) == _spatial_by_id(groundtruth_state), f"spatial mismatch in {state_id}"
        assert _task_constraints_by_id(got) == _task_constraints_by_id(
            groundtruth_state
        ), f"task mismatch in {state_id}"

    # Tasks are wired into the chain identically.
    assert set(populated.tasks_by_id) == set(groundtruth.tasks_by_id)
    for task_id, groundtruth_task in groundtruth.tasks_by_id.items():
        got = populated.tasks_by_id[task_id]
        assert got.type == groundtruth_task.type
        assert got.initial_state_spec_id == groundtruth_task.initial_state_spec_id
        assert got.success_state_spec_id == groundtruth_task.success_state_spec_id
        assert got.task_args == groundtruth_task.task_args


def test_unresolved_graph_is_not_directly_loadable():
    """The init graph has NULL state-spec ids, so the strict loader rejects it — motivating the resolver."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ArenaEnvGraphSpec.from_yaml(_INIT_GRAPH)


def test_chain_wires_each_success_state_as_next_initial_state():
    """task[i].success_state_spec_id == task[i+1].initial_state_spec_id (a single chain)."""
    populated = ArenaEnvGraphSpec.from_yaml(_INIT_GRAPH, is_task_wiring_enabled=False).resolve_constraints()
    ordered = sorted(populated.tasks_by_id.values(), key=lambda t: t.id)
    for earlier, later in zip(ordered, ordered[1:]):
        assert earlier.success_state_spec_id == later.initial_state_spec_id


def test_task_without_a_transition_is_rejected():
    """A task whose class declares no success_state_transition fails loudly rather than silently skipping."""
    import pytest

    spec = ArenaEnvGraphSpec.from_yaml(_INIT_GRAPH, is_task_wiring_enabled=False)
    spec.tasks[0].type = "NoTask"  # registered, but declares no transition
    with pytest.raises(NotImplementedError, match="success_state_transition not implemented"):
        spec.resolve_constraints()
