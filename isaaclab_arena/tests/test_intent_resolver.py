# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`~isaaclab_arena.agentic_environment_generation.intent_resolver.IntentResolver`.

Covers the graph-wiring logic of :meth:`~IntentResolver.resolve`:

- Node ordering (background → embodiment → items)
- ``env_name`` derivation and override
- Trace lifecycle (cleared between calls)
- Spatial constraint construction: id format, subject/reference wiring,
  params pass-through, graceful skip for unknown nodes
- Task spec construction: id format, state-spec wiring, ``unknown_param`` detection
- Resolution-error reporting via :attr:`~IntentResolver.has_resolution_errors`
"""

from __future__ import annotations

from isaaclab_arena.agentic_environment_generation.environment_intent_spec import EnvironmentIntentSpec, Item
from isaaclab_arena.environments.arena_env_graph_types import ArenaEnvGraphNodeType, SpatialRelationSpec, TaskSpec

from ._resolver_test_helpers import make_resolver, make_scene

# ---------------------------------------------------------------------------
# Top-level resolve()
# ---------------------------------------------------------------------------


def test_resolve_happy_path():
    items = [
        Item(query="bowl", category_tags=["bowl"]),
        Item(query="avocado", category_tags=["fruit"]),
    ]
    initial = [
        SpatialRelationSpec(kind="is_anchor", subject="maple_table"),
        SpatialRelationSpec(kind="on", subject="bowl", reference="maple_table"),
        SpatialRelationSpec(kind="on", subject="avocado", reference="maple_table"),
    ]
    tasks = [
        TaskSpec(
            kind="PickAndPlaceTask",
            params={
                "pick_up_object": "avocado",
                "destination_location": "bowl",
                "background_scene": "maple_table",
            },
            description="put avocado in bowl",
        )
    ]
    spec = make_resolver().resolve(make_scene(items=items, initial_state_graph=initial, tasks=tasks))

    # Auto-derived env_name: f"llm_gen_{background}_{first_task_kind}".
    assert spec.env_name == "llm_gen_maple_table_PickAndPlaceTask"

    # Node order: background, embodiment, items in declaration order.
    node_ids = [n.id for n in spec.nodes]
    assert node_ids == ["maple_table", "franka_ik", "bowl", "avocado"]
    assert spec.nodes_by_id["maple_table"].type == ArenaEnvGraphNodeType.BACKGROUND
    assert spec.nodes_by_id["franka_ik"].type == ArenaEnvGraphNodeType.EMBODIMENT
    # node.name reflects the *resolved* asset name, not the query string.
    assert spec.nodes_by_id["bowl"].name == "bowl_ycb_robolab"
    assert spec.nodes_by_id["avocado"].name == "avocado01_fruits_robolab"

    # Initial state spec is directly accessible.
    initial_state = spec.initial_state_spec
    assert initial_state.id == "state_initial"
    assert len(initial_state.spatial_constraints) == 3

    is_anchor = initial_state.spatial_constraints[0]
    assert is_anchor.kind == "is_anchor"
    assert is_anchor.subject == "maple_table"
    assert is_anchor.reference is None
    assert is_anchor.id == "state_initial_0_is_anchor_maple_table"

    on_bowl = initial_state.spatial_constraints[1]
    assert on_bowl.kind == "on"
    assert on_bowl.reference == "maple_table"
    assert on_bowl.subject == "bowl"
    assert on_bowl.id == "state_initial_1_on_maple_table_bowl"

    # Tasks are plain TaskSpec (no id / state-spec wiring — use resolve() for that).
    assert len(spec.tasks) == 1
    task = spec.tasks[0]
    assert task.kind == "PickAndPlaceTask"
    assert task.params == {
        "pick_up_object": "avocado",
        "destination_location": "bowl",
        "background_scene": "maple_table",
    }
    assert task.description == "put avocado in bowl"


def test_resolve_overrides_env_name():
    resolver = make_resolver()
    spec = resolver.resolve(make_scene(), env_name="my_custom_env")
    assert spec.env_name == "my_custom_env"


def test_resolve_clears_trace_between_calls():
    resolver = make_resolver()
    resolver.resolve(make_scene())
    n_after_first = len(resolver.trace)
    # Sanity: at least background, embodiment, and task events should be present.
    assert n_after_first > 0

    resolver.resolve(make_scene())
    n_after_second = len(resolver.trace)
    # Deterministic input → identical trace length when the trace is cleared.
    assert n_after_second == n_after_first


def test_resolve_with_empty_initial_state_graph():
    spec = make_resolver().resolve(make_scene(initial_state_graph=[]))
    assert spec.initial_state_spec.id == "state_initial"
    assert spec.initial_state_spec.spatial_constraints == []
    assert spec.initial_state_spec.task_constraints == []


# ---------------------------------------------------------------------------
# Resolution-error reporting
# ---------------------------------------------------------------------------


def _clean_scene_kwargs() -> dict:
    """Scene where every node resolves and every task param references a known node.

    The default ``make_scene`` uses a "placeholder" task subject/target that
    deliberately doesn't resolve — fine for node-count tests but unsuitable
    here where we need a baseline with zero error-stage trace events.
    """
    return dict(
        items=[Item(query="bowl", category_tags=["bowl"])],
        tasks=[
            TaskSpec(
                kind="PickAndPlaceTask",
                params={
                    "pick_up_object": "bowl",
                    "destination_location": "bowl",
                    "background_scene": "maple_table",
                },
                description="d",
            )
        ],
    )


def test_has_resolution_errors_false_on_clean_run():
    resolver = make_resolver()
    resolver.resolve(make_scene(**_clean_scene_kwargs()))
    assert resolver.resolution_errors == []
    assert resolver.has_resolution_errors is False


def test_has_resolution_errors_true_when_item_unresolvable():
    # An unresolvable item on top of the clean baseline means the only error
    # stage that fires is "item.miss".
    kwargs = _clean_scene_kwargs()
    kwargs["items"] = kwargs["items"] + [Item(query="zzz_no_match_anywhere", category_tags=["object"])]
    resolver = make_resolver()
    resolver.resolve(make_scene(**kwargs))
    assert resolver.has_resolution_errors is True
    assert [e.stage for e in resolver.resolution_errors] == ["item.miss"]


def test_has_resolution_errors_false_when_only_tag_relaxation():
    # Tag-pool relaxation is a warning, not an error: the item is still resolved
    # via the broader object pool.
    # "cracker" is in the catalog (as "cracker_box", tagged "graspable") but
    # NOT tagged "fruit" — so the fruit pool yields no match and the resolver
    # relaxes to the full object pool.
    kwargs = _clean_scene_kwargs()
    kwargs["items"] = [Item(query="cracker", category_tags=["fruit"])]
    kwargs["tasks"] = [
        TaskSpec(
            kind="PickAndPlaceTask",
            params={
                "pick_up_object": "cracker",
                "destination_location": "cracker",
                "background_scene": "maple_table",
            },
            description="d",
        )
    ]
    resolver = make_resolver()
    resolver.resolve(make_scene(**kwargs))
    trace_stages = [e.stage for e in resolver.trace]
    assert "item.no_match_in_tags" in trace_stages
    assert resolver.has_resolution_errors is False


def test_has_resolution_errors_true_when_embodiment_unknown():
    # An unknown embodiment with no fuzzy match emits "embodiment.miss" which
    # is an error stage — no silent fallback to a hardcoded default.
    resolver = make_resolver()
    resolver.resolve(make_scene(embodiment="totally_unknown_robot"))
    assert "embodiment.miss" in [e.stage for e in resolver.trace]
    assert resolver.has_resolution_errors is True


# ---------------------------------------------------------------------------
# Spatial constraint construction
# ---------------------------------------------------------------------------


def test_spatial_constraint_binary_relation_id_and_fields():
    items = [Item(query="cracker_box", category_tags=["graspable"])]
    initial = [SpatialRelationSpec(kind="on", subject="cracker_box", reference="maple_table")]
    spec = make_resolver().resolve(make_scene(items=items, initial_state_graph=initial))
    constraint = spec.initial_state_spec.spatial_constraints[0]
    assert constraint.reference == "maple_table"
    assert constraint.subject == "cracker_box"
    assert constraint.id == "state_initial_0_on_maple_table_cracker_box"


def test_spatial_constraint_unary_relation_id_and_fields():
    initial = [SpatialRelationSpec(kind="is_anchor", subject="maple_table")]
    spec = make_resolver().resolve(make_scene(initial_state_graph=initial))
    constraint = spec.initial_state_spec.spatial_constraints[0]
    assert constraint.kind == "is_anchor"
    assert constraint.subject == "maple_table"
    assert constraint.reference is None
    assert constraint.id == "state_initial_0_is_anchor_maple_table"


def test_spatial_constraint_unknown_subject_skipped():
    initial = [SpatialRelationSpec(kind="on", subject="not_a_node", reference="maple_table")]
    resolver = make_resolver()
    spec = resolver.resolve(make_scene(initial_state_graph=initial))
    assert spec.initial_state_spec.spatial_constraints == []
    assert any(e.stage == "relation.initial.unknown_subject" for e in resolver.trace)


def test_spatial_constraint_unknown_reference_skipped():
    initial = [SpatialRelationSpec(kind="on", subject="maple_table", reference="missing_node")]
    resolver = make_resolver()
    spec = resolver.resolve(make_scene(initial_state_graph=initial))
    assert spec.initial_state_spec.spatial_constraints == []
    assert any(e.stage == "relation.initial.unknown_reference" for e in resolver.trace)


def test_spatial_constraint_params_passed_through():
    items = [Item(query="cracker_box", category_tags=["graspable"])]
    initial = [
        SpatialRelationSpec(
            kind="at_position",
            subject="cracker_box",
            params={"position_xyz": [0.1, 0.2, 0.3]},
        ),
    ]
    spec = make_resolver().resolve(make_scene(items=items, initial_state_graph=initial))
    constraint = spec.initial_state_spec.spatial_constraints[0]
    assert constraint.kind == "at_position"
    # params are passed through verbatim; downstream builders interpret them.
    assert constraint.params == {"position_xyz": (0.1, 0.2, 0.3)}


# ---------------------------------------------------------------------------
# Task spec construction
# ---------------------------------------------------------------------------


def test_multiple_tasks_preserved_in_order():
    tasks = [
        TaskSpec(
            kind="PickAndPlaceTask",
            params={
                "pick_up_object": "bowl",
                "destination_location": "bowl",
                "background_scene": "maple_table",
            },
            description="d1",
        ),
        TaskSpec(kind="OpenDoorTask", params={"openable_object": "bowl"}, description="d2"),
        TaskSpec(kind="CloseDoorTask", params={"openable_object": "bowl"}, description="d3"),
    ]
    items = [Item(query="bowl", category_tags=["bowl"])]
    spec = make_resolver().resolve(make_scene(items=items, tasks=tasks))

    # Tasks are plain TaskSpec entries (no id / wiring) preserved in declaration order.
    assert len(spec.tasks) == 3
    assert [t.kind for t in spec.tasks] == ["PickAndPlaceTask", "OpenDoorTask", "CloseDoorTask"]
    assert [t.description for t in spec.tasks] == ["d1", "d2", "d3"]


def test_task_unknown_param_emits_error_trace():
    # A task param whose value is a string not matching any resolved node id
    # triggers a "task.unknown_param" error-stage trace event.
    items = [Item(query="bowl", category_tags=["bowl"])]
    tasks = [
        TaskSpec(
            kind="PickAndPlaceTask",
            params={
                "pick_up_object": "nonexistent_object",
                "destination_location": "bowl",
                "background_scene": "maple_table",
            },
            description="d",
        )
    ]
    resolver = make_resolver()
    resolver.resolve(make_scene(items=items, tasks=tasks))
    assert resolver.has_resolution_errors is True
    error_stages = [e.stage for e in resolver.resolution_errors]
    assert "task.unknown_param" in error_stages


def test_resolve_scene_with_no_tasks_derives_env_name():
    # When the task list is empty the env_name falls back to "llm_gen_{bg}_task".
    scene = EnvironmentIntentSpec.model_construct(
        reasoning="test",
        background="maple_table",
        embodiment="franka_ik",
        items=[],
        initial_state_graph=[],
        tasks=[],
    )
    spec = make_resolver().resolve(scene)
    assert spec.env_name == "llm_gen_maple_table_task"
