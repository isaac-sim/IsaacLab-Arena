# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`~isaaclab_arena.agentic_environment_generation.intent_compiler`.

Covers :class:`~intent_compiler.IntentCompiler` graph-wiring logic:

  - Node ordering (background → embodiment → items)
  - ``env_name`` derivation and override
  - Trace lifecycle (cleared between calls)
  - Spatial constraint construction: id format, subject/reference wiring,
    params pass-through, graceful skip for unknown nodes
  - Task spec construction: kind/params/description preserved in order
  - Resolution-error reporting via :attr:`~IntentCompiler.has_resolution_errors`
"""

from __future__ import annotations

from isaaclab_arena.agentic_environment_generation.environment_intent_spec import EnvironmentIntentSpec, Item
from isaaclab_arena.agentic_environment_generation.intent_compiler import IntentCompiler
from isaaclab_arena.environments.arena_env_graph_types import ArenaEnvGraphNodeType, SpatialRelationSpec, TaskSpec
from isaaclab_arena.tests._asset_matcher_test_helpers import FakeAsset, make_registry


def _make_compiler(assets: list[FakeAsset] | None = None) -> IntentCompiler:
    """Return an :class:`IntentCompiler` backed by a fake registry."""
    return IntentCompiler(registry=make_registry(assets))


def _make_scene(
    *,
    background: str = "maple_table",
    embodiment: str = "franka_ik",
    items: list[Item] | None = None,
    initial_state_graph: list[SpatialRelationSpec] | None = None,
    tasks: list[TaskSpec] | None = None,
) -> EnvironmentIntentSpec:
    """Build an :class:`EnvironmentIntentSpec` with sane defaults."""
    return EnvironmentIntentSpec(
        reasoning="test scene",
        background=background,
        embodiment=embodiment,
        items=items or [],
        initial_state_graph=initial_state_graph or [],
        tasks=tasks
        or [
            TaskSpec(
                kind="PickAndPlaceTask",
                params={
                    "pick_up_object": "placeholder",
                    "destination_location": "placeholder",
                    "background_scene": "maple_table",
                },
                description="placeholder",
            )
        ],
    )


def _clean_scene_kwargs() -> dict:
    """Scene where every node resolves and every task param references a known node."""
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


def test_item_instance_name_overrides_query_for_node_id():
    # ``instance_name`` becomes the graph node id; ``name`` still reflects the
    # resolved asset, allowing the same asset to appear under different ids.
    items = [Item(query="bowl", category_tags=["bowl"], instance_name="serving_bowl")]
    spec = _make_compiler().compile(_make_scene(items=items))
    assert "serving_bowl" in spec.nodes_by_id
    assert "bowl" not in spec.nodes_by_id
    assert spec.nodes_by_id["serving_bowl"].name == "bowl_ycb_robolab"


def test_background_with_wrong_tag_omitted():
    assets = [
        FakeAsset(name="franka_ik", tags=["embodiment"]),
        FakeAsset(name="maple_table", tags=["object"]),  # wrong tag
    ]
    compiler = _make_compiler(assets)
    spec = compiler.compile(_make_scene(background="maple_table"))
    assert "maple_table" not in spec.nodes_by_id
    assert any(e.stage == "background.required_tags.empty_pool" for e in compiler.trace)
    assert compiler.has_resolution_errors is True


def test_embodiment_required_tag_pool_empty_records_error():
    assets = [
        FakeAsset(name="maple_table", tags=["background"]),
        FakeAsset(name="cracker_box", tags=["object"]),
    ]
    compiler = _make_compiler(assets)
    spec = compiler.compile(_make_scene(embodiment="franka_ik"))
    assert not any(n.type.value == "embodiment" for n in spec.nodes)
    assert any(e.stage == "embodiment.required_tags.empty_pool" for e in compiler.trace)
    assert compiler.has_resolution_errors is True


def test_item_required_tag_pool_empty_records_error():
    assets = [
        FakeAsset(name="maple_table", tags=["background"]),
        FakeAsset(name="franka_ik", tags=["embodiment"]),
    ]
    compiler = _make_compiler(assets)
    spec = compiler.compile(_make_scene(items=[Item(query="bowl", category_tags=["bowl"])]))
    assert "bowl" not in spec.nodes_by_id
    assert any(e.stage == "item.required_tags.empty_pool" for e in compiler.trace)
    assert compiler.has_resolution_errors is True


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
    spec = _make_compiler().compile(_make_scene(items=items, initial_state_graph=initial, tasks=tasks))

    assert spec.env_name == "llm_gen_maple_table_PickAndPlaceTask"

    node_ids = [n.id for n in spec.nodes]
    assert node_ids == ["maple_table", "franka_ik", "bowl", "avocado"]
    assert spec.nodes_by_id["maple_table"].type == ArenaEnvGraphNodeType.BACKGROUND
    assert spec.nodes_by_id["franka_ik"].type == ArenaEnvGraphNodeType.EMBODIMENT
    assert spec.nodes_by_id["bowl"].name == "bowl_ycb_robolab"
    assert spec.nodes_by_id["avocado"].name == "avocado01_fruits_robolab"

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
    spec = _make_compiler().compile(_make_scene(), env_name="my_custom_env")
    assert spec.env_name == "my_custom_env"


def test_resolve_clears_trace_between_calls():
    compiler = _make_compiler()
    compiler.compile(_make_scene())
    n_after_first = len(compiler.trace)
    assert n_after_first > 0

    compiler.compile(_make_scene())
    assert len(compiler.trace) == n_after_first


def test_resolve_with_empty_initial_state_graph():
    spec = _make_compiler().compile(_make_scene(initial_state_graph=[]))
    assert spec.initial_state_spec.id == "state_initial"
    assert spec.initial_state_spec.spatial_constraints == []
    assert spec.initial_state_spec.task_constraints == []


def test_has_resolution_errors_false_on_clean_run():
    compiler = _make_compiler()
    compiler.compile(_make_scene(**_clean_scene_kwargs()))
    assert compiler.resolution_errors == []
    assert compiler.has_resolution_errors is False


def test_has_resolution_errors_true_when_item_unresolvable():
    kwargs = _clean_scene_kwargs()
    kwargs["items"] = kwargs["items"] + [Item(query="zzz_no_match_anywhere", category_tags=["object"])]
    compiler = _make_compiler()
    compiler.compile(_make_scene(**kwargs))
    assert compiler.has_resolution_errors is True
    assert [e.stage for e in compiler.resolution_errors] == ["item.required_tags.miss"]


def test_has_resolution_errors_false_when_only_tag_relaxation():
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
    compiler = _make_compiler()
    compiler.compile(_make_scene(**kwargs))
    assert "item.preferred_tags.miss" in [e.stage for e in compiler.trace]
    assert compiler.has_resolution_errors is False


def test_has_resolution_errors_true_when_embodiment_unknown():
    compiler = _make_compiler()
    compiler.compile(_make_scene(embodiment="totally_unknown_robot"))
    assert "embodiment.required_tags.miss" in [e.stage for e in compiler.trace]
    assert compiler.has_resolution_errors is True


def test_spatial_constraint_binary_relation_id_and_fields():
    items = [Item(query="cracker_box", category_tags=["graspable"])]
    initial = [SpatialRelationSpec(kind="on", subject="cracker_box", reference="maple_table")]
    spec = _make_compiler().compile(_make_scene(items=items, initial_state_graph=initial))
    constraint = spec.initial_state_spec.spatial_constraints[0]
    assert constraint.reference == "maple_table"
    assert constraint.subject == "cracker_box"
    assert constraint.id == "state_initial_0_on_maple_table_cracker_box"


def test_spatial_constraint_unary_relation_id_and_fields():
    initial = [SpatialRelationSpec(kind="is_anchor", subject="maple_table")]
    spec = _make_compiler().compile(_make_scene(initial_state_graph=initial))
    constraint = spec.initial_state_spec.spatial_constraints[0]
    assert constraint.kind == "is_anchor"
    assert constraint.subject == "maple_table"
    assert constraint.reference is None
    assert constraint.id == "state_initial_0_is_anchor_maple_table"


def test_spatial_constraint_unknown_subject_skipped():
    initial = [SpatialRelationSpec(kind="on", subject="not_a_node", reference="maple_table")]
    compiler = _make_compiler()
    spec = compiler.compile(_make_scene(initial_state_graph=initial))
    assert spec.initial_state_spec.spatial_constraints == []
    assert any(e.stage == "relation.initial.unknown_subject" for e in compiler.trace)


def test_spatial_constraint_unknown_reference_skipped():
    initial = [SpatialRelationSpec(kind="on", subject="maple_table", reference="missing_node")]
    compiler = _make_compiler()
    spec = compiler.compile(_make_scene(initial_state_graph=initial))
    assert spec.initial_state_spec.spatial_constraints == []
    assert any(e.stage == "relation.initial.unknown_reference" for e in compiler.trace)


def test_spatial_constraint_params_passed_through():
    items = [Item(query="cracker_box", category_tags=["graspable"])]
    initial = [
        SpatialRelationSpec(
            kind="at_position",
            subject="cracker_box",
            params={"position_xyz": [0.1, 0.2, 0.3]},
        ),
    ]
    spec = _make_compiler().compile(_make_scene(items=items, initial_state_graph=initial))
    constraint = spec.initial_state_spec.spatial_constraints[0]
    assert constraint.kind == "at_position"
    assert constraint.params == {"position_xyz": (0.1, 0.2, 0.3)}


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
    spec = _make_compiler().compile(_make_scene(items=items, tasks=tasks))
    assert len(spec.tasks) == 3
    assert [t.kind for t in spec.tasks] == ["PickAndPlaceTask", "OpenDoorTask", "CloseDoorTask"]
    assert [t.description for t in spec.tasks] == ["d1", "d2", "d3"]


def test_task_unknown_param_emits_error_trace():
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
    compiler = _make_compiler()
    compiler.compile(_make_scene(items=items, tasks=tasks))
    assert compiler.has_resolution_errors is True
    assert "task.unknown_param" in [e.stage for e in compiler.resolution_errors]
