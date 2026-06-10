# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`~isaaclab_arena.agentic_environment_generation.intent_compiler`.

Covers both the asset-matching helpers (:func:`~intent_compiler.match_asset`) and
the graph-wiring logic of :class:`~intent_compiler.IntentCompiler`:

Asset matching
  - Exact name match (no fuzzy search triggered)
  - Substring match within a tag-narrowed pool
  - Tag-pool relaxation when the preferred pool is empty or yields no match
  - Fuzzy (difflib) fallback
  - Miss / omission behaviour
  - Embodiment bare-family expansion via the ``["embodiment", "ik"]`` tag pool
  - Background resolution when the required-tag pool is empty

IntentCompiler.compile()
  - Node ordering (background → embodiment → items)
  - ``env_name`` derivation and override
  - Trace lifecycle (cleared between calls)
  - Spatial constraint construction: id format, subject/reference wiring,
    params pass-through, graceful skip for unknown nodes
  - Task spec construction: kind/params/description preserved in order
  - Resolution-error reporting via :attr:`~IntentCompiler.has_resolution_errors`
"""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab_arena.agentic_environment_generation.environment_intent_spec import EnvironmentIntentSpec, Item
from isaaclab_arena.agentic_environment_generation.intent_compiler import IntentCompiler
from isaaclab_arena.environments.arena_env_graph_types import ArenaEnvGraphNodeType, SpatialRelationSpec, TaskSpec

# =============================================================================
# Test fixtures
# =============================================================================


@dataclass
class FakeAsset:
    """Minimal stand-in for the asset classes the resolvers inspect.

    Real asset classes are decorated classes pulled in via
    ``ensure_assets_registered()``.  The resolvers only ever read ``.name``
    and ``.tags``, so a plain dataclass keeps tests independent of Isaac Sim.
    """

    name: str
    tags: list[str]


class FakeAssetRegistry:
    """Duck-typed AssetRegistry for unit tests.

    Implements the methods the resolvers call — ``is_registered``,
    ``get_asset_by_name``, ``get_assets_by_tag``, ``get_assets_with_all_tags``,
    ``get_all_keys`` — without pulling in isaaclab.  We deliberately don't
    subclass :class:`AssetRegistry` because it uses ``SingletonMeta``, which
    would force test-isolation gymnastics.  Duck-typing via the resolver's
    ``registry`` argument is the supported injection point.
    """

    def __init__(self, assets: list[FakeAsset]) -> None:
        self._by_name: dict[str, FakeAsset] = {a.name: a for a in assets}

    def is_registered(self, key: str) -> bool:
        return key in self._by_name

    def get_asset_by_name(self, name: str) -> FakeAsset:
        assert name in self._by_name, f"unregistered asset: {name}"
        return self._by_name[name]

    def get_assets_by_tag(self, tag: str) -> list[FakeAsset]:
        return [a for a in self._by_name.values() if tag in a.tags]

    def get_assets_with_all_tags(self, tags: list[str]) -> list[str]:
        return sorted(asset.name for asset in self._by_name.values() if all(tag in asset.tags for tag in tags))

    def get_all_keys(self) -> list[str]:
        return list(self._by_name)


def _default_assets() -> list[FakeAsset]:
    """Small but representative catalog covering all three asset categories.

    Object names intentionally include the suffix conventions seen in the real
    registry (e.g. ``bowl_ycb_robolab``) so substring-match tests exercise
    realistic behaviour.
    """
    return [
        FakeAsset(name="maple_table", tags=["background"]),
        FakeAsset(name="franka_ik", tags=["embodiment", "ik"]),
        FakeAsset(name="franka_joint_pos", tags=["embodiment"]),
        FakeAsset(name="bowl_ycb_robolab", tags=["object", "bowl"]),
        FakeAsset(name="avocado01_fruits_robolab", tags=["object", "fruit"]),
        FakeAsset(name="apple01_fruits_robolab", tags=["object", "fruit"]),
        FakeAsset(name="cracker_box", tags=["object", "graspable"]),
    ]


def _make_compiler(assets: list[FakeAsset] | None = None) -> IntentCompiler:
    """Return an :class:`IntentCompiler` backed by a :class:`FakeAssetRegistry`."""
    return IntentCompiler(registry=FakeAssetRegistry(assets or _default_assets()))


def _make_scene(
    *,
    background: str = "maple_table",
    embodiment: str = "franka_ik",
    items: list[Item] | None = None,
    initial_state_graph: list[SpatialRelationSpec] | None = None,
    tasks: list[TaskSpec] | None = None,
) -> EnvironmentIntentSpec:
    """Build an :class:`EnvironmentIntentSpec` with sane defaults.

    Tests that don't care about a particular field can leave it unset; the
    defaults produce a valid spec that resolves cleanly against
    :func:`_default_assets`.
    """
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
    """Scene where every node resolves and every task param references a known node.

    The default ``_make_scene`` uses a "placeholder" task subject/target that
    deliberately doesn't resolve — fine for node-count tests but unsuitable
    where we need a baseline with zero error-stage trace events.
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


# =============================================================================
# Item resolution (match_asset)
# =============================================================================


def test_item_exact_name_match():
    # A query that is already a registered asset name skips all fuzzy matching.
    items = [Item(query="cracker_box", category_tags=["graspable"])]
    compiler = _make_compiler()
    spec = compiler.compile(_make_scene(items=items))
    assert spec.nodes_by_id["cracker_box"].name == "cracker_box"
    assert any(e.stage == "item.exact" for e in compiler.trace)


def test_item_substring_match_in_tag_pool():
    items = [Item(query="bowl", category_tags=["bowl"])]
    compiler = _make_compiler()
    spec = compiler.compile(_make_scene(items=items))
    assert spec.nodes_by_id["bowl"].name == "bowl_ycb_robolab"
    assert any(e.stage == "item.preferred_tags.substring" for e in compiler.trace)


def test_item_relaxes_when_tag_pool_yields_no_match():
    # ``category_tags`` points to a real pool ("fruit") but the query
    # ("cracker") doesn't substring-match any fruit asset.  The resolver
    # relaxes to the full object pool and finds ``cracker_box``.
    items = [Item(query="cracker", category_tags=["fruit"])]
    compiler = _make_compiler()
    spec = compiler.compile(_make_scene(items=items))
    assert spec.nodes_by_id["cracker"].name == "cracker_box"
    trace_stages = [e.stage for e in compiler.trace]
    assert "item.preferred_tags.miss" in trace_stages
    assert any(s.startswith("item.required_tags") for s in trace_stages)


def test_item_relaxes_when_tag_pool_empty():
    # Unknown tag → empty pool → resolver relaxes to the required-tag pool.
    items = [Item(query="cracker", category_tags=["nonexistent"])]
    compiler = _make_compiler()
    spec = compiler.compile(_make_scene(items=items))
    assert spec.nodes_by_id["cracker"].name == "cracker_box"
    assert any(e.stage == "item.preferred_tags.empty_pool" for e in compiler.trace)


def test_item_miss_omits_node():
    # A query with no substring or fuzzy candidate is silently dropped;
    # the resolver records a trace event but never raises.
    items = [Item(query="zzz_no_match_anywhere", category_tags=["object"])]
    compiler = _make_compiler()
    spec = compiler.compile(_make_scene(items=items))
    assert "zzz_no_match_anywhere" not in spec.nodes_by_id
    assert any(e.stage == "item.required_tags.miss" for e in compiler.trace)


def test_item_instance_name_overrides_query_for_node_id():
    # ``instance_name`` becomes the graph node id; ``name`` still reflects the
    # resolved asset, allowing the same asset to appear under different ids.
    items = [Item(query="bowl", category_tags=["bowl"], instance_name="serving_bowl")]
    spec = _make_compiler().compile(_make_scene(items=items))
    assert "serving_bowl" in spec.nodes_by_id
    assert "bowl" not in spec.nodes_by_id
    assert spec.nodes_by_id["serving_bowl"].name == "bowl_ycb_robolab"


# =============================================================================
# Embodiment resolution (match_asset)
# =============================================================================


def test_embodiment_exact_match():
    # Node ID matches the original query so task params can reference it by
    # the agent-emitted name.
    compiler = _make_compiler()
    spec = compiler.compile(_make_scene(embodiment="franka_ik"))
    node = spec.nodes_by_id["franka_ik"]
    assert node.type == ArenaEnvGraphNodeType.EMBODIMENT
    assert node.name == "franka_ik"
    assert any(e.stage == "embodiment.exact" for e in compiler.trace)


def test_embodiment_joint_pos_not_fuzzy_matched_to_ik():
    # Exact hit on the required-tag pool must run before the ik-narrowed
    # preferred pool, so joint-position control is not fuzzy-matched to franka_ik.
    compiler = _make_compiler()
    spec = compiler.compile(_make_scene(embodiment="franka_joint_pos"))
    node = spec.nodes_by_id["franka_joint_pos"]
    assert node.type == ArenaEnvGraphNodeType.EMBODIMENT
    assert node.name == "franka_joint_pos"
    trace_stages = [e.stage for e in compiler.trace]
    assert "embodiment.exact" in trace_stages
    assert "embodiment.preferred_tags.fuzzy" not in trace_stages


def test_embodiment_ik_default_for_bare_family():
    # Bare family names (e.g. "franka") are expanded to the IK variant by
    # narrowing embodiment candidates by the "ik" tag and picking the shortest
    # match.  The node ID stays as the original query so downstream task params
    # that reference the robot by its original name resolve correctly.
    compiler = _make_compiler()
    spec = compiler.compile(_make_scene(embodiment="franka"))
    node = spec.nodes_by_id["franka"]  # ID = original query, not "franka_ik"
    assert node.type == ArenaEnvGraphNodeType.EMBODIMENT
    assert node.name == "franka_ik"  # name = resolved asset
    assert any(e.stage == "embodiment.preferred_tags.substring" for e in compiler.trace)


def test_embodiment_unknown_records_miss_and_omits_node():
    # Completely unknown names emit an "embodiment.required_tags.miss" trace
    # event and produce no embodiment node (no silent fallback).
    compiler = _make_compiler()
    spec = compiler.compile(_make_scene(embodiment="totally_unknown_robot"))
    assert not any(n.type == ArenaEnvGraphNodeType.EMBODIMENT for n in spec.nodes)
    assert any(e.stage == "embodiment.required_tags.miss" for e in compiler.trace)


# =============================================================================
# Background resolution (match_asset)
# =============================================================================


def test_background_with_wrong_tag_omitted():
    # A registered name that lacks the required background tag is not in the
    # background pool; the background node is absent from the resulting spec.
    assets = [
        FakeAsset(name="franka_ik", tags=["embodiment"]),
        FakeAsset(name="maple_table", tags=["object"]),  # wrong tag
    ]
    compiler = _make_compiler(assets)
    spec = compiler.compile(_make_scene(background="maple_table"))
    assert "maple_table" not in spec.nodes_by_id
    assert any(e.stage == "background.required_tags.empty_pool" for e in compiler.trace)


# =============================================================================
# IntentCompiler.compile() — graph wiring
# =============================================================================


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


# =============================================================================
# Resolution-error reporting
# =============================================================================


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
    # Tag-pool relaxation is not an error: the item still resolves via the
    # broader object pool.
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
    # An unknown embodiment with no fuzzy match emits "embodiment.required_tags.miss"
    # which is an error stage — no silent fallback to a hardcoded default.
    compiler = _make_compiler()
    compiler.compile(_make_scene(embodiment="totally_unknown_robot"))
    assert "embodiment.required_tags.miss" in [e.stage for e in compiler.trace]
    assert compiler.has_resolution_errors is True


# =============================================================================
# Spatial constraint construction
# =============================================================================


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


# =============================================================================
# Task spec construction
# =============================================================================


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
    compiler = _make_compiler()
    compiler.compile(_make_scene(items=items, tasks=tasks))
    assert compiler.has_resolution_errors is True
    assert "task.unknown_param" in [e.stage for e in compiler.resolution_errors]
