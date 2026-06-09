# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`isaaclab_arena.agentic_environment_generation.intent_resolver.IntentResolver`.

The resolver is pure Python — no Isaac Sim / Kit / pxr dependency — so these
tests run as plain pytest functions against an injected fake AssetRegistry.
They exercise the resolver's deterministic logic in isolation: asset binding
strategies (exact / substring / fuzzy / tag-pool relaxation / miss),
embodiment family defaults, spatial constraint construction (binary vs unary
relations, ``in`` skipping, unknown-node defensive traces), task spec wiring,
and trace lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab_arena.agentic_environment_generation.asset_resolver import IK_DEFAULTS
from isaaclab_arena.agentic_environment_generation.environment_intent_spec import EnvironmentIntentSpec, Item
from isaaclab_arena.agentic_environment_generation.intent_resolver import IntentResolver
from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphNodeType
from isaaclab_arena.environments.arena_env_graph_types import SpatialRelationSpec, TaskSpec

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeAsset:
    """Minimal stand-in for the asset classes the resolver inspects.

    Real asset classes are decorated classes pulled in via
    ``ensure_assets_registered()``. The resolver only ever reads ``.name`` and
    ``.tags`` off them, so a simple dataclass suffices and keeps the tests
    independent of isaaclab / Kit.
    """

    name: str
    tags: list[str]


class FakeAssetRegistry:
    """Duck-typed AssetRegistry for unit tests.

    Implements the four methods the resolver calls — ``is_registered``,
    ``get_asset_by_name``, ``get_assets_by_tag``, ``get_all_keys`` — without
    pulling in isaaclab. We deliberately don't subclass :class:`AssetRegistry`
    directly because it uses ``SingletonMeta``, which would force test
    isolation gymnastics. Duck-typing via the resolver's ``registry`` argument
    is the supported injection point.
    """

    def __init__(self, assets: list[FakeAsset]):
        self._by_name: dict[str, FakeAsset] = {a.name: a for a in assets}

    def is_registered(self, key: str) -> bool:
        return key in self._by_name

    def get_asset_by_name(self, name: str) -> FakeAsset:
        assert name in self._by_name, f"unregistered asset: {name}"
        return self._by_name[name]

    def get_assets_by_tag(self, tag: str) -> list[FakeAsset]:
        return [a for a in self._by_name.values() if tag in a.tags]

    def get_all_keys(self) -> list[str]:
        return list(self._by_name)


def _default_assets() -> list[FakeAsset]:
    """Small but representative catalog covering all three asset categories.

    Object names intentionally include the suffix conventions seen in the
    real registry (e.g. ``bowl_ycb_robolab``) so substring-match tests
    exercise realistic fuzzy/substring behaviour.
    """
    return [
        FakeAsset(name="maple_table", tags=["background"]),
        FakeAsset(name="franka_ik", tags=["embodiment"]),
        FakeAsset(name="franka_joint_pos", tags=["embodiment"]),
        FakeAsset(name="bowl_ycb_robolab", tags=["object", "bowl"]),
        FakeAsset(name="avocado01_fruits_robolab", tags=["object", "fruit"]),
        FakeAsset(name="apple01_fruits_robolab", tags=["object", "fruit"]),
        FakeAsset(name="cracker_box", tags=["object", "graspable"]),
    ]


def _make_resolver(assets: list[FakeAsset] | None = None) -> IntentResolver:
    return IntentResolver(registry=FakeAssetRegistry(assets or _default_assets()))


def _make_scene(
    *,
    background: str = "maple_table",
    embodiment: str = "franka_ik",
    items: list[Item] | None = None,
    initial_state_graph: list[SpatialRelationSpec] | None = None,
    tasks: list[TaskSpec] | None = None,
) -> EnvironmentIntentSpec:
    """Build an :class:`EnvironmentIntentSpec` with sane defaults for tests that don't care."""
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
    spec = _make_resolver().resolve(_make_scene(items=items, initial_state_graph=initial, tasks=tasks))

    # Auto-derived env_name: f"llm_gen_{background}_{first_task_kind}".
    assert spec.env_name == "llm_gen_maple_table_PickAndPlaceTask"

    # Node order: background, embodiment, items in declaration order.
    node_ids = [n.id for n in spec.nodes]
    assert node_ids == ["maple_table", "franka_ik", "bowl", "avocado"]
    assert spec.nodes_by_id["maple_table"].type == ArenaEnvGraphNodeType.BACKGROUND
    assert spec.nodes_by_id["franka_ik"].type == ArenaEnvGraphNodeType.EMBODIMENT
    # Item node.name reflects the *resolved* asset name, not the query.
    assert spec.nodes_by_id["bowl"].name == "bowl_ycb_robolab"
    assert spec.nodes_by_id["avocado"].name == "avocado01_fruits_robolab"

    # State specs: 1 initial + 1 success placeholder per task.
    assert len(spec.state_specs) == 2
    initial_state = spec.state_specs_by_id["state_initial"]
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

    # Tasks.
    assert len(spec.tasks) == 1
    task = spec.tasks_by_id["task_0_PickAndPlaceTask"]
    assert task.initial_state_spec_id == "state_initial"
    assert task.success_state_spec_id == "state_success_0"
    assert task.params == {
        "pick_up_object": "avocado",
        "destination_location": "bowl",
        "background_scene": "maple_table",
    }
    assert task.description == "put avocado in bowl"


def test_resolve_overrides_env_name():
    resolver = _make_resolver()
    spec = resolver.resolve(_make_scene(), env_name="my_custom_env")
    assert spec.env_name == "my_custom_env"


def test_resolve_clears_trace_between_calls():
    resolver = _make_resolver()
    resolver.resolve(_make_scene())
    n_after_first = len(resolver.trace)
    # Sanity: at least background, embodiment, and task events should be present.
    assert n_after_first > 0

    resolver.resolve(_make_scene())
    n_after_second = len(resolver.trace)
    # If trace persisted across calls, the second count would be > the first.
    # Deterministic input → identical trace length when the trace is cleared.
    assert n_after_second == n_after_first


def test_resolve_with_empty_initial_state_graph():
    spec = _make_resolver().resolve(_make_scene(initial_state_graph=[]))
    initial_state = spec.state_specs_by_id["state_initial"]
    assert initial_state.spatial_constraints == []
    # Even with no constraints, the spec should still be well-formed.
    assert initial_state.task_constraints == []


# ---------------------------------------------------------------------------
# Resolution-error reporting
# ---------------------------------------------------------------------------


def _clean_scene_kwargs() -> dict:
    """Scene where every node resolves and every task arg references a known node.

    The default ``_make_scene`` uses a "placeholder" task subject/target that
    deliberately doesn't resolve — fine for tests that only care about node
    counts but unsuitable for resolution-error tests where we need a baseline
    where the resolver succeeds completely.
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
    # Fully resolvable env; no error-bearing trace events should appear.
    resolver = _make_resolver()
    resolver.resolve(_make_scene(**_clean_scene_kwargs()))
    assert resolver.resolution_errors == []
    assert resolver.has_resolution_errors is False


def test_has_resolution_errors_true_when_item_unresolvable():
    # Add an unresolvable item on top of the clean baseline so the *only*
    # error stage that fires is ``item.miss``.
    kwargs = _clean_scene_kwargs()
    kwargs["items"] = kwargs["items"] + [Item(query="zzz_no_match_anywhere", category_tags=["object"])]
    resolver = _make_resolver()
    resolver.resolve(_make_scene(**kwargs))
    assert resolver.has_resolution_errors is True
    assert [e.stage for e in resolver.resolution_errors] == ["item.miss"]


def test_has_resolution_errors_false_when_only_relaxation_or_fallback():
    # Both events the bot's heuristic would mistakenly flag: tag-pool
    # relaxation (successful) and embodiment fallback (franka_ik). Neither
    # drops data from the resolved spec, so neither should count as an
    # error. ``cracker`` is in the default catalog but tagged ``graspable``,
    # not ``fruit`` — so the fruit-tag pool yields no match and the resolver
    # relaxes to the full object pool.
    kwargs = _clean_scene_kwargs()
    kwargs["items"] = [Item(query="cracker", category_tags=["fruit"])]
    # Switch the task subject to match the new item id so task args still resolve.
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
    resolver = _make_resolver()
    resolver.resolve(_make_scene(embodiment="totally_unknown_robot", **kwargs))
    trace_stages = [e.stage for e in resolver.trace]
    # Sanity: the warning events actually fired in this run.
    assert "item.no_match_in_tags" in trace_stages
    assert "embodiment.miss" in trace_stages
    # But neither shows up as an error.
    assert resolver.has_resolution_errors is False
    assert resolver.resolution_errors == []


# ---------------------------------------------------------------------------
# Item resolution strategies
# ---------------------------------------------------------------------------


def test_item_exact_name_match():
    # Query that's already a registered asset name skips fuzzy matching.
    items = [Item(query="cracker_box", category_tags=["graspable"])]
    resolver = _make_resolver()
    spec = resolver.resolve(_make_scene(items=items))
    assert spec.nodes_by_id["cracker_box"].name == "cracker_box"
    assert any(e.stage == "item.exact" for e in resolver.trace)


def test_item_substring_match_in_tag_pool():
    items = [Item(query="bowl", category_tags=["bowl"])]
    resolver = _make_resolver()
    spec = resolver.resolve(_make_scene(items=items))
    assert spec.nodes_by_id["bowl"].name == "bowl_ycb_robolab"
    assert any(e.stage == "item.in_tags.substring" for e in resolver.trace)


def test_item_relaxes_when_tag_pool_yields_no_match():
    # category_tags points to a real tag pool ('fruit') but the query
    # ('cracker') doesn't substring-match either fruit. The resolver should
    # relax to the full object pool and find cracker_box.
    items = [Item(query="cracker", category_tags=["fruit"])]
    resolver = _make_resolver()
    spec = resolver.resolve(_make_scene(items=items))
    assert spec.nodes_by_id["cracker"].name == "cracker_box"
    trace_stages = [e.stage for e in resolver.trace]
    assert "item.no_match_in_tags" in trace_stages
    assert any(s.startswith("item.relaxed") for s in trace_stages)


def test_item_relaxes_when_tag_pool_empty():
    # Unknown tag → empty tag pool → resolver short-circuits the pool-search
    # and relaxes immediately.
    items = [Item(query="cracker", category_tags=["nonexistent"])]
    resolver = _make_resolver()
    spec = resolver.resolve(_make_scene(items=items))
    assert spec.nodes_by_id["cracker"].name == "cracker_box"
    assert any(e.stage == "item.tag_pool_empty" for e in resolver.trace)


def test_item_miss_omits_node():
    # Query that matches no asset (and no substring / fuzzy candidate) is
    # silently dropped — the resolver records a trace but doesn't raise.
    items = [Item(query="zzz_no_match_anywhere", category_tags=["object"])]
    resolver = _make_resolver()
    spec = resolver.resolve(_make_scene(items=items))
    assert "zzz_no_match_anywhere" not in spec.nodes_by_id
    assert any(e.stage == "item.miss" for e in resolver.trace)


def test_item_instance_name_overrides_query_for_node_id():
    items = [Item(query="bowl", category_tags=["bowl"], instance_name="serving_bowl")]
    spec = _make_resolver().resolve(_make_scene(items=items))
    # ``instance_name`` controls the *node id*; ``name`` still reflects the
    # resolved asset, so the same asset can appear twice under different ids.
    assert "serving_bowl" in spec.nodes_by_id
    assert "bowl" not in spec.nodes_by_id
    assert spec.nodes_by_id["serving_bowl"].name == "bowl_ycb_robolab"


# ---------------------------------------------------------------------------
# Embodiment resolution
# ---------------------------------------------------------------------------


def test_embodiment_exact_match():
    spec = _make_resolver().resolve(_make_scene(embodiment="franka_joint_pos"))
    assert spec.nodes_by_id["franka_joint_pos"].type == ArenaEnvGraphNodeType.EMBODIMENT


def test_embodiment_ik_default_for_bare_family():
    resolver = _make_resolver()
    spec = resolver.resolve(_make_scene(embodiment="franka"))
    # The mapping is exported from the resolver so callers can introspect it.
    assert IK_DEFAULTS["franka"] == "franka_ik"
    assert spec.nodes_by_id["franka_ik"].type == ArenaEnvGraphNodeType.EMBODIMENT
    assert any(e.stage == "embodiment.ik_default" for e in resolver.trace)


def test_embodiment_unknown_falls_back_to_franka_ik():
    # Unknown family names never raise — they fall back to franka_ik and
    # record a miss trace. ``franka_ik`` must therefore be registered.
    resolver = _make_resolver()
    spec = resolver.resolve(_make_scene(embodiment="totally_unknown_robot"))
    assert spec.nodes_by_id["franka_ik"].type == ArenaEnvGraphNodeType.EMBODIMENT
    assert any(e.stage == "embodiment.miss" for e in resolver.trace)


# ---------------------------------------------------------------------------
# Background resolution
# ---------------------------------------------------------------------------


def test_background_with_wrong_tag_omitted():
    # An asset registered under the name "maple_table" but NOT tagged
    # "background" is rejected with a name.wrong_tag trace, so the background
    # node is absent from the resulting spec.
    assets = [
        FakeAsset(name="franka_ik", tags=["embodiment"]),
        FakeAsset(name="maple_table", tags=["object"]),  # wrong tag
    ]
    resolver = _make_resolver(assets)
    spec = resolver.resolve(_make_scene(background="maple_table"))
    assert "maple_table" not in spec.nodes_by_id
    assert any(e.stage == "name.wrong_tag" for e in resolver.trace)


# ---------------------------------------------------------------------------
# Spatial constraint construction
# ---------------------------------------------------------------------------


def test_spatial_constraint_binary_relation_id_and_parent_child():
    items = [Item(query="cracker_box", category_tags=["graspable"])]
    initial = [SpatialRelationSpec(kind="on", subject="cracker_box", reference="maple_table")]
    spec = _make_resolver().resolve(_make_scene(items=items, initial_state_graph=initial))
    constraint = spec.state_specs_by_id["state_initial"].spatial_constraints[0]
    assert constraint.reference == "maple_table"
    assert constraint.subject == "cracker_box"
    assert constraint.id == "state_initial_0_on_maple_table_cracker_box"


def test_spatial_constraint_unary_relation_id_and_parent_child():
    initial = [SpatialRelationSpec(kind="is_anchor", subject="maple_table")]
    spec = _make_resolver().resolve(_make_scene(initial_state_graph=initial))
    constraint = spec.state_specs_by_id["state_initial"].spatial_constraints[0]
    assert constraint.kind == "is_anchor"
    assert constraint.subject == "maple_table"
    assert constraint.reference is None
    assert constraint.id == "state_initial_0_is_anchor_maple_table"


def test_spatial_constraint_unknown_subject_skipped():
    initial = [SpatialRelationSpec(kind="on", subject="not_a_node", reference="maple_table")]
    resolver = _make_resolver()
    spec = resolver.resolve(_make_scene(initial_state_graph=initial))
    assert spec.state_specs_by_id["state_initial"].spatial_constraints == []
    assert any(e.stage == "relation.initial.unknown_subject" for e in resolver.trace)


def test_spatial_constraint_unknown_parent_skipped():
    initial = [SpatialRelationSpec(kind="on", subject="maple_table", reference="missing_node")]
    resolver = _make_resolver()
    spec = resolver.resolve(_make_scene(initial_state_graph=initial))
    assert spec.state_specs_by_id["state_initial"].spatial_constraints == []
    assert any(e.stage == "relation.initial.unknown_parent" for e in resolver.trace)


def test_spatial_constraint_params_passed_through():
    items = [Item(query="cracker_box", category_tags=["graspable"])]
    initial = [
        SpatialRelationSpec(
            kind="at_position",
            subject="cracker_box",
            params={"position_xyz": [0.1, 0.2, 0.3]},
        ),
    ]
    spec = _make_resolver().resolve(_make_scene(items=items, initial_state_graph=initial))
    constraint = spec.state_specs_by_id["state_initial"].spatial_constraints[0]
    assert constraint.kind == "at_position"
    # ``params`` are passed through verbatim — the resolver doesn't validate
    # the schema of relation-kind-specific params; that's the downstream
    # builder's job.
    assert constraint.params == {"position_xyz": (0.1, 0.2, 0.3)}


# ---------------------------------------------------------------------------
# Task spec construction
# ---------------------------------------------------------------------------


def test_multiple_tasks_get_distinct_success_state_ids():
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
    spec = _make_resolver().resolve(_make_scene(items=items, tasks=tasks))

    # Task ids follow ``task_{index}_{kind}``.
    task_ids = [t.id for t in spec.tasks]
    assert task_ids == ["task_0_PickAndPlaceTask", "task_1_OpenDoorTask", "task_2_CloseDoorTask"]

    # Each task points at its own per-task placeholder success state.
    success_ids = [t.success_state_spec_id for t in spec.tasks]
    assert success_ids == ["state_success_0", "state_success_1", "state_success_2"]

    # state_specs contains 1 initial + 3 placeholder success specs.
    assert len(spec.state_specs) == 4
    for i in range(3):
        # Placeholders are empty — downstream synthesis is responsible for them.
        assert spec.state_specs_by_id[f"state_success_{i}"].spatial_constraints == []
        assert spec.state_specs_by_id[f"state_success_{i}"].task_constraints == []
