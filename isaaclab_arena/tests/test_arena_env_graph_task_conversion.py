# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for graph-task -> live-task conversion."""

from __future__ import annotations

import pytest

from isaaclab_arena.affordances.affordance_base import AffordanceBase
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.environments import arena_env_graph_task_conversion_utils as conversion


# An affordance is only ever mixed into an Asset; Placeable stands in for that contract.
class Placeable(AffordanceBase):
    pass


class _Task:
    """Task signature covering every annotation shape the resolver distinguishes."""

    def __init__(
        self,
        pick_up_object: Asset,  # scalar node ref
        targets: list[Asset],  # collection node ref
        placeable: Placeable,  # affordance node ref (matched via AffordanceBase)
        optional_object: Asset | None,  # optional node ref (matched via non-None branch)
        minimum_height: float,  # plain scalar — passes through
        robot_name: str,  # plain string (NOT a node id) — passes through
    ):
        pass


# --------------------------- find_node_ref_params_in_signature ---------------------------


def test_find_node_ref_params_classifies_each_shape():
    """Scalar/affordance/optional refs -> False, list ref -> True, plain params omitted."""
    result = conversion.find_node_ref_params_in_signature(_Task)
    assert result == {
        "pick_up_object": False,
        "targets": True,
        "placeable": False,
        "optional_object": False,
    }
    # Plain params are absent — membership is the "is this a node ref?" test the resolver uses.
    assert "minimum_height" not in result
    assert "robot_name" not in result


def test_find_node_ref_params_ignores_self_and_return():
    """The implicit ``self`` slot and a ``-> T`` return annotation are never kwargs."""

    class T:
        def __init__(self, obj: Asset) -> None:
            pass

    assert conversion.find_node_ref_params_in_signature(T) == {"obj": False}


def test_find_node_ref_params_rejects_unsupported_collection_shapes():
    """tuple[Asset, ...], bare list, and nested list[list[Asset]] are not node refs."""

    class T:
        def __init__(self, tup: tuple[Asset, ...], bare: list, nested: list[list[Asset]]):
            pass

    assert conversion.find_node_ref_params_in_signature(T) == {}


def test_find_node_ref_params_empty_when_no_refs():
    """A signature of only plain params yields an empty map."""

    class T:
        def __init__(self, height: float, name: str, count: int):
            pass

    assert conversion.find_node_ref_params_in_signature(T) == {}


# --------------------------------- _classify_node_ref ------------------------------------


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (Asset, False),
        (Placeable, False),
        (Asset | None, False),
        (list[Asset], True),
        (list[Placeable] | None, True),
        (list, None),
        (list[list[Asset]], None),
        (tuple[Asset, ...], None),
        (float, None),
        (str, None),
        (None, None),
    ],
)
def test_classify_node_ref(annotation, expected):
    """False = scalar ref, True = list ref, None = not a node ref."""
    assert conversion._classify_node_ref(annotation) is expected


# ----------------------------- _resolve_node_refs_in_task_args ---------------------------


def test_resolve_swaps_refs_and_forwards_plain_values():
    """Node-id strings become live assets; floats/strings pass through untouched."""
    cube, ball, mug = object(), object(), object()
    assets_by_node_id = {"cube": cube, "ball": ball, "mug": mug}
    raw_task_args = {
        "pick_up_object": "cube",
        "targets": ["cube", "ball"],
        "placeable": "mug",
        "optional_object": "ball",
        "minimum_height": 0.1,
        "robot_name": "franka",
    }

    resolved = conversion._resolve_node_refs_in_task_args(_Task, raw_task_args, assets_by_node_id)

    assert resolved["pick_up_object"] is cube
    assert resolved["targets"] == [cube, ball]
    assert resolved["placeable"] is mug
    assert resolved["optional_object"] is ball
    assert resolved["minimum_height"] == 0.1  # forwarded unchanged
    assert resolved["robot_name"] == "franka"  # a plain string, NOT looked up as a node id


def test_resolve_omits_node_ref_params_the_spec_did_not_supply():
    """Only supplied args appear; a node-ref param absent from the spec is not invented."""
    resolved = conversion._resolve_node_refs_in_task_args(_Task, {"minimum_height": 0.5}, {"cube": object()})
    assert resolved == {"minimum_height": 0.5}


def test_resolve_raises_on_unknown_node_id():
    """An unresolvable node id raises AssertionError naming the task and param."""
    with pytest.raises(AssertionError, match=r"_Task\.pick_up_object: unknown node id 'missing'"):
        conversion._resolve_node_refs_in_task_args(_Task, {"pick_up_object": "missing"}, {"cube": object()})


def test_resolve_raises_on_unknown_node_id_inside_collection():
    """A bad element inside a list[Asset] arg is caught too."""
    with pytest.raises(AssertionError, match=r"_Task\.targets: unknown node id 'nope'"):
        conversion._resolve_node_refs_in_task_args(_Task, {"targets": ["cube", "nope"]}, {"cube": object()})


def test_resolve_raises_on_non_string_node_id():
    """A non-string where a node id is expected is rejected (catches malformed specs)."""
    with pytest.raises(AssertionError, match="unknown node id"):
        conversion._resolve_node_refs_in_task_args(_Task, {"pick_up_object": 123}, {"cube": object()})


# --------------------------------- build_task_from_specs ---------------------------------


def test_build_task_from_specs_empty_returns_none():
    """No specs -> no task (the empty path short-circuits before any registry/sim lookup).

    The single-task and SequentialTaskBase-wrapping paths run real registered tasks, so they
    are covered by the end-to-end sim test in ``test_arena_env_graph_conversion.py``.
    """
    assert conversion.build_task_from_specs([], {}) is None
