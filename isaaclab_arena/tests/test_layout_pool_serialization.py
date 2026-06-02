# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pure layout-pool JSON serialization helpers (no simulation)."""

import pytest

from isaaclab_arena.relations.layout_pool_serialization import (
    PoolDocument,
    deserialize_layout,
    read_pool_document,
    serialize_layout,
    write_pool_document,
)
from isaaclab_arena.relations.placement_result import PlacementResult, ValidationReport


class _Obj:
    """Minimal stand-in for an ObjectBase: hashable, with a name."""

    def __init__(self, name: str):
        self.name = name


def _layout_dict():
    """A valid serialized single-layout dict for objects 'a' and 'b'."""
    return {
        "positions": {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]},
        "orientations": {"a": 0.5},
        "validation": {"no_overlap": True, "on_relations": True},
        "final_loss": 0.25,
        "attempts": 3,
    }


def _name_to_obj():
    return {"a": _Obj("a"), "b": _Obj("b")}


def _pool_dict(num_envs: int = 1):
    return {
        "placement_seed": 42,
        "num_envs": num_envs,
        "uses_env_specific_bboxes": False,
        "had_fallbacks": False,
        "env_pools": [[_layout_dict()] for _ in range(num_envs)],
    }


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_serialize_deserialize_layout_round_trip():
    name_to_obj = _name_to_obj()
    a, b = name_to_obj["a"], name_to_obj["b"]
    result = PlacementResult(
        positions={a: (1.0, 2.0, 3.0), b: (4.0, 5.0, 6.0)},
        orientations={a: 0.5},
        validation=ValidationReport(checks={"no_overlap": True}),
        final_loss=0.25,
        attempts=3,
    )

    restored = deserialize_layout(serialize_layout(result), name_to_obj)

    assert restored.positions == {a: (1.0, 2.0, 3.0), b: (4.0, 5.0, 6.0)}
    assert restored.orientations == {a: 0.5}
    assert dict(restored.validation.checks) == {"no_overlap": True}
    assert restored.final_loss == 0.25
    assert restored.attempts == 3


def test_pool_document_to_from_dict_round_trip():
    document = PoolDocument(
        placement_seed=7, num_envs=2, uses_env_specific_bboxes=True, had_fallbacks=True, env_pools=[[], []]
    )
    assert PoolDocument.from_dict(document.to_dict(), path="mem") == document


def test_write_then_read_round_trip(tmp_path):
    path = tmp_path / "pool.json"
    document = PoolDocument.from_dict(_pool_dict(num_envs=2), path=path)
    write_pool_document(path, document)
    assert read_pool_document(path) == document


# ---------------------------------------------------------------------------
# PoolDocument.from_dict structural guards
# ---------------------------------------------------------------------------


def test_from_dict_rejects_non_dict():
    with pytest.raises(AssertionError, match="not a JSON object"):
        PoolDocument.from_dict([1, 2, 3], path="mem")


def test_from_dict_rejects_missing_key():
    data = _pool_dict()
    del data["had_fallbacks"]
    with pytest.raises(AssertionError, match="missing required key 'had_fallbacks'"):
        PoolDocument.from_dict(data, path="mem")


@pytest.mark.parametrize(
    "key, value, message",
    [
        ("env_pools", {}, "'env_pools' must be a list"),
        ("num_envs", True, "'num_envs' must be an int"),
        ("num_envs", "x", "'num_envs' must be an int"),
        ("placement_seed", True, "'placement_seed' must be int or null"),
        ("placement_seed", 1.5, "'placement_seed' must be int or null"),
        ("uses_env_specific_bboxes", "no", "'uses_env_specific_bboxes' must be a bool"),
        ("had_fallbacks", 1, "'had_fallbacks' must be a bool"),
    ],
)
def test_from_dict_rejects_wrong_top_level_type(key, value, message):
    data = _pool_dict()
    data[key] = value
    with pytest.raises(AssertionError, match=message):
        PoolDocument.from_dict(data, path="mem")


def test_from_dict_rejects_num_envs_length_mismatch():
    data = _pool_dict(num_envs=1)
    data["num_envs"] = 2
    with pytest.raises(AssertionError, match="num_envs does not match env_pools length"):
        PoolDocument.from_dict(data, path="mem")


def test_from_dict_rejects_non_list_env_entry():
    data = _pool_dict(num_envs=1)
    data["env_pools"] = [_layout_dict()]
    with pytest.raises(AssertionError, match="env 0 must be a list"):
        PoolDocument.from_dict(data, path="mem")


def test_from_dict_rejects_non_dict_layout_entry():
    data = _pool_dict(num_envs=1)
    data["env_pools"] = [["not a dict"]]
    with pytest.raises(AssertionError, match="non-dict layout entry"):
        PoolDocument.from_dict(data, path="mem")


# ---------------------------------------------------------------------------
# write/read IO
# ---------------------------------------------------------------------------


def test_write_pool_document_rejects_non_finite_leaving_no_files(tmp_path):
    path = tmp_path / "pool.json"
    document = PoolDocument(
        placement_seed=0,
        num_envs=1,
        uses_env_specific_bboxes=False,
        had_fallbacks=False,
        env_pools=[[{"positions": {"a": [float("inf"), 0.0, 0.0]}}]],
    )
    with pytest.raises(ValueError, match="JSON compliant"):
        write_pool_document(path, document)
    assert not path.exists()
    assert not path.with_name(f"{path.name}.tmp").exists()


def test_read_pool_document_missing_file(tmp_path):
    with pytest.raises(AssertionError, match="not found"):
        read_pool_document(tmp_path / "missing.json")


def test_read_pool_document_malformed_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{ not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_pool_document(path)


# ---------------------------------------------------------------------------
# deserialize_layout leaf guards
# ---------------------------------------------------------------------------


def test_deserialize_rejects_missing_layout_key():
    data = _layout_dict()
    del data["final_loss"]
    with pytest.raises(AssertionError, match="missing required key 'final_loss'"):
        deserialize_layout(data, _name_to_obj())


@pytest.mark.parametrize(
    "key, value, message",
    [
        ("positions", [1, 2], "'positions' must be a dict"),
        ("orientations", [1, 2], "'orientations' must be a dict"),
        ("validation", [1, 2], "'validation' must be a dict"),
        ("validation", {}, "empty validation map"),
        ("final_loss", "x", "'final_loss' must be a finite number"),
        ("final_loss", float("nan"), "'final_loss' must be a finite number"),
        ("attempts", 1.5, "'attempts' must be an int"),
        ("attempts", True, "'attempts' must be an int"),
    ],
)
def test_deserialize_rejects_bad_leaf(key, value, message):
    data = _layout_dict()
    data[key] = value
    with pytest.raises(AssertionError, match=message):
        deserialize_layout(data, _name_to_obj())


def test_deserialize_rejects_non_bool_validation_value():
    data = _layout_dict()
    data["validation"] = {"no_overlap": "true"}
    with pytest.raises(AssertionError, match="must be a JSON bool"):
        deserialize_layout(data, _name_to_obj())


@pytest.mark.parametrize(
    "pos, message",
    [
        ([1.0, 2.0], "length-3 sequence"),
        ([1.0, 2.0, 3.0, 4.0], "length-3 sequence"),
        (["a", "b", "c"], "finite numbers"),
        ([float("inf"), 0.0, 0.0], "finite numbers"),
    ],
)
def test_deserialize_rejects_bad_position(pos, message):
    data = _layout_dict()
    data["positions"]["a"] = pos
    with pytest.raises(AssertionError, match=message):
        deserialize_layout(data, _name_to_obj())


def test_deserialize_rejects_non_finite_yaw():
    data = _layout_dict()
    data["orientations"]["a"] = float("nan")
    with pytest.raises(AssertionError, match="orientation for 'a' must be a finite number"):
        deserialize_layout(data, _name_to_obj())


def test_deserialize_rejects_object_set_mismatch_missing_live_object():
    data = _layout_dict()
    name_to_obj = {"a": _Obj("a"), "b": _Obj("b"), "c": _Obj("c")}
    with pytest.raises(AssertionError, match="do not match the provided objects"):
        deserialize_layout(data, name_to_obj)


def test_deserialize_rejects_unknown_saved_object():
    data = _layout_dict()
    with pytest.raises(AssertionError, match="do not match the provided objects"):
        deserialize_layout(data, {"a": _Obj("a")})


def test_deserialize_rejects_orientation_without_position():
    # positions == live objects {a, b}, but an orientation names 'z' which has no position.
    data = _layout_dict()
    data["orientations"]["z"] = 0.1
    with pytest.raises(AssertionError, match="not a subset of positions"):
        deserialize_layout(data, _name_to_obj())
