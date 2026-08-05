# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sim-free tests for declaring clutter in a spec and turning a layout into poses.

These cover the seams between clutter and the surrounding placement machinery: the spec
parser that builds the relation, and the pose path that carries a settled rotation into the
scene. Both were previously exercised only by running the feature.
"""

from __future__ import annotations

import math

import pytest

from isaaclab_arena.relations.placement_events import get_pose_from_layout
from isaaclab_arena.relations.relations import ClutteredOn, RotateAroundSolution


class _Asset:
    """Minimal stand-in exposing only what the pose path reads."""

    def __init__(self, name: str):
        self.name = name
        self.relations: list = []

    def add_relation(self, relation) -> None:
        self.relations.append(relation)

    def get_relations(self) -> list:
        return self.relations


class _Layout:
    """Stands in for PlacementResult, which carries validation machinery these do not use."""

    def __init__(self):
        self.positions: dict = {}
        self.rotations: dict = {}
        self.orientations: dict = {}


def _yaw_quat(degrees: float) -> tuple[float, float, float, float]:
    half = math.radians(degrees) * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


# ------------------------------------------------------------------ declaring in a spec


def test_spec_kind_cluttered_on_builds_the_relation():
    """The documented YAML form must produce a ClutteredOn bound to its reference asset."""
    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import _attach_spatial_relations_to_assets
    from isaaclab_arena.environment_spec.arena_env_graph_types import SpatialRelationSpec

    table, mug = _Asset("table"), _Asset("mug")
    relation_spec = SpatialRelationSpec(
        kind="cluttered_on",
        subject="mug",
        reference="table",
        params={"group": "tools", "spread": 0.8, "gap_m": 0.02},
    )

    _attach_spatial_relations_to_assets([relation_spec], {"table": table, "mug": mug})

    (relation,) = mug.get_relations()
    assert isinstance(relation, ClutteredOn)
    assert relation.parent is table
    assert relation.group == "tools"
    assert relation.spread == pytest.approx(0.8)
    assert relation.gap_m == pytest.approx(0.02)
    assert table.get_relations() == [], "the support gains nothing from a member's relation"


def test_spec_rejects_cluttered_on_params_the_relation_would_reject():
    """Invalid params must fail while building the relation, not silently later."""
    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import _attach_spatial_relations_to_assets
    from isaaclab_arena.environment_spec.arena_env_graph_types import SpatialRelationSpec

    table, mug = _Asset("table"), _Asset("mug")
    relation_spec = SpatialRelationSpec(kind="cluttered_on", subject="mug", reference="table", params={"spread": 1.5})

    with pytest.raises(AssertionError, match="spread must be in"):
        _attach_spatial_relations_to_assets([relation_spec], {"table": table, "mug": mug})


# --------------------------------------------------------- a layout rotation reaching sim


def test_pose_uses_a_full_rotation_when_the_layout_carries_one():
    """A settled rotation is already final, so it is used as-is."""
    asset = _Asset("mug")
    layout = _Layout()
    layout.positions[asset] = (0.1, 0.2, 0.3)
    layout.rotations[asset] = _yaw_quat(30.0)

    pose = get_pose_from_layout(asset, layout)
    assert pose.position_xyz == (0.1, 0.2, 0.3)
    assert pose.rotation_xyzw == pytest.approx(_yaw_quat(30.0))


def test_a_full_rotation_wins_over_a_scalar_yaw():
    """Both present means the layout settled the object; the yaw is the stale one."""
    asset = _Asset("mug")
    layout = _Layout()
    layout.positions[asset] = (0.0, 0.0, 0.0)
    layout.orientations[asset] = math.radians(90.0)
    layout.rotations[asset] = _yaw_quat(30.0)

    pose = get_pose_from_layout(asset, layout)
    assert pose.rotation_xyzw == pytest.approx(_yaw_quat(30.0))


def test_a_full_rotation_ignores_the_marker_rotation():
    """A marker composes with a solved yaw, but a settled rotation is already in world terms."""
    asset = _Asset("mug")
    asset.add_relation(RotateAroundSolution(yaw_rad=math.radians(45.0)))
    layout = _Layout()
    layout.positions[asset] = (0.0, 0.0, 0.0)
    layout.rotations[asset] = _yaw_quat(30.0)

    pose = get_pose_from_layout(asset, layout)
    assert pose.rotation_xyzw == pytest.approx(_yaw_quat(30.0)), "marker yaw must not be composed in"


def test_a_scalar_yaw_still_composes_with_its_marker():
    """Layouts without a full rotation keep the existing yaw-composition behaviour."""
    asset = _Asset("mug")
    asset.add_relation(RotateAroundSolution(yaw_rad=math.radians(45.0)))
    layout = _Layout()
    layout.positions[asset] = (0.0, 0.0, 0.0)
    layout.orientations[asset] = math.radians(45.0)

    pose = get_pose_from_layout(asset, layout)
    assert pose.rotation_xyzw == pytest.approx(_yaw_quat(45.0))


def test_pose_requires_the_asset_to_be_in_the_layout():
    with pytest.raises(AssertionError, match="missing non-anchor asset"):
        get_pose_from_layout(_Asset("absent"), _Layout())
