# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the mesh-only In relation: container-cavity containment via the placement solver."""

from __future__ import annotations

import trimesh

import pytest

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_validation import PlacementCheck
from isaaclab_arena.relations.relation_solver_params import CollisionMode, RelationSolverParams
from isaaclab_arena.relations.relations import In, IsAnchor
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

try:
    import warp as wp

    wp.init()
    _WARP_AVAILABLE = True
except Exception:
    _WARP_AVAILABLE = False

requires_warp = pytest.mark.skipif(not _WARP_AVAILABLE, reason="Warp not available")

# Cavity proxy: a watertight box interior of half-extents (0.20, 0.20, 0.15) centered on the container.
_CAVITY_HALF = (0.20, 0.20, 0.15)
_CONTAINER_POS = (1.0, 0.5, 0.8)
_CHILD_HALF = 0.05


def _make_container() -> DummyObject:
    cavity = trimesh.creation.box(extents=(2 * _CAVITY_HALF[0], 2 * _CAVITY_HALF[1], 2 * _CAVITY_HALF[2]))
    container = DummyObject(
        name="container",
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-_CAVITY_HALF[0], -_CAVITY_HALF[1], -_CAVITY_HALF[2]),
            max_point=(_CAVITY_HALF[0], _CAVITY_HALF[1], _CAVITY_HALF[2]),
        ),
        cavity_mesh=cavity,
    )
    container.set_initial_pose(Pose(position_xyz=_CONTAINER_POS, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    container.add_relation(IsAnchor())
    return container


def _make_child_for(container: DummyObject) -> DummyObject:
    child = DummyObject(
        name="child",
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-_CHILD_HALF, -_CHILD_HALF, -_CHILD_HALF),
            max_point=(_CHILD_HALF, _CHILD_HALF, _CHILD_HALF),
        ),
    )
    child.add_relation(In(container))
    return child


def _mesh_placer() -> ObjectPlacer:
    solver_params = RelationSolverParams(collision_mode=CollisionMode.MESH, verbose=False)
    return ObjectPlacer(
        params=ObjectPlacerParams(solver_params=solver_params, placement_seed=3, apply_positions_to_objects=False)
    )


@requires_warp
def test_in_places_child_inside_cavity():
    """An In child is solved fully inside the container's cavity proxy and validates."""
    container = _make_container()
    child = _make_child_for(container)

    (result,) = _mesh_placer().place([container, child])
    failed = result.validation_results.get_failed_validation_check_names
    assert result.success, f"expected valid In placement, failed checks: {failed}"
    assert result.validation_results.validation_results[PlacementCheck.IN_RELATION]

    x, y, z = result.positions[child]
    # Child bbox must sit fully within the cavity: center within (cavity_half - child_half) of the container.
    assert abs(x - _CONTAINER_POS[0]) <= _CAVITY_HALF[0] - _CHILD_HALF + 1e-3
    assert abs(y - _CONTAINER_POS[1]) <= _CAVITY_HALF[1] - _CHILD_HALF + 1e-3
    assert abs(z - _CONTAINER_POS[2]) <= _CAVITY_HALF[2] - _CHILD_HALF + 1e-3


@requires_warp
def test_in_rejects_bbox_mode():
    """In is mesh-only: solving with CollisionMode.BBOX fails loud."""
    container = _make_container()
    child = _make_child_for(container)
    placer = ObjectPlacer(
        params=ObjectPlacerParams(
            solver_params=RelationSolverParams(collision_mode=CollisionMode.BBOX, verbose=False),
            apply_positions_to_objects=False,
        )
    )
    with pytest.raises(AssertionError, match="mesh-only"):
        placer.place([container, child])


@requires_warp
def test_in_rejects_tilted_container():
    """In requires an identity or pure-Z container: a roll/pitch pose fails loud."""
    container = _make_container()
    # 90 deg about X -> non-zero qx, disallowed in MESH mode.
    container.set_initial_pose(Pose(position_xyz=_CONTAINER_POS, rotation_xyzw=(0.7071068, 0.0, 0.0, 0.7071068)))
    child = _make_child_for(container)
    with pytest.raises(AssertionError, match="pure-Z rotation"):
        _mesh_placer().place([container, child])


@requires_warp
def test_in_with_container_on_table():
    """The RoboLab case: the container itself is optimizable (bowl On table); the item still lands inside it."""
    from isaaclab_arena.relations.relations import On

    table = DummyObject(
        name="table",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.5, -0.5, 0.0), max_point=(0.5, 0.5, 0.04)),
    )
    table.set_initial_pose(Pose(position_xyz=(1.0, 0.5, 0.7), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())

    cavity = trimesh.creation.box(extents=(2 * _CAVITY_HALF[0], 2 * _CAVITY_HALF[1], 2 * _CAVITY_HALF[2]))
    bowl = DummyObject(
        name="bowl",
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-_CAVITY_HALF[0], -_CAVITY_HALF[1], -_CAVITY_HALF[2]),
            max_point=(_CAVITY_HALF[0], _CAVITY_HALF[1], _CAVITY_HALF[2]),
        ),
        cavity_mesh=cavity,
    )
    bowl.add_relation(On(table))  # optimizable container, not an anchor

    item = DummyObject(
        name="item",
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-_CHILD_HALF, -_CHILD_HALF, -_CHILD_HALF),
            max_point=(_CHILD_HALF, _CHILD_HALF, _CHILD_HALF),
        ),
    )
    item.add_relation(In(bowl))

    (result,) = _mesh_placer().place([table, bowl, item])
    assert result.success, f"failed checks: {result.validation_results.get_failed_validation_check_names}"
    bx, by, bz = result.positions[bowl]
    ix, iy, iz = result.positions[item]
    # Item bbox fully within the bowl cavity, measured relative to the bowl's own solved position.
    assert abs(ix - bx) <= _CAVITY_HALF[0] - _CHILD_HALF + 1e-3
    assert abs(iy - by) <= _CAVITY_HALF[1] - _CHILD_HALF + 1e-3
    assert abs(iz - bz) <= _CAVITY_HALF[2] - _CHILD_HALF + 1e-3


def _open_top_box_shell(extents: tuple[float, float, float]) -> trimesh.Trimesh:
    """A box with its top (+Z) faces removed — a non-watertight container shell to be capped."""
    shell = trimesh.creation.box(extents=extents)
    top_faces = shell.triangles.mean(axis=1)[:, 2] > (extents[2] / 2 - 1e-6)
    shell.update_faces(~top_faces)
    shell.remove_unreferenced_vertices()
    return shell


@requires_warp
def test_in_derives_cavity_from_container_mesh():
    """With no explicit proxy, In caps the container's open-top collision mesh into a watertight cavity."""
    shell = _open_top_box_shell((2 * _CAVITY_HALF[0], 2 * _CAVITY_HALF[1], 2 * _CAVITY_HALF[2]))
    assert not shell.is_watertight, "shell should be open before capping"
    container = DummyObject(
        name="container",
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-_CAVITY_HALF[0], -_CAVITY_HALF[1], -_CAVITY_HALF[2]),
            max_point=(_CAVITY_HALF[0], _CAVITY_HALF[1], _CAVITY_HALF[2]),
        ),
        collision_mesh=shell,  # no cavity_mesh: In must derive it from this shell
    )
    container.set_initial_pose(Pose(position_xyz=_CONTAINER_POS, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    container.add_relation(IsAnchor())
    child = _make_child_for(container)

    (result,) = _mesh_placer().place([container, child])
    assert result.success, f"failed checks: {result.validation_results.get_failed_validation_check_names}"
    x, y, z = result.positions[child]
    assert abs(x - _CONTAINER_POS[0]) <= _CAVITY_HALF[0] - _CHILD_HALF + 1e-3
    assert abs(y - _CONTAINER_POS[1]) <= _CAVITY_HALF[1] - _CHILD_HALF + 1e-3
    assert abs(z - _CONTAINER_POS[2]) <= _CAVITY_HALF[2] - _CHILD_HALF + 1e-3


@requires_warp
def test_in_falls_back_to_bounding_box_cavity():
    """When the container mesh can't be capped watertight, In uses a bounding-box cavity.

    A container with no explicit proxy and no cappable collision mesh (here: no collision mesh at all)
    falls back to a box the size of its bounding box, so a box-like container still places the child
    inside — matching the real bins/crates/pails whose shell meshes don't cap cleanly.
    """
    container = DummyObject(
        name="container",
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-_CAVITY_HALF[0], -_CAVITY_HALF[1], -_CAVITY_HALF[2]),
            max_point=(_CAVITY_HALF[0], _CAVITY_HALF[1], _CAVITY_HALF[2]),
        ),
    )
    container.set_initial_pose(Pose(position_xyz=_CONTAINER_POS, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    container.add_relation(IsAnchor())
    child = _make_child_for(container)

    (result,) = _mesh_placer().place([container, child])
    assert result.success, f"failed checks: {result.validation_results.get_failed_validation_check_names}"
    x, y, z = result.positions[child]
    assert abs(x - _CONTAINER_POS[0]) <= _CAVITY_HALF[0] - _CHILD_HALF + 1e-3
    assert abs(y - _CONTAINER_POS[1]) <= _CAVITY_HALF[1] - _CHILD_HALF + 1e-3
    assert abs(z - _CONTAINER_POS[2]) <= _CAVITY_HALF[2] - _CHILD_HALF + 1e-3


@requires_warp
def test_in_reproducible_with_seed():
    """Same placement_seed produces identical In placements across independent runs."""

    def _run():
        container = _make_container()
        child = _make_child_for(container)
        (result,) = _mesh_placer().place([container, child])
        return next(pos for obj, pos in result.positions.items() if obj.name == "child")

    assert _run() == _run()
