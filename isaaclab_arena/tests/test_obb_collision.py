# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Candidate collision validation with oriented and heterogeneous bounding boxes."""

import math
import trimesh

import pytest

from isaaclab_arena.relations.bounding_box_helpers import update_candidate_bounds
from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_candidate_batch import PlacementCandidate, PlacementCandidateBatch
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import ClutterOn, IsAnchor, NextTo, On, RotateAroundSolution, Side
from isaaclab_arena.relations.validation.pre_physics import NoOverlapValidator
from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


def _validator() -> NoOverlapValidator:
    return NoOverlapValidator(
        ObjectPlacerParams(solver_params=RelationSolverParams(clearance_m=0.0, verbose=False), verbose=False)
    )


def test_validator_preserves_oriented_geometry_and_candidate_environment_rows():
    """A wider assigned member collides where two narrow parallel rods clear despite overlapping AABBs."""
    thin = AxisAlignedBoundingBox((-0.4, -0.025, -0.03), (0.4, 0.025, 0.03))
    wide = AxisAlignedBoundingBox((-0.4, -0.15, -0.03), (0.4, 0.15, 0.03))
    first = DummyObject("first", thin)
    second = DummyObject("second", thin)
    per_env = [{first: thin, second: thin}, {first: thin, second: wide}]
    orientations = {first: math.pi / 4, second: math.pi / 4}
    separated = {first: (0.0, 0.0, 0.0), second: (-0.1, 0.1, 0.0)}
    overlapping = {first: (0.0, 0.0, 0.0), second: (-0.015, 0.015, 0.0)}
    batch = PlacementCandidateBatch([
        PlacementCandidate(1, 0, separated, orientations, per_env[1]),
        PlacementCandidate(0, 0, separated, orientations, per_env[0]),
        PlacementCandidate(0, 1, overlapping, orientations, per_env[0]),
    ])
    update_candidate_bounds(batch, per_env)
    validator = _validator()

    enclosing_only = PlacementCandidateBatch([batch.candidates[1]])
    assert validator.validate_batch(enclosing_only, []) == [False]
    assert validator.validate_batch(batch, []) == [False, True, False]
    selected = batch.select([1, 0])
    assert selected.collision_bboxes[second].half_extents.equal(batch.collision_bboxes[second].half_extents[[1, 0]])
    assert validator.validate_batch(selected, []) == [True, False]
    assert validator.validate_batch(batch.select([]), []) == []


def test_validator_rotates_off_origin_centers_with_tilted_markers():
    """Marker-only pitch rotates both the axes and center of each collision box."""
    bounds = AxisAlignedBoundingBox((0.5, -0.025, -0.025), (1.5, 0.025, 0.025))
    first = DummyObject("first", bounds, relations=[RotateAroundSolution(pitch_rad=math.pi / 4)])
    cube = AxisAlignedBoundingBox((-0.025, -0.025, -0.025), (0.025, 0.025, 0.025))
    second = DummyObject("second", cube)
    base_boxes = {first: bounds, second: cube}
    candidates = []
    center = 1 / math.sqrt(2)
    for index, distance in enumerate((0.15, 0.03)):
        offset = distance / math.sqrt(2)
        positions = {first: (0.0, 0.0, 0.0), second: (center + offset, 0.0, -center + offset)}
        candidates.append(PlacementCandidate(0, index, positions, {}, base_boxes))
    batch = PlacementCandidateBatch(candidates)
    update_candidate_bounds(batch, [base_boxes])

    assert _validator().validate_batch(batch, []) == [True, False]


def test_validator_without_collision_bounds_falls_back_to_enclosing_bounds():
    """A selected batch without collision bounds validates the enclosing AABBs, which reach a cleared obstacle."""
    bounds = AxisAlignedBoundingBox((-0.4, -0.025, -0.03), (0.4, 0.025, 0.03))
    rod = DummyObject("rod", bounds)
    obstacle = DummyObject(
        "obstacle",
        AxisAlignedBoundingBox((-0.025, -0.025, -0.025), (0.025, 0.025, 0.025)),
        initial_pose=Pose(position_xyz=(-0.15, 0.15, 0.0)),
    )
    batch = PlacementCandidateBatch([
        PlacementCandidate(0, 0, {rod: (0.0, 0.0, 0.0)}, {rod: math.pi / 4}, {rod: bounds}),
    ])
    update_candidate_bounds(batch, [{rod: bounds}])
    validator = _validator()

    assert validator.validate_batch(batch, [obstacle]) == [True]
    batch.collision_bboxes = None
    assert validator.validate_batch(batch.select([0]), [obstacle]) == [False]


@pytest.mark.parametrize(
    "make_relation, exempt",
    [(On, True), (ClutterOn, True), (lambda parent: NextTo(parent, side=Side.POSITIVE_X), False)],
    ids=["On", "ClutterOn", "NextTo"],
)
def test_validator_exempts_only_on_and_clutter_on_parent_pairs(make_relation, exempt):
    """On and ClutterOn exempt contact with their parent; other parent relations such as NextTo must clear."""
    bounds = AxisAlignedBoundingBox((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1))
    support = DummyObject("support", bounds, relations=[IsAnchor()])
    child = DummyObject("child", bounds, relations=[make_relation(support)])
    positions = {support: (0.0, 0.0, 0.0), child: (0.0, 0.0, 0.0)}
    boxes = {support: bounds, child: bounds}
    batch = PlacementCandidateBatch([PlacementCandidate(0, 0, positions, {}, boxes)])
    update_candidate_bounds(batch, [boxes])

    assert _validator().validate_batch(batch, []) == [exempt]


def test_validator_exempts_fixed_fixed_pairs():
    """Overlapping anchors and passive obstacles do not invalidate a fixed layout."""
    bounds = AxisAlignedBoundingBox((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1))
    first = DummyObject("first", bounds, initial_pose=Pose.identity(), relations=[IsAnchor()])
    second = DummyObject("second", bounds, initial_pose=Pose.identity(), relations=[IsAnchor()])
    passive = DummyObject("passive", bounds, initial_pose=Pose.identity())
    positions = {first: (0.0, 0.0, 0.0), second: (0.0, 0.0, 0.0)}
    boxes = {first: bounds, second: bounds}
    batch = PlacementCandidateBatch([PlacementCandidate(0, 0, positions, {}, boxes)])
    update_candidate_bounds(batch, [boxes])

    assert _validator().validate_batch(batch, [passive]) == [True]


def test_validator_missing_mesh_uses_oriented_box_fallback():
    """A MESH override without an extractable mesh keeps the same OBB verdicts as BBOX mode."""
    bounds = AxisAlignedBoundingBox((-0.4, -0.025, -0.03), (0.4, 0.025, 0.03))
    first = DummyObject("first", bounds)
    second = DummyObject("second", bounds)
    first.collision_mode = CollisionMode.MESH
    base_boxes = {first: bounds, second: bounds}
    orientations = {first: math.pi / 4, second: math.pi / 4}
    batch = PlacementCandidateBatch([
        PlacementCandidate(0, 0, {first: (0.0, 0.0, 0.0), second: (-0.1, 0.1, 0.0)}, orientations, base_boxes),
        PlacementCandidate(0, 1, {first: (0.0, 0.0, 0.0), second: (0.0, 0.0, 0.0)}, orientations, base_boxes),
    ])
    update_candidate_bounds(batch, [base_boxes])

    assert _validator().validate_batch(batch, []) == [True, False]


def test_validator_keeps_mesh_pairs_on_mesh_path():
    """Disjoint meshes remain valid even if their conservative proxy boxes overlap."""
    bounds = AxisAlignedBoundingBox((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    mesh = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
    first = DummyObject("first", bounds, collision_mesh=mesh)
    second = DummyObject("second", bounds, collision_mesh=mesh.copy())
    first.collision_mode = second.collision_mode = CollisionMode.MESH
    boxes = {first: bounds, second: bounds}
    batch = PlacementCandidateBatch([
        PlacementCandidate(0, 0, {first: (0.0, 0.0, 0.0), second: (0.3, 0.0, 0.0)}, {}, boxes),
        PlacementCandidate(0, 1, {first: (0.0, 0.0, 0.0), second: (0.0, 0.0, 0.0)}, {}, boxes),
    ])
    update_candidate_bounds(batch, [boxes])

    assert _validator().validate_batch(batch, []) == [True, False]


@pytest.mark.parametrize("fixed_obstacle", [True, False])
def test_mixed_mesh_fallback_penalizes_layouts_rejected_by_enclosing_box_proxy(fixed_obstacle):
    """Varying meshless boxes use enclosing-box penalties consistent with mesh-proxy validation."""
    table = DummyObject(
        "table",
        AxisAlignedBoundingBox((-2.0, -2.0, -0.1), (2.0, 2.0, 0.0)),
        initial_pose=Pose.identity(),
        relations=[IsAnchor()],
    )
    rod = DummyObject("rod", AxisAlignedBoundingBox((-0.4, -0.02, -0.05), (0.4, 0.02, 0.05)))
    obstacle = DummyObject(
        "mesh_obstacle",
        AxisAlignedBoundingBox((-0.06, -0.06, -0.06), (0.06, 0.06, 0.06)),
        initial_pose=Pose(position_xyz=(0.18, -0.18, 1.0)),
        collision_mesh=trimesh.creation.box(extents=(0.12, 0.12, 0.12)),
    )
    obstacle.collision_mode = CollisionMode.MESH
    objects = [table, rod]
    positions = {table: (0.0, 0.0, 0.0), rod: (0.0, 0.0, 1.0)}
    collision_objects = [obstacle] if fixed_obstacle else []
    if not fixed_obstacle:
        objects.append(obstacle)
        positions[obstacle] = obstacle.get_initial_pose().position_xyz
    boxes = {obj: obj.get_bounding_box() for obj in objects}
    batch = PlacementCandidateBatch([
        PlacementCandidate(0, index, dict(positions), {rod: yaw}, boxes) for index, yaw in enumerate((0.0, math.pi / 4))
    ])
    update_candidate_bounds(batch, [boxes])
    params = RelationSolverParams(max_iters=0, clearance_m=0.0, num_spheres=300, verbose=False)
    solver = RelationSolver(params)

    solver.solve_candidates(objects, batch, collision_objects)

    assert solver._mesh_cache is None
    assert batch.candidates[0].loss == 0.0
    directed_pairs = 1 if fixed_obstacle else 2
    assert batch.candidates[1].loss == pytest.approx(
        directed_pairs * solver._no_collision_strategy.slope * 0.12 * 0.12 * 0.1, rel=1e-5
    )
    validator = NoOverlapValidator(ObjectPlacerParams(solver_params=params, verbose=False))
    assert validator.validate_batch(batch, collision_objects) == [True, False]

    # Pure box mode can use the tighter oriented rod geometry for the same candidates.
    obstacle.collision_mode = CollisionMode.BBOX
    solver.solve_candidates(objects, batch, collision_objects)
    assert [candidate.loss for candidate in batch.candidates] == [0.0, 0.0]
    assert validator.validate_batch(batch, collision_objects) == [True, True]


def test_meshed_movable_uses_oriented_boxes_against_meshless_passive_obstacle():
    """A meshed rod and meshless passive obstacle share the same box geometry in loss and validation."""
    table = DummyObject(
        "table",
        AxisAlignedBoundingBox((-2.0, -2.0, -0.1), (2.0, 2.0, 0.0)),
        initial_pose=Pose.identity(),
        relations=[IsAnchor()],
    )
    rod = DummyObject(
        "rod",
        AxisAlignedBoundingBox((-0.4, -0.02, -0.05), (0.4, 0.02, 0.05)),
        collision_mesh=trimesh.creation.box(extents=(0.8, 0.04, 0.1)),
    )
    rod.collision_mode = CollisionMode.MESH
    obstacle = DummyObject(
        "passive",
        AxisAlignedBoundingBox((-0.06, -0.06, -0.06), (0.06, 0.06, 0.06)),
        initial_pose=Pose(position_xyz=(0.18, -0.18, 1.0)),
    )
    obstacle.collision_mode = CollisionMode.BBOX
    objects = [table, rod]
    boxes = {obj: obj.get_bounding_box() for obj in objects}
    batch = PlacementCandidateBatch([
        PlacementCandidate(0, index, {table: (0.0, 0.0, 0.0), rod: position}, {rod: math.pi / 4}, boxes)
        for index, position in enumerate(((0.0, 0.0, 1.0), obstacle.get_initial_pose().position_xyz))
    ])
    update_candidate_bounds(batch, [boxes])
    params = RelationSolverParams(max_iters=0, clearance_m=0.0, verbose=False)
    solver = RelationSolver(params)
    validator = NoOverlapValidator(ObjectPlacerParams(solver_params=params, verbose=False))

    solver.solve_candidates(objects, batch, [obstacle])

    mesh_losses = [candidate.loss for candidate in batch.candidates]
    assert mesh_losses[0] == 0.0
    assert mesh_losses[1] > 0.0
    assert validator.validate_batch(batch, [obstacle]) == [True, False]
    rod.collision_mode = CollisionMode.BBOX
    solver.solve_candidates(objects, batch, [obstacle])
    assert [candidate.loss for candidate in batch.candidates] == pytest.approx(mesh_losses)
    assert validator.validate_batch(batch, [obstacle]) == [True, False]
