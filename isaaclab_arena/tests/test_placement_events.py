# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for placement-on-reset event: fresh layouts on successive resets."""

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_result import MultiEnvPlacementResult
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, NextTo, On, Side
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


def _create_test_objects():
    """Create a desk (anchor) with two boxes (On + NextTo)."""

    desk = DummyObject(
        name="desk",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())

    box1 = DummyObject(
        name="box1",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )
    box2 = DummyObject(
        name="box2",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.15, 0.15, 0.15)),
    )

    box1.add_relation(On(desk, clearance_m=0.01))
    box2.add_relation(On(desk, clearance_m=0.01))
    box2.add_relation(NextTo(box1, side=Side.POSITIVE_X, distance_m=0.05))

    return desk, box1, box2


def test_successive_placements_without_seed_produce_different_layouts():
    """Two place() calls with placement_seed=None should produce different positions."""

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3)
    params = ObjectPlacerParams(
        solver_params=solver_params,
        apply_positions_to_objects=False,
        placement_seed=None,
    )

    desk1, box1_a, box2_a = _create_test_objects()
    placer_a = ObjectPlacer(params=params)
    result_a = placer_a.place([desk1, box1_a, box2_a], num_envs=1)

    desk2, box1_b, box2_b = _create_test_objects()
    placer_b = ObjectPlacer(params=params)
    result_b = placer_b.place([desk2, box1_b, box2_b], num_envs=1)

    any_different = False
    for obj_a, obj_b in zip([box1_a, box2_a], [box1_b, box2_b]):
        if result_a.positions[obj_a] != result_b.positions[obj_b]:
            any_different = True
            break
    assert any_different, "Two unseeded placements should produce different layouts"


def test_placement_without_seed_multi_env_gives_different_layouts():
    """Multi-env placement with seed=None should give distinct per-env layouts."""

    num_envs = 4
    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3)
    params = ObjectPlacerParams(
        solver_params=solver_params,
        apply_positions_to_objects=False,
        placement_seed=None,
    )

    desk, box1, box2 = _create_test_objects()
    placer = ObjectPlacer(params=params)
    result = placer.place([desk, box1, box2], num_envs=num_envs)

    assert isinstance(result, MultiEnvPlacementResult)
    positions_box1 = [result.results[e].positions[box1] for e in range(num_envs)]
    any_different = any(positions_box1[i] != positions_box1[j] for i in range(num_envs) for j in range(i + 1, num_envs))
    assert any_different, "Unseeded multi-env placement should produce different positions across environments"


def test_successive_seeded_placements_produce_same_layout():
    """Two place() calls with the same seed should produce identical positions."""

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3)
    params = ObjectPlacerParams(
        solver_params=solver_params,
        apply_positions_to_objects=False,
        placement_seed=42,
    )

    desk1, box1_a, box2_a = _create_test_objects()
    placer_a = ObjectPlacer(params=params)
    result_a = placer_a.place([desk1, box1_a, box2_a], num_envs=1)

    desk2, box1_b, box2_b = _create_test_objects()
    placer_b = ObjectPlacer(params=params)
    result_b = placer_b.place([desk2, box1_b, box2_b], num_envs=1)

    for obj_a, obj_b in zip([box1_a, box2_a], [box1_b, box2_b]):
        assert (
            result_a.positions[obj_a] == result_b.positions[obj_b]
        ), f"Same seed should produce identical results for {obj_a.name}"
