# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""Clutter releases through Arena's ordinary object placer."""

import math
import torch

import pytest

from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_events import get_pose_from_layout
from isaaclab_arena.relations.relations import ClutterOn, IsAnchor, On, RotateAroundSolution
from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


def _scene():
    support = DummyObject(
        "table", AxisAlignedBoundingBox((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)), Pose.identity(), [IsAnchor()]
    )
    objects = [
        DummyObject(
            f"box_{i}",
            AxisAlignedBoundingBox((-0.03, -0.03, -0.02), (0.03, 0.03, 0.02)),
            relations=[ClutterOn(support, spread=0.08, random_yaw=False)],
        )
        for i in range(3)
    ]
    return support, objects


def test_clutter_on_produces_a_valid_column_through_object_placer():
    support, objects = _scene()
    placer = ObjectPlacer(
        ObjectPlacerParams(placement_seed=7, max_placement_attempts=1, apply_positions_to_objects=False)
    )
    result = placer.place([support, *objects], num_envs=2)
    for layout in result:
        assert layout.success
        bottoms = [layout.positions[obj][2] - 0.02 for obj in objects]
        assert bottoms == pytest.approx([0.11, 0.18, 0.25], abs=1e-5)
        for obj in objects:
            assert abs(layout.positions[obj][0]) <= 0.01 + 1e-6
            assert abs(layout.positions[obj][1]) <= 0.01 + 1e-6


@pytest.mark.parametrize("ceiling_bottom,expected_z", [(1.0, 0.13), (0.12, 0.27)])
def test_clutter_release_uses_the_first_free_vertical_interval(ceiling_bottom, expected_z):
    support, objects = _scene()
    ceiling = DummyObject(
        "ceiling", AxisAlignedBoundingBox((-1, -1, 0), (1, 1, 0.1)), Pose((0, 0, ceiling_bottom)), [IsAnchor()]
    )
    layout = ObjectPlacer(ObjectPlacerParams(max_placement_attempts=1)).place([support, ceiling, objects[0]])[0]
    assert layout.success
    assert layout.positions[objects[0]][2] == pytest.approx(expected_z, abs=1e-5)


@pytest.mark.parametrize("random_yaw", [True, False])
def test_clutter_yaw_composes_with_a_tilted_marker(random_yaw):
    from isaaclab_arena.utils.yaw import yaw_from_quat_xyzw

    support, objects = _scene()
    obj = objects[0]
    marker = RotateAroundSolution(roll_rad=0.4, pitch_rad=0.3, yaw_rad=0.7)
    obj.relations = [ClutterOn(support, spread=0.5, random_yaw=random_yaw), marker]
    layout = ObjectPlacer(ObjectPlacerParams(placement_seed=5, max_placement_attempts=1)).place([support, obj])[0]
    pose = get_pose_from_layout(obj, layout)
    base = Pose(rotation_xyzw=marker.get_rotation_xyzw()).to_transform_matrix("cpu")[:3, :3]
    rotation = pose.to_transform_matrix("cpu")[:3, :3]
    delta = rotation @ base.T
    assert float(torch.linalg.det(rotation)) == pytest.approx(1.0)
    torch.testing.assert_close(delta[2], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6, rtol=0)
    assert yaw_from_quat_xyzw(pose.rotation_xyzw) == pytest.approx(layout.orientations[obj], abs=1e-6)
    torch.testing.assert_close(obj.get_initial_pose().to_tensor("cpu"), pose.to_tensor("cpu"))
    if random_yaw:
        assert abs(float(delta[0, 1])) > 0.01
    else:
        torch.testing.assert_close(rotation, base, atol=1e-6, rtol=0)


def test_clutter_on_rejects_an_oversized_member():
    support, objects = _scene()
    objects[0].relations = [ClutterOn(support, spread=0.01)]
    placer = ObjectPlacer(ObjectPlacerParams(max_placement_attempts=1, allow_best_loss_fallbacks=False))
    with pytest.raises(AssertionError, match="on_relation"):
        placer.place([support, objects[0]])
    assert objects[0].get_initial_pose() is None


def test_clutter_on_requires_a_fixed_support():
    support, objects = _scene()
    support.relations = []
    with pytest.raises(AssertionError, match="IsAnchor support"):
        objects[0].relations[0].validate_placement_configuration(objects[0], {support, objects[0]})


@pytest.mark.parametrize("relation_type", [On, ClutterOn])
def test_release_respects_a_rotated_offset_support(relation_type):
    support, objects = _scene()
    support.bounding_box = AxisAlignedBoundingBox((0.0, 0.0, 0.0), (1.0, 0.2, 0.1))
    support.set_initial_pose(Pose((2.0, 3.0, 0.0), (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))))
    obj = objects[0]
    relation = ClutterOn(support, spread=1.0, random_yaw=False) if relation_type is ClutterOn else On(support)
    obj.relations = [relation]
    layout = ObjectPlacer(ObjectPlacerParams(max_placement_attempts=1, placement_seed=4)).place([support, obj])[0]
    assert layout.success
    x, y, _ = layout.positions[obj]
    assert 1.83 <= x <= 1.97
    assert 3.03 <= y <= 3.97


def test_clutter_on_composes_with_a_requested_release_position():
    from isaaclab_arena.relations.relations import AtPosition

    support, objects = _scene()
    obj = objects[0]
    obj.relations = [ClutterOn(support, spread=1.0, random_yaw=False), AtPosition(x=0.2, y=0.1, z=0.6)]
    placer = ObjectPlacer(
        ObjectPlacerParams(placement_seed=4, max_placement_attempts=1, allow_best_loss_fallbacks=False)
    )
    layout = placer.place([support, obj])[0]
    assert layout.success
    assert layout.positions[obj] == pytest.approx((0.2, 0.1, 0.6), abs=0.005)
    assert obj.get_initial_pose().position_xyz == layout.positions[obj]


@pytest.mark.parametrize("relation_type", [On, ClutterOn])
@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("allow_fallback", [False, True])
def test_failure_policy_is_shared_by_relations_and_placers(relation_type, pooled, allow_fallback):
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import AtPosition

    support, objects = _scene()
    obj = objects[0]
    obj.relations = [relation_type(support), AtPosition(x=5.0)]
    params = ObjectPlacerParams(
        placement_seed=3,
        max_placement_attempts=1,
        allow_best_loss_fallbacks=allow_fallback,
        solver_params=RelationSolverParams(max_iters=1, lr=2.0, verbose=False),
    )

    def place():
        if pooled:
            return PooledObjectPlacer([support, obj], params, pool_size=1, num_envs=1).sample_without_replacement(1)
        return ObjectPlacer(params).place([support, obj])

    if allow_fallback:
        assert not place()[0].success
    else:
        with pytest.raises(RuntimeError if pooled else AssertionError):
            place()
        assert obj.get_initial_pose() is None


@pytest.mark.parametrize("max_iters", [0, 1, 3])
def test_reported_loss_matches_the_returned_clutter_position(max_iters):
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import AtPosition

    support, objects = _scene()
    obj = objects[0]
    obj.relations = [ClutterOn(support, random_yaw=False), AtPosition(z=0.8)]
    params = ObjectPlacerParams(
        max_placement_attempts=1,
        placement_seed=2,
        solver_params=RelationSolverParams(max_iters=max_iters, verbose=False),
    )
    placer = ObjectPlacer(params)
    layout = placer.place([support, obj])[0]
    assert layout.success
    assert placer.last_loss_history[-1] == layout.final_loss
    expected = 100.0 * abs(layout.positions[obj][2] - 0.8)
    assert layout.final_loss == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("relation_type", [On, ClutterOn])
@pytest.mark.parametrize("support_width", [0.12, 0.32])
def test_fixed_yaw_uses_the_rotated_footprint(relation_type, support_width):
    support, objects = _scene()
    support.bounding_box = AxisAlignedBoundingBox((-0.2, -support_width / 2, 0), (0.2, support_width / 2, 0.1))
    obj = objects[0]
    obj.bounding_box = AxisAlignedBoundingBox((-0.15, -0.025, -0.02), (0.15, 0.025, 0.02))
    relation = (
        ClutterOn(support, spread=1, random_yaw=False) if relation_type is ClutterOn else On(support, edge_margin_m=0)
    )
    obj.relations = [relation, RotateAroundSolution(yaw_rad=math.pi / 4)]
    placer = ObjectPlacer(
        ObjectPlacerParams(placement_seed=0, max_placement_attempts=1, allow_best_loss_fallbacks=False)
    )
    if support_width == 0.12:
        with pytest.raises(AssertionError, match="on_relation"):
            placer.place([support, obj])
        assert obj.get_initial_pose() is None
    else:
        layout = placer.place([support, obj])[0]
        assert layout.success
        pose = get_pose_from_layout(obj, layout)
        bounds = obj.get_bounding_box().rotated_by_quat(pose.rotation_xyzw).translated(pose.position_xyz)
        assert bool((bounds.min_point[0, :2] >= support.bounding_box.min_point[0, :2]).all())
        assert bool((bounds.max_point[0, :2] <= support.bounding_box.max_point[0, :2]).all())


@pytest.mark.parametrize("attempts", [1, 10])
def test_unfit_random_yaw_does_not_discard_valid_candidates(attempts):
    support, objects = _scene()
    support.bounding_box = AxisAlignedBoundingBox((-0.08, -0.04, 0), (0.08, 0.04, 0.1))
    obj = objects[0]
    obj.bounding_box = AxisAlignedBoundingBox((-0.07, -0.02, -0.02), (0.07, 0.02, 0.02))
    obj.relations = [ClutterOn(support, spread=1)]
    placer = ObjectPlacer(
        ObjectPlacerParams(
            placement_seed=6,
            max_placement_attempts=attempts,
            apply_positions_to_objects=False,
            allow_best_loss_fallbacks=False,
        )
    )
    assert placer.place([support, obj])[0].success


def test_ranked_clutter_keeps_failed_candidates_after_valid_candidates():
    support, objects = _scene()
    support.bounding_box = AxisAlignedBoundingBox((-0.08, -0.04, 0), (0.08, 0.04, 0.1))
    obj = objects[0]
    obj.bounding_box = AxisAlignedBoundingBox((-0.07, -0.02, -0.02), (0.07, 0.02, 0.02))
    obj.relations = [ClutterOn(support, spread=1)]
    placer = ObjectPlacer(ObjectPlacerParams(max_placement_attempts=1, placement_seed=6))
    ranked = placer.place_ranked_per_env([support, obj], num_envs=1, results_per_env=10)[0]
    valid = [layout.success for layout in ranked]
    assert any(valid) and not all(valid)
    assert valid == sorted(valid, reverse=True)


@pytest.mark.parametrize("name", ["gap_m", "clearance_m", "edge_margin_m", "spread"])
@pytest.mark.parametrize("value", [-0.01, float("inf"), float("nan")])
def test_clutter_on_rejects_invalid_parameters(name, value):
    support, _ = _scene()
    with pytest.raises(AssertionError):
        ClutterOn(support, **{name: value})


def test_clutter_on_rejects_anchor_members_and_post_solve_randomization():
    from isaaclab_arena.relations.relations import RandomAroundSolution

    support, objects = _scene()
    obj = objects[0]
    for marker, message in ((IsAnchor(), "cannot be an anchor"), (RandomAroundSolution(), "cannot randomize")):
        obj.relations = [ClutterOn(support), marker]
        with pytest.raises(AssertionError, match=message):
            ObjectPlacer().place([support, obj])


def test_disjoint_clutter_releases_stay_at_surface_height():
    from isaaclab_arena.relations.relations import AtPosition

    support, objects = _scene()
    for obj, x in zip(objects, (-0.3, 0.0, 0.3), strict=True):
        obj.relations = [ClutterOn(support, spread=1, random_yaw=False), AtPosition(x=x, y=0.0)]
    placer = ObjectPlacer(ObjectPlacerParams(max_placement_attempts=1))
    positions = {support: (0, 0, 0), **{obj: (x, 0, 1) for obj, x in zip(objects, (-0.3, 0.0, 0.3), strict=True)}}
    placer._initialize_clutter_positions(positions, {obj: obj.get_bounding_box() for obj in positions}, [])
    assert [positions[obj][2] for obj in objects] == pytest.approx([0.13] * 3)


@pytest.mark.parametrize("batched", [False, True])
def test_clutter_loss_penalizes_only_footprint_and_lower_height_violations(batched):
    from isaaclab_arena.relations.relation_loss_strategies import ClutterOnLossStrategy

    support, objects = _scene()
    relation = ClutterOn(support, spread=0.2, random_yaw=False, relation_loss_weight=2)
    positions = torch.tensor([[0.0, 0.0, 0.5], [0.1, 0.0, 0.1]], requires_grad=True)
    strategy = ClutterOnLossStrategy()
    if batched:
        losses = strategy.compute_loss(relation, positions, objects[0].get_bounding_box(), support.get_bounding_box())
    else:
        losses = torch.stack([
            strategy.compute_loss(relation, position, objects[0].get_bounding_box(), support.get_bounding_box())
            for position in positions
        ])
    torch.testing.assert_close(losses, torch.tensor([0.0, 1.2]), atol=1e-6, rtol=0)
    losses.sum().backward()
    torch.testing.assert_close(positions.grad, torch.tensor([[0.0, 0.0, 0.0], [20.0, 0.0, -20.0]]))


@pytest.mark.parametrize("quarters", [1, 3])
def test_anchor_bounds_match_fixed_world_bounds(quarters):
    from isaaclab_arena.relations.bounding_box_helpers import build_per_env_bounding_boxes

    support, _ = _scene()
    support.bounding_box = AxisAlignedBoundingBox((0, 0, 0), (1, 0.2, 0.1))
    yaw = quarters * math.pi / 2
    support.set_initial_pose(Pose((2, 3, 0), (0, 0, math.sin(yaw / 2), math.cos(yaw / 2))))
    bounds = build_per_env_bounding_boxes([support], 2).object_bboxes[support].translated((2, 3, 0))
    expected = support.get_world_bounding_box()
    torch.testing.assert_close(bounds.min_point, expected.min_point.expand(2, 3))
    torch.testing.assert_close(bounds.max_point, expected.max_point.expand(2, 3))


def test_unsupported_anchor_rotation_names_the_asset():
    support, objects = _scene()
    support.set_initial_pose(Pose(rotation_xyzw=RotateAroundSolution(yaw_rad=math.pi / 4).get_rotation_xyzw()))
    with pytest.raises(AssertionError, match="Anchor 'table'"):
        ObjectPlacer().place([support, *objects])


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -1.0])
def test_clutter_rejects_invalid_loss_weights(weight):
    support, _ = _scene()
    with pytest.raises(AssertionError, match="relation_loss_weight"):
        ClutterOn(support, relation_loss_weight=weight)


@pytest.mark.parametrize("random_yaw", ["false", 0, None])
def test_clutter_requires_a_boolean_yaw_setting(random_yaw):
    support, _ = _scene()
    with pytest.raises(AssertionError, match="random_yaw must be a boolean"):
        ClutterOn(support, random_yaw=random_yaw)


@pytest.mark.parametrize("relation_type", [On, ClutterOn])
@pytest.mark.parametrize("allow_fallback", [False, True])
@pytest.mark.parametrize("invalid", ["position", "loss"])
def test_nonfinite_solver_output_is_never_applied(monkeypatch, relation_type, allow_fallback, invalid):
    support, objects = _scene()
    obj = objects[0]
    placer = ObjectPlacer(ObjectPlacerParams(max_placement_attempts=1, allow_best_loss_fallbacks=allow_fallback))
    obj.relations = [relation_type(support)]
    solve = placer._solver.solve

    def nonfinite_solve(*args, **kwargs):
        results = solve(*args, **kwargs)
        if invalid == "position":
            results[0][obj] = (float("nan"), 0, 1)
        else:
            placer._solver.last_loss_per_env[0] = float("inf")
        return results

    monkeypatch.setattr(placer._solver, "solve", nonfinite_solve)
    with pytest.raises(AssertionError, match="Non-finite solver output for environment 0, candidate 0"):
        placer.place([support, obj])
    assert obj.get_initial_pose() is None


def test_clutter_release_stays_inside_a_room_mesh():
    import trimesh

    from isaaclab_arena.relations.background_collision_object import FixedCollisionObject

    support, objects = _scene()
    floor = trimesh.creation.box(extents=(2, 2, 0.1))
    floor.apply_translation((0, 0, -0.05))
    ceiling = trimesh.creation.box(extents=(2, 2, 0.1))
    ceiling.apply_translation((0, 0, 1.05))
    room = FixedCollisionObject(trimesh.util.concatenate([floor, ceiling]))
    placer = ObjectPlacer(
        ObjectPlacerParams(max_placement_attempts=1, allow_best_loss_fallbacks=False, apply_positions_to_objects=False)
    )
    layout = placer.place([support, *objects], collision_objects=[room])[0]
    assert layout.success
    assert [layout.positions[obj][2] for obj in objects] == pytest.approx([0.13, 0.20, 0.27], abs=1e-5)


def test_release_rejects_an_incomplete_layout_before_writing():
    from unittest.mock import MagicMock

    from isaaclab_arena.relations.clutter.settle import _release_objects

    support, objects = _scene()
    assets = [support, *objects]
    layout = ObjectPlacer(ObjectPlacerParams(max_placement_attempts=1, apply_positions_to_objects=False)).place(assets)[
        0
    ]
    del layout.positions[objects[1]]
    env = MagicMock()
    with pytest.raises(AssertionError, match="missing non-anchor assets.*box_1"):
        _release_objects(env, 0, layout, {support}, assets)
    assert not env.mock_calls


@pytest.mark.parametrize("collision_mode", ["bbox", "mesh"])
def test_clutter_release_keeps_meshless_collision_bounds(collision_mode):
    from isaaclab_arena.relations.collision_mode import CollisionMode

    support, objects = _scene()
    obstacle = DummyObject("obstacle", AxisAlignedBoundingBox((-1, -1, 0), (1, 1, 0.1)), Pose((0, 0, 0.12)))
    obstacle.collision_mode = CollisionMode(collision_mode)
    placer = ObjectPlacer(ObjectPlacerParams(max_placement_attempts=1, allow_best_loss_fallbacks=False))
    layout = placer.place([support, objects[0]], collision_objects=[obstacle])[0]
    assert layout.success
    assert layout.positions[objects[0]][2] == pytest.approx(0.27, abs=1e-5)


@pytest.mark.parametrize("kind,valid", [("on_overlap", True), ("on_contained", False), ("clutter_on", False)])
def test_overhang_policy_agrees_between_loss_and_validation(kind, valid):
    from isaaclab_arena.relations.placement_validators import OnRelationValidator
    from isaaclab_arena.relations.relation_loss_strategies import ClutterOnLossStrategy, OnLossStrategy

    support, objects = _scene()
    obj = objects[0]
    if kind == "clutter_on":
        relation = ClutterOn(support, spread=1, random_yaw=False)
        strategy = ClutterOnLossStrategy()
    else:
        relation = On(support, overlap=kind == "on_overlap", edge_margin_m=0)
        strategy = OnLossStrategy()
    obj.relations = [relation]
    position = (0.49, 0.0, 0.13)
    bounds = {asset: asset.get_bounding_box() for asset in (support, obj)}
    loss = strategy.compute_loss(relation, torch.tensor(position), bounds[obj], bounds[support])
    assert float(loss) == pytest.approx(0.0 if valid else 0.2, abs=1e-6)
    validator = OnRelationValidator(ObjectPlacerParams())
    assert validator.validate_batch([{support: (0, 0, 0), obj: position}], [{}], [bounds], []) == [valid]
