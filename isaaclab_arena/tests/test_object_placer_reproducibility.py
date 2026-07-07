# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ObjectPlacer and RelationSolver reproducibility."""

import math

import pytest

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_result import PlacementResult
from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, NextTo, On, RotateAroundSolution, Side
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, get_random_pose_within_bounding_box
from isaaclab_arena.utils.pose import Pose, PosePerEnv, rotate_quat_by_yaw, wrap_angle_to_pi


def _create_test_objects():
    """Create test objects with relations (without setting initial poses for non-anchors)."""

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


def test_get_random_pose_same_seed_produces_identical_result():
    """Test that get_random_pose_within_bounding_box with same seed produces identical poses."""

    bbox = AxisAlignedBoundingBox(min_point=(-1.0, -1.0, 0.0), max_point=(1.0, 1.0, 1.0))

    pose1 = get_random_pose_within_bounding_box(bbox, seed=42)
    pose2 = get_random_pose_within_bounding_box(bbox, seed=42)

    assert pose1.position_xyz == pose2.position_xyz


def test_relation_solver_same_inputs_produces_identical_result():
    """Test that RelationSolver with identical initial positions produces identical results."""

    desk_pos = (0.0, 0.0, 0.0)
    fixed_box1_pos = (0.5, 0.5, 0.5)
    fixed_box2_pos = (0.3, 0.7, 0.3)
    solver_params = RelationSolverParams(max_iters=10)

    # Run 1
    desk1, box1_run1, box2_run1 = _create_test_objects()
    initial_positions1 = {desk1: desk_pos, box1_run1: fixed_box1_pos, box2_run1: fixed_box2_pos}

    solver1 = RelationSolver(params=solver_params)
    result1 = solver1.solve(objects=[desk1, box1_run1, box2_run1], initial_positions=[initial_positions1])[0]

    # Run 2 (fresh objects, same initial positions)
    desk2, box1_run2, box2_run2 = _create_test_objects()
    initial_positions2 = {desk2: desk_pos, box1_run2: fixed_box1_pos, box2_run2: fixed_box2_pos}

    solver2 = RelationSolver(params=solver_params)
    result2 = solver2.solve(objects=[desk2, box1_run2, box2_run2], initial_positions=[initial_positions2])[0]

    # Compare by name (different object instances)
    for obj1 in result1:
        pos1 = result1[obj1]
        pos2 = next(result2[obj2] for obj2 in result2 if obj2.name == obj1.name)
        assert pos1 == pos2, f"Mismatch for {obj1.name}: {pos1} != {pos2}"


def test_object_placer_same_seed_produces_identical_result():
    """Test that ObjectPlacer with same seed produces identical final results."""

    seed = 42
    solver_params = RelationSolverParams(max_iters=10)

    # Run 1
    desk1, box1_run1, box2_run1 = _create_test_objects()
    objects1 = [desk1, box1_run1, box2_run1]
    placer1 = ObjectPlacer(params=ObjectPlacerParams(placement_seed=seed, solver_params=solver_params))
    (result1,) = placer1.place(objects=objects1)

    # Run 2
    desk2, box1_run2, box2_run2 = _create_test_objects()
    objects2 = [desk2, box1_run2, box2_run2]
    placer2 = ObjectPlacer(params=ObjectPlacerParams(placement_seed=seed, solver_params=solver_params))
    (result2,) = placer2.place(objects=objects2)

    # Compare by name
    for obj1, obj2 in zip(objects1, objects2):
        pos1 = result1.positions[obj1]
        pos2 = result2.positions[obj2]
        assert pos1 == pos2, f"Mismatch for {obj1.name}: {pos1} != {pos2}"


def test_object_placer_different_seeds_produce_different_results():
    """Test that ObjectPlacer with different seeds produces different results."""

    solver_params = RelationSolverParams(max_iters=10)

    # Run 1 with seed 42
    desk1, box1_run1, box2_run1 = _create_test_objects()
    objects1 = [desk1, box1_run1, box2_run1]
    placer1 = ObjectPlacer(params=ObjectPlacerParams(placement_seed=42, solver_params=solver_params))
    (result1,) = placer1.place(objects=objects1)

    # Run 2 with seed 123
    desk2, box1_run2, box2_run2 = _create_test_objects()
    objects2 = [desk2, box1_run2, box2_run2]
    placer2 = ObjectPlacer(params=ObjectPlacerParams(placement_seed=123, solver_params=solver_params))
    (result2,) = placer2.place(objects=objects2)

    # Check that at least one non-anchor position differs
    any_different = False
    for obj1, obj2 in zip(objects1[1:], objects2[1:]):  # Skip anchor
        pos1 = result1.positions[obj1]
        pos2 = result2.positions[obj2]
        if pos1 != pos2:
            any_different = True
            break

    assert any_different, "Different seeds should produce different results"


def test_object_placer_multi_env_returns_multi_env_result():
    """place() with num_envs>1 returns one PlacementResult per env."""
    num_envs = 4
    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3)
    desk, box1, box2 = _create_test_objects()
    objects = [desk, box1, box2]
    placer = ObjectPlacer(params=ObjectPlacerParams(placement_seed=42, solver_params=solver_params))
    result = placer.place(objects, num_envs=num_envs)

    assert len(result) == num_envs
    for r in result:
        assert isinstance(r, PlacementResult)
        assert box1 in r.positions
        assert box2 in r.positions
        assert len(r.positions[box1]) == 3
        assert len(r.positions[box2]) == 3


def test_object_placer_multi_env_produces_different_positions():
    """Multi-env placement produces different positions across envs."""
    num_envs = 4
    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3)
    desk, box1, box2 = _create_test_objects()
    objects = [desk, box1, box2]
    placer = ObjectPlacer(params=ObjectPlacerParams(placement_seed=42, solver_params=solver_params))
    result = placer.place(objects, num_envs=num_envs)

    assert len(result) == num_envs
    positions_box1 = [result[e].positions[box1] for e in range(num_envs)]
    any_different = any(positions_box1[i] != positions_box1[j] for i in range(num_envs) for j in range(i + 1, num_envs))
    assert any_different, "Multi-env placement should produce different positions across environments"


def test_relation_solver_multi_env_batched_positions():
    """Test that solver with list[dict] input returns list[dict] output."""
    solver_params = RelationSolverParams(max_iters=50)
    desk, box1, box2 = _create_test_objects()
    objects = [desk, box1, box2]

    initial_positions = [
        {desk: (0.0, 0.0, 0.0), box1: (0.2, 0.2, 0.11), box2: (0.5, 0.5, 0.11)},
        {desk: (0.0, 0.0, 0.0), box1: (0.3, 0.3, 0.11), box2: (0.6, 0.6, 0.11)},
    ]

    solver = RelationSolver(params=solver_params)
    result = solver.solve(objects=objects, initial_positions=initial_positions)

    assert isinstance(result, list)
    assert len(result) == 2
    for d in result:
        assert isinstance(d, dict)
        for obj in objects:
            assert obj in d
            assert len(d[obj]) == 3


def test_object_placer_applies_pose_per_env():
    """place(num_envs>1) sets PosePerEnv on each object."""
    num_envs = 4
    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3)
    desk, box1, box2 = _create_test_objects()
    objects = [desk, box1, box2]
    placer = ObjectPlacer(
        params=ObjectPlacerParams(placement_seed=42, solver_params=solver_params, apply_positions_to_objects=True)
    )
    placer.place(objects, num_envs=num_envs)

    for obj in [box1, box2]:
        pose = obj.get_initial_pose()
        assert isinstance(pose, PosePerEnv), f"{obj.name} should have PosePerEnv, got {type(pose).__name__}"
        assert len(pose.poses) == num_envs


def _is_identity_quat(rotation_xyzw: tuple[float, float, float, float], atol: float = 1e-6) -> bool:
    """Whether a quaternion is (approximately) the identity (0, 0, 0, 1)."""
    x, y, z, w = rotation_xyzw
    return abs(x) < atol and abs(y) < atol and abs(z) < atol and abs(abs(w) - 1.0) < atol


def _yaw_rad_from_quat(rotation_xyzw: tuple[float, float, float, float]) -> float:
    """Z-yaw (radians) of a pure-Z quaternion (x, y, z, w)."""
    return 2.0 * math.atan2(rotation_xyzw[2], rotation_xyzw[3])


def test_random_yaw_init_rotates_non_anchors_only_when_enabled():
    """Enabled rotates non-anchors (never the anchor); disabled (same seed) keeps identity."""
    solver_params = RelationSolverParams(max_iters=10, verbose=False)

    desk, box1, box2 = _create_test_objects()
    ObjectPlacer(params=ObjectPlacerParams(placement_seed=42, solver_params=solver_params, random_yaw_init=True)).place(
        objects=[desk, box1, box2], num_envs=1
    )
    assert _is_identity_quat(desk.get_initial_pose().rotation_xyzw), "anchor is never rotated"
    for box in [box1, box2]:
        assert abs(_yaw_rad_from_quat(box.get_initial_pose().rotation_xyzw)) > 1e-3, f"{box.name} should be rotated"

    desk, box1, box2 = _create_test_objects()
    ObjectPlacer(params=ObjectPlacerParams(placement_seed=42, solver_params=solver_params)).place(
        objects=[desk, box1, box2], num_envs=1
    )
    for box in [box1, box2]:
        assert _is_identity_quat(box.get_initial_pose().rotation_xyzw), f"{box.name} should be unrotated"


def test_compose_rotation_combines_sampled_yaw_and_marker():
    """rotate_quat_by_yaw composes the sampled yaw on top of a RotateAroundSolution marker."""
    marker = RotateAroundSolution(yaw_rad=math.pi / 6)
    composed = rotate_quat_by_yaw(marker.get_rotation_xyzw(), math.pi / 3)
    # Both rotations are about Z, so the total yaw is pi/6 + pi/3 = pi/2.
    assert abs(composed[0]) < 1e-6 and abs(composed[1]) < 1e-6
    assert abs(_yaw_rad_from_quat(composed) - math.pi / 2) < 1e-5


def test_random_yaw_init_multi_env_distinct_yaws():
    """Multi-env placement with random_yaw_init yields distinct per-env yaws via PosePerEnv."""
    num_envs = 4
    solver_params = RelationSolverParams(max_iters=10, verbose=False)
    desk, box1, box2 = _create_test_objects()
    objects = [desk, box1, box2]
    placer = ObjectPlacer(
        params=ObjectPlacerParams(placement_seed=42, solver_params=solver_params, random_yaw_init=True)
    )
    placer.place(objects, num_envs=num_envs)

    pose = box1.get_initial_pose()
    assert isinstance(pose, PosePerEnv)
    yaws = [_yaw_rad_from_quat(p.rotation_xyzw) for p in pose.poses]
    assert any(
        abs(yaws[i] - yaws[j]) > 1e-6 for i in range(num_envs) for j in range(i + 1, num_envs)
    ), "Per-env yaws should differ across environments"


def _placed_yaws(seed: int) -> tuple[float, float]:
    """Place two boxes with random_yaw_init at the given seed; return their applied yaws."""
    solver_params = RelationSolverParams(max_iters=10, verbose=False)
    desk, box1, box2 = _create_test_objects()
    ObjectPlacer(
        params=ObjectPlacerParams(placement_seed=seed, solver_params=solver_params, random_yaw_init=True)
    ).place([desk, box1, box2], num_envs=1)
    return (
        _yaw_rad_from_quat(box1.get_initial_pose().rotation_xyzw),
        _yaw_rad_from_quat(box2.get_initial_pose().rotation_xyzw),
    )


def test_random_yaw_init_seed_determinism():
    """Same seed -> identical sampled yaws; a different seed -> different yaws."""
    assert _placed_yaws(42) == _placed_yaws(42), "same seed must reproduce identical yaws"
    assert _placed_yaws(42) != _placed_yaws(7), "different seeds should produce different yaws"


def test_random_yaw_init_applied_yaw_matches_selected_candidate():
    """Applied yaw equals the selected candidate's recorded orientation (apply stays in sync with validation)."""
    solver_params = RelationSolverParams(max_iters=10, verbose=False)
    desk, box1, box2 = _create_test_objects()
    placer = ObjectPlacer(
        params=ObjectPlacerParams(placement_seed=11, solver_params=solver_params, random_yaw_init=True)
    )
    (result,) = placer.place([desk, box1, box2], num_envs=1)
    for box in (box1, box2):
        applied = _yaw_rad_from_quat(box.get_initial_pose().rotation_xyzw)
        assert abs(wrap_angle_to_pi(applied - result.orientations[box])) < 1e-5


def test_random_yaw_init_composes_marker_yaw():
    marker_yaw = math.pi / 6
    solver_params = RelationSolverParams(max_iters=10, verbose=False)
    desk, box1, box2 = _create_test_objects()
    box1.add_relation(RotateAroundSolution(yaw_rad=marker_yaw))
    placer = ObjectPlacer(
        params=ObjectPlacerParams(placement_seed=3, solver_params=solver_params, random_yaw_init=True)
    )
    (result,) = placer.place([desk, box1, box2], num_envs=1)
    applied = _yaw_rad_from_quat(box1.get_initial_pose().rotation_xyzw)
    # result.orientations now carries total yaw = marker + sampled
    assert abs(wrap_angle_to_pi(applied - result.orientations[box1])) < 1e-5


def test_random_yaw_init_rejects_roll_pitch_marker():
    """A roll/pitch marker cannot be enclosed by a Z-rotated bbox, so placement must fail loudly."""
    solver_params = RelationSolverParams(max_iters=5, verbose=False)
    desk, box1, box2 = _create_test_objects()
    box1.add_relation(RotateAroundSolution(roll_rad=0.3))
    placer = ObjectPlacer(
        params=ObjectPlacerParams(placement_seed=1, solver_params=solver_params, random_yaw_init=True)
    )
    with pytest.raises(AssertionError):
        placer.place([desk, box1, box2], num_envs=1)


def test_marker_yaw_applied_without_random_yaw_init():
    """RotateAroundSolution marker must be applied even when random_yaw_init=False."""
    marker_yaw = math.pi / 4
    solver_params = RelationSolverParams(max_iters=5, verbose=False)
    desk, box1, box2 = _create_test_objects()
    box1.add_relation(RotateAroundSolution(yaw_rad=marker_yaw))
    placer = ObjectPlacer(
        params=ObjectPlacerParams(placement_seed=1, solver_params=solver_params, random_yaw_init=False)
    )
    placer.place([desk, box1, box2], num_envs=1)
    applied = _yaw_rad_from_quat(box1.get_initial_pose().rotation_xyzw)
    assert abs(wrap_angle_to_pi(applied - marker_yaw)) < 1e-5, f"Marker yaw {marker_yaw} must be applied; got {applied}"


def _positions_by_name(result: PlacementResult) -> dict[str, tuple[float, float, float]]:
    return {obj.name: pos for obj, pos in result.positions.items()}


# ---------------------------------------------------------------------------
# PooledObjectPlacer reproducibility — homogeneous objects.
# Heterogeneous-object (per-env variant) counterparts live in test_heterogeneous_placement.py.
# ---------------------------------------------------------------------------


def test_pooled_placer_homogeneous_same_seed_produces_identical_samples():
    """PooledObjectPlacer.sample_with_replacement must be reproducible under placement_seed."""
    solver_params = RelationSolverParams(max_iters=50)
    placer_params = ObjectPlacerParams(placement_seed=42, solver_params=solver_params, apply_positions_to_objects=False)

    pool1 = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=8)
    pool2 = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=8)

    samples1 = pool1.sample_with_replacement(20)
    samples2 = pool2.sample_with_replacement(20)

    for s1, s2 in zip(samples1, samples2):
        assert _positions_by_name(s1) == _positions_by_name(s2)


def test_pooled_placer_homogeneous_different_seeds_produce_different_samples():
    """Different placement_seed values should produce different sample sequences."""
    solver_params = RelationSolverParams(max_iters=50)

    pool1 = PooledObjectPlacer(
        objects=list(_create_test_objects()),
        placer_params=ObjectPlacerParams(
            placement_seed=42, solver_params=solver_params, apply_positions_to_objects=False
        ),
        pool_size=8,
    )
    pool2 = PooledObjectPlacer(
        objects=list(_create_test_objects()),
        placer_params=ObjectPlacerParams(
            placement_seed=123, solver_params=solver_params, apply_positions_to_objects=False
        ),
        pool_size=8,
    )

    samples1 = pool1.sample_with_replacement(20)
    samples2 = pool2.sample_with_replacement(20)

    any_different = any(_positions_by_name(s1) != _positions_by_name(s2) for s1, s2 in zip(samples1, samples2))
    assert any_different, "Different seeds should produce different samples"


def test_pooled_placer_homogeneous_sample_with_replacement_reproducible_per_env_id():
    """sample_with_replacement should reproduce each env's draws under a fixed (placement_seed, num_envs)."""
    num_envs = 3
    solver_params = RelationSolverParams(max_iters=50)
    placer_params = ObjectPlacerParams(placement_seed=42, solver_params=solver_params, apply_positions_to_objects=False)

    pool1 = PooledObjectPlacer(
        objects=list(_create_test_objects()), placer_params=placer_params, pool_size=12, num_envs=num_envs
    )
    pool2 = PooledObjectPlacer(
        objects=list(_create_test_objects()), placer_params=placer_params, pool_size=12, num_envs=num_envs
    )

    draws1 = pool1.sample_with_replacement(num_envs * 4)
    draws2 = pool2.sample_with_replacement(num_envs * 4)

    for i, (d1, d2) in enumerate(zip(draws1, draws2)):
        assert _positions_by_name(d1) == _positions_by_name(d2), f"slot {i} (env {i % num_envs}) not reproducible"


def test_pooled_placer_homogeneous_builds_identical_layouts_for_same_seed_and_objects():
    """Same objects and placement_seed should produce bit-identical layouts (no sampling RNG involved)."""
    solver_params = RelationSolverParams(max_iters=50)
    placer_params = ObjectPlacerParams(placement_seed=42, solver_params=solver_params, apply_positions_to_objects=False)

    pool1 = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=4)
    pool2 = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=4)

    layouts1 = pool1.sample_without_replacement(4)
    layouts2 = pool2.sample_without_replacement(4)

    for layout1, layout2 in zip(layouts1, layouts2):
        assert _positions_by_name(layout1) == _positions_by_name(layout2)


def test_pooled_placer_homogeneous_continues_seed_stream_across_refill():
    """Pool refill must advance the candidate seed stream so it doesn't replay the initial batch."""
    solver_params = RelationSolverParams(max_iters=50)
    placer_params = ObjectPlacerParams(placement_seed=42, solver_params=solver_params, apply_positions_to_objects=False)

    pool1 = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=2)
    pool2 = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=2)

    # 4 single draws: initial batch (0-1), then a forced refill (2-3).
    draws1 = [pool1.sample_without_replacement(1)[0] for _ in range(4)]
    draws2 = [pool2.sample_without_replacement(1)[0] for _ in range(4)]

    for d1, d2 in zip(draws1, draws2):
        assert _positions_by_name(d1) == _positions_by_name(d2), "Same seed must give same draws across refill"

    # Refill must use a fresh seed range, so its layouts can't duplicate the initial batch.
    initial_batch = {repr(_positions_by_name(draws1[0])), repr(_positions_by_name(draws1[1]))}
    refill_batch = {repr(_positions_by_name(draws1[2])), repr(_positions_by_name(draws1[3]))}
    assert initial_batch.isdisjoint(refill_batch), "Refill replayed the initial seed range; seed stream not advancing"


def test_pooled_placer_homogeneous_unseeded_does_not_crash():
    """placement_seed=None must work without requiring deterministic seed bookkeeping."""
    solver_params = RelationSolverParams(max_iters=50)
    placer_params = ObjectPlacerParams(
        placement_seed=None, solver_params=solver_params, apply_positions_to_objects=False
    )

    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=4)
    samples = pool.sample_with_replacement(20)

    assert len(samples) == 20
    for sample in samples:
        assert _positions_by_name(sample)


def test_pooled_placer_homogeneous_stored_layouts_have_distinct_positions_dicts():
    """Each stored layout must own a distinct positions dict (no aliasing across pool entries)."""
    solver_params = RelationSolverParams(max_iters=50)
    placer_params = ObjectPlacerParams(placement_seed=42, solver_params=solver_params, apply_positions_to_objects=False)

    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=4)
    draws = pool.sample_without_replacement(4)

    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            assert (
                draws[i].positions is not draws[j].positions
            ), f"Layouts {i} and {j} share the same positions dict reference"


def test_pooled_placer_homogeneous_sample_without_replacement_count_exceeds_pool_size():
    """sample_without_replacement(count) where count > pool_size must solve a larger batch in one shot."""
    solver_params = RelationSolverParams(max_iters=50)
    placer_params = ObjectPlacerParams(placement_seed=42, solver_params=solver_params, apply_positions_to_objects=False)

    pool1 = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=2)
    pool2 = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=2)

    draws1 = pool1.sample_without_replacement(5)
    draws2 = pool2.sample_without_replacement(5)

    assert len(draws1) == 5 and len(draws2) == 5
    for d1, d2 in zip(draws1, draws2):
        assert _positions_by_name(d1) == _positions_by_name(d2)


def test_pooled_placer_homogeneous_sample_with_replacement_reproducible_across_refill():
    """sample_with_replacement must remain reproducible after a refill mutates the pool."""
    solver_params = RelationSolverParams(max_iters=50)
    placer_params = ObjectPlacerParams(placement_seed=42, solver_params=solver_params, apply_positions_to_objects=False)

    def draw_sequence(pool: PooledObjectPlacer) -> list:
        before = pool.sample_with_replacement(5)
        # Consume the initial pool to force a refill.
        pool.sample_without_replacement(2)
        pool.sample_without_replacement(1)
        after = pool.sample_with_replacement(5)
        return before + after

    pool1 = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=2)
    pool2 = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=2)

    seq1 = draw_sequence(pool1)
    seq2 = draw_sequence(pool2)

    for s1, s2 in zip(seq1, seq2):
        assert _positions_by_name(s1) == _positions_by_name(s2)


def test_pooled_placer_homogeneous_rejects_pool_size_below_one():
    """pool_size < 1 is invalid public constructor input."""
    placer_params = ObjectPlacerParams(placement_seed=42, solver_params=RelationSolverParams(max_iters=10))
    with pytest.raises(AssertionError, match="pool_size must be >= 1"):
        PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=0)
