# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ObjectPlacer and RelationSolver reproducibility."""

import json
import math

import pytest

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_result import MultiEnvPlacementResult, PlacementResult, ValidationReport
from isaaclab_arena.relations.pooled_object_placer import PooledLayout, PooledObjectPlacer
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, NextTo, On, RotateAroundSolution, Side
from isaaclab_arena.tests.utils.placement import layout_signature
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
    result1 = placer1.place(objects=objects1)

    # Run 2
    desk2, box1_run2, box2_run2 = _create_test_objects()
    objects2 = [desk2, box1_run2, box2_run2]
    placer2 = ObjectPlacer(params=ObjectPlacerParams(placement_seed=seed, solver_params=solver_params))
    result2 = placer2.place(objects=objects2)

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
    result1 = placer1.place(objects=objects1)

    # Run 2 with seed 123
    desk2, box1_run2, box2_run2 = _create_test_objects()
    objects2 = [desk2, box1_run2, box2_run2]
    placer2 = ObjectPlacer(params=ObjectPlacerParams(placement_seed=123, solver_params=solver_params))
    result2 = placer2.place(objects=objects2)

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
    """Test that ObjectPlacer.place with num_envs>1 returns MultiEnvPlacementResult."""
    num_envs = 4
    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3)
    desk, box1, box2 = _create_test_objects()
    objects = [desk, box1, box2]
    placer = ObjectPlacer(params=ObjectPlacerParams(placement_seed=42, solver_params=solver_params))
    result = placer.place(objects, num_envs=num_envs)

    assert isinstance(result, MultiEnvPlacementResult)
    assert len(result.results) == num_envs
    for r in result.results:
        assert isinstance(r, PlacementResult)
        assert box1 in r.positions
        assert box2 in r.positions
        assert len(r.positions[box1]) == 3
        assert len(r.positions[box2]) == 3


def test_object_placer_multi_env_produces_different_positions():
    """Test that multi-env placement produces different positions across environments."""
    num_envs = 4
    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3)
    desk, box1, box2 = _create_test_objects()
    objects = [desk, box1, box2]
    placer = ObjectPlacer(params=ObjectPlacerParams(placement_seed=42, solver_params=solver_params))
    result = placer.place(objects, num_envs=num_envs)

    assert isinstance(result, MultiEnvPlacementResult)
    # At least one pair of envs should have different positions for a non-anchor object.
    positions_box1 = [result.results[e].positions[box1] for e in range(num_envs)]
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
    result = placer.place([desk, box1, box2], num_envs=1)
    for box in (box1, box2):
        applied = _yaw_rad_from_quat(box.get_initial_pose().rotation_xyzw)
        assert abs(wrap_angle_to_pi(applied - result.orientations[box])) < 1e-5


def test_random_yaw_init_composes_marker_yaw():
    """A yaw RotateAroundSolution marker composes with the sampled yaw: applied == marker + sampled."""
    marker_yaw = math.pi / 6
    solver_params = RelationSolverParams(max_iters=10, verbose=False)
    desk, box1, box2 = _create_test_objects()
    box1.add_relation(RotateAroundSolution(yaw_rad=marker_yaw))
    placer = ObjectPlacer(
        params=ObjectPlacerParams(placement_seed=3, solver_params=solver_params, random_yaw_init=True)
    )
    result = placer.place([desk, box1, box2], num_envs=1)
    applied = _yaw_rad_from_quat(box1.get_initial_pose().rotation_xyzw)
    assert abs(wrap_angle_to_pi(applied - (marker_yaw + result.orientations[box1]))) < 1e-5


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


def _positions_by_name(result: PlacementResult) -> dict[str, tuple[float, float, float]]:
    return {obj.name: pos for obj, pos in result.positions.items()}


def _pool_signatures(pool: PooledObjectPlacer) -> list[list[tuple]]:
    """All stored layouts per env, as comparable signatures."""
    return [[layout_signature(layout.result) for layout in env_pool.layouts] for env_pool in pool._env_pools]


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


def test_pooled_placer_stored_layouts_groups_live_results_by_env():
    """stored_layouts must return a per-env snapshot of the live stored PlacementResults."""
    num_envs = 3
    solver_params = RelationSolverParams(max_iters=50)
    placer_params = ObjectPlacerParams(placement_seed=42, solver_params=solver_params, apply_positions_to_objects=False)

    pool = PooledObjectPlacer(
        objects=list(_create_test_objects()), placer_params=placer_params, pool_size=12, num_envs=num_envs
    )

    grouped = pool.stored_layouts
    assert len(grouped) == num_envs
    assert all(isinstance(env_layouts, tuple) for env_layouts in grouped)
    # Each returned result is the same live object the pool stores (no copy), grouped per env.
    for cur_env, env_layouts in enumerate(grouped):
        assert list(env_layouts) == [pooled.result for pooled in pool._env_pools[cur_env].layouts]


def test_pooled_placer_stored_layouts_post_validation_flows_into_accepts():
    """Enriching a stored layout's report in place (the post-validation use case) must change accepts()."""
    solver_params = RelationSolverParams(max_iters=50)
    placer_params = ObjectPlacerParams(placement_seed=42, solver_params=solver_params, apply_positions_to_objects=False)

    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=4)
    result = pool.stored_layouts[0][0]
    assert pool.accepts(result)

    # Record a failing post-pool check (e.g. a simulation collision test) on the live layout.
    result.validation = result.validation.with_check("sim_collision_free", False)
    assert "sim_collision_free" in result.validation.failed_checks
    assert not pool.accepts(result)


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


def test_pooled_placer_layout_filter_receives_validation_reports():
    """The injected layout_filter should be consulted with each layout's ValidationReport."""
    solver_params = RelationSolverParams(max_iters=50)
    placer_params = ObjectPlacerParams(placement_seed=42, solver_params=solver_params, apply_positions_to_objects=False)

    seen: list[ValidationReport] = []

    def record(report: ValidationReport) -> bool:
        seen.append(report)
        return report.passed

    PooledObjectPlacer(
        objects=list(_create_test_objects()), placer_params=placer_params, pool_size=4, layout_filter=record
    )
    assert seen, "layout_filter should be consulted while filling the pool"
    assert all(isinstance(report, ValidationReport) for report in seen)


def test_pooled_placer_rejecting_layout_filter_forces_fallback():
    """A layout_filter that rejects everything should drive the pool onto best-loss fallbacks."""
    solver_params = RelationSolverParams(max_iters=50)
    placer_params = ObjectPlacerParams(placement_seed=42, solver_params=solver_params, apply_positions_to_objects=False)

    default_pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=4)
    assert default_pool.had_fallbacks is False

    strict_pool = PooledObjectPlacer(
        objects=list(_create_test_objects()),
        placer_params=placer_params,
        pool_size=4,
        layout_filter=lambda report: False,
    )
    assert strict_pool.had_fallbacks is True


def test_summarize_rejections_attributes_filter_only_rejections():
    """A layout the filter rejects despite passing every built-in check counts under 'layout_filter'."""
    solver_params = RelationSolverParams(max_iters=50)
    placer_params = ObjectPlacerParams(placement_seed=42, solver_params=solver_params, apply_positions_to_objects=False)
    pool = PooledObjectPlacer(
        objects=list(_create_test_objects()), placer_params=placer_params, pool_size=4, layout_filter=lambda r: False
    )

    filter_only = PlacementResult(
        positions={}, final_loss=0.0, attempts=1, validation=ValidationReport(checks={"no_overlap": True})
    )
    check_failed = PlacementResult(
        positions={}, final_loss=0.0, attempts=1, validation=ValidationReport(checks={"no_overlap": False})
    )
    summary = pool._summarize_rejections([filter_only, check_failed])
    assert summary == {"layout_filter": 1, "no_overlap": 1}


def test_solve_and_store_resets_stale_rejection_summary():
    """A fresh solve must not surface rejection counts left over from an earlier solve."""
    solver_params = RelationSolverParams(max_iters=50)
    placer_params = ObjectPlacerParams(placement_seed=42, solver_params=solver_params, apply_positions_to_objects=False)
    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=4)

    pool._last_rejection_summary = {"stale_check": 99}
    pool._solve_and_store(4)

    assert pool._last_rejection_summary == {}


def test_pooled_placer_layout_filter_can_accept_a_subset_of_checks():
    """A layout_filter keyed on a single named check should accept/reject via accepts accordingly."""
    solver_params = RelationSolverParams(max_iters=50)
    placer_params = ObjectPlacerParams(placement_seed=42, solver_params=solver_params, apply_positions_to_objects=False)

    # Accept any layout whose no_overlap check passes, even if on_relations failed.
    pool = PooledObjectPlacer(
        objects=list(_create_test_objects()),
        placer_params=placer_params,
        pool_size=4,
        layout_filter=lambda report: report.checks.get("no_overlap", False),
    )

    overlap_ok = PlacementResult(
        positions={},
        final_loss=0.0,
        attempts=1,
        validation=ValidationReport(checks={"no_overlap": True, "on_relations": False}),
    )
    overlap_bad = PlacementResult(
        positions={},
        final_loss=0.0,
        attempts=1,
        validation=ValidationReport(checks={"no_overlap": False, "on_relations": True}),
    )
    assert pool.accepts(overlap_ok) is True
    assert pool.accepts(overlap_bad) is False


def test_pooled_layout_mark_used_increments_use_count():
    """PooledLayout.mark_used should increment use_count without replacing the wrapped result."""
    result = PlacementResult(
        positions={}, final_loss=0.0, attempts=1, validation=ValidationReport(checks={"no_overlap": True})
    )
    layout = PooledLayout(result)
    assert layout.use_count == 0
    layout.mark_used()
    layout.mark_used()
    assert layout.use_count == 2
    assert layout.result is result


def test_pooled_placer_sample_with_replacement_tracks_use_count():
    """Each sample_with_replacement draw should bump exactly one stored layout's use_count."""
    solver_params = RelationSolverParams(max_iters=50)
    placer_params = ObjectPlacerParams(placement_seed=42, solver_params=solver_params, apply_positions_to_objects=False)

    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=8)
    pool.sample_with_replacement(20)

    total_uses = sum(layout.use_count for env_pool in pool._env_pools for layout in env_pool.layouts)
    assert total_uses == 20


def test_pooled_placer_reusable_draws_span_multiple_env_pools():
    """Reusable sampling flattens every env pool, so a multi-env pool serves layouts from >1 origin."""
    solver_params = RelationSolverParams(max_iters=50)
    placer_params = ObjectPlacerParams(placement_seed=42, solver_params=solver_params, apply_positions_to_objects=False)

    pool = PooledObjectPlacer(
        objects=list(_create_test_objects()), placer_params=placer_params, pool_size=9, num_envs=3
    )
    origin_by_id = {
        id(layout.result): env_idx for env_idx, env_pool in enumerate(pool._env_pools) for layout in env_pool.layouts
    }
    assert len(set(origin_by_id.values())) > 1, "fixture must spread layouts across multiple env pools"

    draws = pool.sample_with_replacement(30)
    origins_hit = {origin_by_id[id(result)] for result in draws}
    assert len(origins_hit) > 1, "reusable draws should span more than one origin env pool"


def test_pooled_placer_sample_without_replacement_marks_consumed_layouts_used():
    """sample_without_replacement should mark exactly the consumed layouts used once each."""
    solver_params = RelationSolverParams(max_iters=50)
    placer_params = ObjectPlacerParams(placement_seed=42, solver_params=solver_params, apply_positions_to_objects=False)

    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=8)
    pool.sample_without_replacement(4)

    use_counts = [layout.use_count for env_pool in pool._env_pools for layout in env_pool.layouts]
    assert sorted(use_counts, reverse=True)[:4] == [1, 1, 1, 1]
    assert sum(use_counts) == 4


def test_pooled_placer_use_count_progression_replays_under_fixed_seed():
    """Per-slot use_count after sampling should replay identically under a fixed seed."""
    solver_params = RelationSolverParams(max_iters=50)
    placer_params = ObjectPlacerParams(placement_seed=42, solver_params=solver_params, apply_positions_to_objects=False)

    pool1 = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=8)
    pool2 = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=8)

    pool1.sample_with_replacement(20)
    pool2.sample_with_replacement(20)

    counts1 = [layout.use_count for env_pool in pool1._env_pools for layout in env_pool.layouts]
    counts2 = [layout.use_count for env_pool in pool2._env_pools for layout in env_pool.layouts]
    assert counts1 == counts2
    assert sum(counts1) == 20
    # Guard against a degenerate _draw (e.g. always slot 0): draws must spread across the pool.
    assert sum(1 for count in counts1 if count > 0) > 1


def _make_seeded_params(seed: int = 42) -> ObjectPlacerParams:
    return ObjectPlacerParams(
        placement_seed=seed,
        solver_params=RelationSolverParams(max_iters=50),
        apply_positions_to_objects=False,
    )


def test_pooled_placer_save_load_round_trip_preserves_layouts(tmp_path):
    """save() then load() must reproduce every stored layout, with use_count reset to 0."""
    placer_params = _make_seeded_params()
    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=6)

    path = tmp_path / "layouts.json"
    pool.save(path)
    loaded = PooledObjectPlacer.load(path, list(_create_test_objects()), placer_params)

    assert _pool_signatures(loaded) == _pool_signatures(pool)
    assert all(layout.use_count == 0 for env_pool in loaded._env_pools for layout in env_pool.layouts)


def test_pooled_placer_save_load_round_trip_multi_env_homogeneous(tmp_path):
    """A multi-env homogeneous pool's per-env layouts survive the round trip (variants covered separately)."""
    num_envs = 3
    placer_params = _make_seeded_params()
    pool = PooledObjectPlacer(
        objects=list(_create_test_objects()), placer_params=placer_params, pool_size=9, num_envs=num_envs
    )

    path = tmp_path / "layouts.json"
    pool.save(path)
    loaded = PooledObjectPlacer.load(path, list(_create_test_objects()), placer_params, num_envs=num_envs)

    assert loaded.num_envs == num_envs
    assert _pool_signatures(loaded) == _pool_signatures(pool)


def test_pooled_placer_load_replays_saved_poses_without_resolving(tmp_path):
    """load() reuses stored poses: a different solve seed must not change the loaded layouts."""
    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=_make_seeded_params(42), pool_size=6)
    path = tmp_path / "layouts.json"
    pool.save(path)

    # A fresh solve under seed 999 would differ; loading must replay the seed-42 layouts instead.
    loaded = PooledObjectPlacer.load(path, list(_create_test_objects()), _make_seeded_params(999))
    assert _pool_signatures(loaded) == _pool_signatures(pool)


def test_pooled_placer_loaded_pool_samples_match_origin(tmp_path):
    """A loaded pool draws the same layouts as the saved one under a shared seed."""
    placer_params = _make_seeded_params()
    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=6)
    path = tmp_path / "layouts.json"
    pool.save(path)
    loaded = PooledObjectPlacer.load(path, list(_create_test_objects()), placer_params)

    for original, restored in zip(pool.sample_with_replacement(20), loaded.sample_with_replacement(20)):
        assert _positions_by_name(original) == _positions_by_name(restored)


def test_pooled_placer_load_rejects_unknown_object(tmp_path):
    """A saved layout naming an object absent from the provided objects must fail loudly."""
    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=_make_seeded_params(), pool_size=4)
    path = tmp_path / "layouts.json"
    pool.save(path)

    desk, box1, _box2 = _create_test_objects()
    with pytest.raises(AssertionError, match="do not match the provided objects"):
        PooledObjectPlacer.load(path, [desk, box1], _make_seeded_params())


def test_pooled_placer_load_rejects_num_envs_mismatch(tmp_path):
    """Requesting a num_envs different from the saved file must fail loudly."""
    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=_make_seeded_params(), pool_size=4)
    path = tmp_path / "layouts.json"
    pool.save(path)

    with pytest.raises(AssertionError, match="num_envs=3 does not match"):
        PooledObjectPlacer.load(path, list(_create_test_objects()), _make_seeded_params(), num_envs=3)


def test_pooled_placer_save_load_round_trip_preserves_orientations(tmp_path):
    """With random_yaw_init, per-object yaws must survive the round trip (not silently empty)."""
    placer_params = ObjectPlacerParams(
        placement_seed=42,
        solver_params=RelationSolverParams(max_iters=50),
        apply_positions_to_objects=False,
        random_yaw_init=True,
    )
    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=placer_params, pool_size=6)
    # Guard the fixture: the round trip must actually exercise non-empty orientations.
    assert any(layout.result.orientations for env_pool in pool._env_pools for layout in env_pool.layouts)

    path = tmp_path / "layouts.json"
    pool.save(path)
    loaded = PooledObjectPlacer.load(path, list(_create_test_objects()), placer_params)

    assert _pool_signatures(loaded) == _pool_signatures(pool)


def test_pooled_placer_load_missing_file_raises(tmp_path):
    """A missing cache file names the path rather than raising a bare FileNotFoundError."""
    with pytest.raises(AssertionError, match="not found"):
        PooledObjectPlacer.load(tmp_path / "nope.json", list(_create_test_objects()), _make_seeded_params())


def test_pooled_placer_load_rejects_malformed_json(tmp_path):
    """A truncated/hand-edited file raises an attributable ValueError, not a bare JSONDecodeError."""
    path = tmp_path / "bad.json"
    path.write_text("{ not valid json")
    with pytest.raises(ValueError, match="not valid JSON"):
        PooledObjectPlacer.load(path, list(_create_test_objects()), _make_seeded_params())


def test_pooled_placer_load_rejects_missing_key(tmp_path):
    """A file missing a required top-level key names that key."""
    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=_make_seeded_params(), pool_size=4)
    path = tmp_path / "layouts.json"
    pool.save(path)

    data = json.loads(path.read_text())
    del data["env_pools"]
    path.write_text(json.dumps(data))

    with pytest.raises(AssertionError, match="missing required key 'env_pools'"):
        PooledObjectPlacer.load(path, list(_create_test_objects()), _make_seeded_params())


def test_pooled_placer_load_rejects_non_bool_validation(tmp_path):
    """A corrupt non-bool validation value fails loudly instead of coercing to passing."""
    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=_make_seeded_params(), pool_size=4)
    path = tmp_path / "layouts.json"
    pool.save(path)

    data = json.loads(path.read_text())
    checks = data["env_pools"][0][0]["validation"]
    checks[next(iter(checks))] = "false"
    path.write_text(json.dumps(data))

    with pytest.raises(AssertionError, match="must be a JSON bool"):
        PooledObjectPlacer.load(path, list(_create_test_objects()), _make_seeded_params())


def test_pooled_placer_load_rejects_empty_validation(tmp_path):
    """An empty validation map would load as a failing layout, so it must be rejected on load."""
    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=_make_seeded_params(), pool_size=4)
    path = tmp_path / "layouts.json"
    pool.save(path)

    data = json.loads(path.read_text())
    data["env_pools"][0][0]["validation"] = {}
    path.write_text(json.dumps(data))

    with pytest.raises(AssertionError, match="empty validation map"):
        PooledObjectPlacer.load(path, list(_create_test_objects()), _make_seeded_params())


def test_pooled_placer_load_rejects_non_bool_had_fallbacks(tmp_path):
    """A non-bool had_fallbacks must fail loudly rather than re-suppressing the fallback warning."""
    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=_make_seeded_params(), pool_size=4)
    path = tmp_path / "layouts.json"
    pool.save(path)

    data = json.loads(path.read_text())
    data["had_fallbacks"] = "yes"
    path.write_text(json.dumps(data))

    with pytest.raises(AssertionError, match="had_fallbacks' must be a bool"):
        PooledObjectPlacer.load(path, list(_create_test_objects()), _make_seeded_params())


def test_pooled_placer_load_rejects_malformed_position(tmp_path):
    """A wrong-length position would silently mis-place an object, so load must reject it."""
    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=_make_seeded_params(), pool_size=4)
    path = tmp_path / "layouts.json"
    pool.save(path)

    data = json.loads(path.read_text())
    positions = data["env_pools"][0][0]["positions"]
    positions[next(iter(positions))] = [1.0, 2.0]
    path.write_text(json.dumps(data))

    with pytest.raises(AssertionError, match="length-3 sequence"):
        PooledObjectPlacer.load(path, list(_create_test_objects()), _make_seeded_params())


def test_pooled_placer_load_rejects_duplicate_object_names(tmp_path):
    """Duplicate object names would collapse to one slot, so load must reject them."""
    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=_make_seeded_params(), pool_size=4)
    path = tmp_path / "layouts.json"
    pool.save(path)

    objects = list(_create_test_objects())
    objects.append(objects[0])
    with pytest.raises(AssertionError, match="Object names must be unique"):
        PooledObjectPlacer.load(path, objects, _make_seeded_params())


def test_pooled_placer_save_rejects_non_finite_pose(tmp_path):
    """A non-finite coordinate must fail at save, leaving neither a target nor an orphan temp file."""
    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=_make_seeded_params(), pool_size=4)
    result = pool._env_pools[0].layouts[0].result
    obj = next(iter(result.positions))
    result.positions[obj] = (float("nan"), 0.0, 0.0)

    path = tmp_path / "layouts.json"
    with pytest.raises(ValueError, match="JSON compliant"):
        pool.save(path)
    assert not path.exists()
    assert not path.with_name(f"{path.name}.tmp").exists()


def test_pooled_placer_save_rejects_duplicate_object_names(tmp_path):
    """Saving with duplicate names would collapse a pose, so it must fail loudly like load does."""
    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=_make_seeded_params(), pool_size=4)
    pool._objects.append(pool._objects[0])

    with pytest.raises(AssertionError, match="must be unique to save"):
        pool.save(tmp_path / "layouts.json")


def test_pooled_placer_load_rejects_non_numeric_position(tmp_path):
    """A non-numeric coordinate must fail loudly rather than crash deep in float()."""
    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=_make_seeded_params(), pool_size=4)
    path = tmp_path / "layouts.json"
    pool.save(path)

    data = json.loads(path.read_text())
    positions = data["env_pools"][0][0]["positions"]
    positions[next(iter(positions))] = ["a", "b", "c"]
    path.write_text(json.dumps(data))

    with pytest.raises(AssertionError, match="must be finite numbers"):
        PooledObjectPlacer.load(path, list(_create_test_objects()), _make_seeded_params())


def test_pooled_placer_load_rejects_layout_missing_key(tmp_path):
    """A layout missing a required field names that field."""
    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=_make_seeded_params(), pool_size=4)
    path = tmp_path / "layouts.json"
    pool.save(path)

    data = json.loads(path.read_text())
    del data["env_pools"][0][0]["final_loss"]
    path.write_text(json.dumps(data))

    with pytest.raises(AssertionError, match="missing required key 'final_loss'"):
        PooledObjectPlacer.load(path, list(_create_test_objects()), _make_seeded_params())


def test_pooled_placer_load_rejects_non_bool_heterogeneity_flag(tmp_path):
    """A non-bool uses_env_specific_bboxes is a structural problem and must fail loudly."""
    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=_make_seeded_params(), pool_size=4)
    path = tmp_path / "layouts.json"
    pool.save(path)

    data = json.loads(path.read_text())
    data["uses_env_specific_bboxes"] = "no"
    path.write_text(json.dumps(data))

    with pytest.raises(AssertionError, match="'uses_env_specific_bboxes' must be a bool"):
        PooledObjectPlacer.load(path, list(_create_test_objects()), _make_seeded_params())


def test_pooled_placer_load_restores_saved_seed_not_passed_seed(tmp_path):
    """load() must sample under the saved seed, not the seed in the freshly-passed params."""
    pool_a = PooledObjectPlacer(
        objects=list(_create_test_objects()), placer_params=_make_seeded_params(42), pool_size=6
    )
    path = tmp_path / "layouts.json"
    pool_a.save(path)

    # Load with a different seed in params; restoration must ignore it.
    loaded = PooledObjectPlacer.load(path, list(_create_test_objects()), _make_seeded_params(999))
    assert loaded._base_placement_seed == 42

    # A fresh pool under seed 42 is the reference: matching it proves the saved seed drives sampling.
    pool_a_ref = PooledObjectPlacer(
        objects=list(_create_test_objects()), placer_params=_make_seeded_params(42), pool_size=6
    )
    loaded_seq = [_positions_by_name(r) for r in loaded.sample_with_replacement(4)]
    ref_seq = [_positions_by_name(r) for r in pool_a_ref.sample_with_replacement(4)]
    assert loaded_seq == ref_seq


def test_pooled_placer_save_load_preserves_had_fallbacks(tmp_path):
    """had_fallbacks must survive the round trip so a post-load caller gating a warning isn't misled."""
    pool = PooledObjectPlacer(objects=list(_create_test_objects()), placer_params=_make_seeded_params(), pool_size=4)
    pool._had_fallbacks = True
    path = tmp_path / "layouts.json"
    pool.save(path)

    loaded = PooledObjectPlacer.load(path, list(_create_test_objects()), _make_seeded_params())
    assert loaded.had_fallbacks
