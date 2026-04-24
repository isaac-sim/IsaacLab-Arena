# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for heterogeneous object placement with per-env bounding boxes."""

import torch

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_result import MultiEnvPlacementResult
from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class HeterogeneousDummyObject(DummyObject):
    """DummyObject that provides different bounding boxes per environment.

    Used to exercise the heterogeneous placement path without requiring
    RigidObjectSet's USD machinery.
    """

    def __init__(self, name: str, bboxes: list[AxisAlignedBoundingBox], **kwargs):
        super().__init__(name=name, bounding_box=bboxes[0], **kwargs)
        self._per_env_bboxes = bboxes
        self.heterogeneous_bbox = True
        self.objects = bboxes

    def get_bounding_box_per_env(self, num_envs: int) -> AxisAlignedBoundingBox:
        n_variants = len(self._per_env_bboxes)
        indices = [i % n_variants for i in range(num_envs)]
        min_pts = torch.stack([self._per_env_bboxes[idx].min_point[0] for idx in indices])
        max_pts = torch.stack([self._per_env_bboxes[idx].max_point[0] for idx in indices])
        return AxisAlignedBoundingBox(min_point=min_pts, max_point=max_pts)


def _make_desk() -> DummyObject:
    desk = DummyObject(
        name="desk",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())
    return desk


# ---------------------------------------------------------------------------
# ObjectBase.get_bounding_box_per_env
# ---------------------------------------------------------------------------


def test_dummy_object_bbox_per_env_expands_single():
    """Default get_bounding_box_per_env should repeat the single bbox."""

    obj = DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )

    per_env = obj.get_bounding_box_per_env(4)
    assert per_env.min_point.shape == (4, 3)
    assert per_env.max_point.shape == (4, 3)
    assert torch.allclose(per_env.min_point[0], per_env.min_point[3])


def test_heterogeneous_dummy_returns_different_bboxes():
    """HeterogeneousDummyObject should cycle through its member bboxes."""

    small = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1))
    large = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.3, 0.3, 0.3))
    obj = HeterogeneousDummyObject(name="set", bboxes=[small, large])

    per_env = obj.get_bounding_box_per_env(4)
    assert per_env.max_point.shape == (4, 3)
    # env 0 and 2 should use small; env 1 and 3 should use large
    assert torch.allclose(per_env.max_point[0], torch.tensor([0.1, 0.1, 0.1]))
    assert torch.allclose(per_env.max_point[1], torch.tensor([0.3, 0.3, 0.3]))


# ---------------------------------------------------------------------------
# Solver with per-row bboxes
# ---------------------------------------------------------------------------


def test_solver_accepts_per_row_bboxes():
    """Solver should accept bboxes_per_row and produce valid results."""

    desk = _make_desk()
    box = DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )
    box.add_relation(On(desk, clearance_m=0.01))

    objects = [desk, box]
    batch_size = 4

    initial_positions = [{desk: (0.0, 0.0, 0.0), box: (0.5, 0.5, 0.11)} for _ in range(batch_size)]

    # Create per-row bboxes with varying sizes across the batch.
    min_pts = torch.zeros(batch_size, 3)
    max_pts = torch.stack([torch.tensor([0.1 + 0.05 * i, 0.1 + 0.05 * i, 0.2]) for i in range(batch_size)])
    per_row_bbox = AxisAlignedBoundingBox(min_point=min_pts, max_point=max_pts)

    solver_params = RelationSolverParams(max_iters=100, convergence_threshold=1e-3, verbose=False)
    solver = RelationSolver(params=solver_params)
    result = solver.solve(objects, initial_positions, bboxes_per_row={box: per_row_bbox})

    assert len(result) == batch_size
    for pos_dict in result:
        assert box in pos_dict
        assert desk in pos_dict


# ---------------------------------------------------------------------------
# ObjectPlacer heterogeneous path
# ---------------------------------------------------------------------------


def test_placer_heterogeneous_produces_per_env_results():
    """Placer should detect heterogeneous objects and solve per-env."""

    desk = _make_desk()

    small = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1))
    large = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.3, 0.3, 0.3))
    hetero_box = HeterogeneousDummyObject(name="hetero_box", bboxes=[small, large])
    hetero_box.add_relation(On(desk, clearance_m=0.01))

    objects = [desk, hetero_box]
    num_envs = 4

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3, verbose=False)
    params = ObjectPlacerParams(
        solver_params=solver_params,
        apply_positions_to_objects=False,
        placement_seed=None,
    )

    placer = ObjectPlacer(params=params)
    result = placer.place(objects, num_envs=num_envs, result_per_env=True)

    assert isinstance(result, MultiEnvPlacementResult)
    assert len(result.results) == num_envs
    for r in result.results:
        assert hetero_box in r.positions


def test_placer_heterogeneous_z_height_matches_variant():
    """Objects should be placed at z-height matching their env's variant bbox."""

    desk = _make_desk()

    # "tall" variant: height 0.4 → bottom at z ≈ 0.11 (desk top 0.1 + clearance 0.01)
    tall = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.4))
    # "short" variant: height 0.1 → bottom at z ≈ 0.11 (same clearance)
    short = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.1))
    hetero = HeterogeneousDummyObject(name="hetero", bboxes=[tall, short])
    hetero.add_relation(On(desk, clearance_m=0.01))

    objects = [desk, hetero]
    num_envs = 2

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3, verbose=False)
    params = ObjectPlacerParams(
        solver_params=solver_params,
        apply_positions_to_objects=False,
        placement_seed=42,
    )

    placer = ObjectPlacer(params=params)
    result = placer.place(objects, num_envs=num_envs, result_per_env=True)

    assert isinstance(result, MultiEnvPlacementResult)
    # Both envs should have solved z near the desk top + clearance (0.11).
    # The On loss targets: z = parent_top + clearance - child_min_z = 0.1 + 0.01 - 0.0 = 0.11
    for env_idx, r in enumerate(result.results):
        z = r.positions[hetero][2]
        assert abs(z - 0.11) < 0.05, f"Env {env_idx}: z={z:.4f}, expected ~0.11"


def test_mixed_heterogeneous_and_homogeneous_placement():
    """Mixed scene: heterogeneous A (RigidObjectSet-like) + homogeneous X (plain Object).

    Both sit On(desk) with NoCollision between them. The solver must produce
    valid, non-overlapping placements in every env even though A has different
    bboxes per env while X is identical everywhere.
    """

    desk = _make_desk()

    # A: heterogeneous — small variant in even envs, large in odd envs.
    small_a = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1))
    large_a = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.25, 0.25, 0.25))
    obj_a = HeterogeneousDummyObject(name="A", bboxes=[small_a, large_a])
    obj_a.add_relation(On(desk, clearance_m=0.01))

    # X: homogeneous — same bbox in all envs.
    obj_x = DummyObject(
        name="X",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.15, 0.15, 0.15)),
    )
    obj_x.add_relation(On(desk, clearance_m=0.01))

    # No-overlap is handled automatically by the solver's built-in clearance.

    objects = [desk, obj_a, obj_x]
    num_envs = 4

    solver_params = RelationSolverParams(max_iters=300, convergence_threshold=1e-3, verbose=False)
    params = ObjectPlacerParams(
        solver_params=solver_params,
        apply_positions_to_objects=False,
        placement_seed=42,
    )

    placer = ObjectPlacer(params=params)
    result = placer.place(objects, num_envs=num_envs, result_per_env=True)

    assert isinstance(result, MultiEnvPlacementResult)
    assert len(result.results) == num_envs

    for env_idx, r in enumerate(result.results):
        assert obj_a in r.positions and obj_x in r.positions
        # Verify z-height is near desk top + clearance for both objects.
        for obj in (obj_a, obj_x):
            z = r.positions[obj][2]
            assert abs(z - 0.11) < 0.05, f"Env {env_idx}, {obj.name}: z={z:.4f}, expected ~0.11"

        # Verify XY non-overlap using each env's actual variant bbox.
        variant_idx = env_idx % len([small_a, large_a])
        a_bbox = [small_a, large_a][variant_idx]
        x_bbox = obj_x.get_bounding_box()
        a_world = a_bbox.translated(r.positions[obj_a])
        x_world = x_bbox.translated(r.positions[obj_x])
        assert not a_world.overlaps(
            x_world
        ).item(), f"Env {env_idx}: A and X bboxes overlap at positions A={r.positions[obj_a]}, X={r.positions[obj_x]}"


def test_homogeneous_path_unchanged():
    """When no heterogeneous objects exist, the homogeneous path is used."""

    desk = _make_desk()
    box = DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )
    box.add_relation(On(desk, clearance_m=0.01))

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3, verbose=False)
    params = ObjectPlacerParams(
        solver_params=solver_params,
        apply_positions_to_objects=False,
        placement_seed=None,
    )

    placer = ObjectPlacer(params=params)
    result = placer.place([desk, box], num_envs=2, result_per_env=True)

    assert isinstance(result, MultiEnvPlacementResult)
    assert len(result.results) == 2


# ---------------------------------------------------------------------------
# PooledObjectPlacer heterogeneous mode
# ---------------------------------------------------------------------------


def _make_hetero_pool_objects():
    """Create desk + heterogeneous box for pool tests."""
    desk = _make_desk()
    small = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1))
    large = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.3, 0.3, 0.3))
    hetero = HeterogeneousDummyObject(name="hetero", bboxes=[small, large])
    hetero.add_relation(On(desk, clearance_m=0.01))

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3, verbose=False)
    placer_params = ObjectPlacerParams(solver_params=solver_params, placement_seed=None)
    return desk, hetero, placer_params


def test_pooled_placer_heterogeneous_is_detected():
    """PooledObjectPlacer should detect heterogeneous objects and create variant sub-pools."""
    desk, hetero, placer_params = _make_hetero_pool_objects()
    pool = PooledObjectPlacer(objects=[desk, hetero], placer_params=placer_params, pool_size=20, num_envs=4)

    assert pool.is_heterogeneous
    assert pool.remaining > 0


def test_pooled_placer_heterogeneous_sample_without_replacement():
    """sample_without_replacement with env_ids should return one layout per env from correct variant."""
    desk, hetero, placer_params = _make_hetero_pool_objects()
    pool = PooledObjectPlacer(objects=[desk, hetero], placer_params=placer_params, pool_size=20, num_envs=4)

    env_ids = torch.tensor([0, 1, 2, 3])
    draws = pool.sample_without_replacement(4, env_ids=env_ids)
    assert len(draws) == 4
    for d in draws:
        assert hetero in d.positions


def test_pooled_placer_heterogeneous_sample_without_replacement_requires_env_ids():
    """Heterogeneous pool should assert when env_ids is not provided."""
    desk, hetero, placer_params = _make_hetero_pool_objects()
    pool = PooledObjectPlacer(objects=[desk, hetero], placer_params=placer_params, pool_size=20, num_envs=4)

    try:
        pool.sample_without_replacement(2, env_ids=None)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


def test_pooled_placer_heterogeneous_sample_with_replacement():
    """sample_with_replacement should return per-variant layouts without consuming."""
    desk, hetero, placer_params = _make_hetero_pool_objects()
    pool = PooledObjectPlacer(objects=[desk, hetero], placer_params=placer_params, pool_size=20, num_envs=4)

    initial_remaining = pool.remaining
    samples = pool.sample_with_replacement(4)
    assert len(samples) == 4
    assert pool.remaining == initial_remaining, "sample_with_replacement should not consume layouts"


def test_pooled_placer_heterogeneous_refill():
    """Exhausting a variant sub-pool should trigger a refill."""
    desk, hetero, placer_params = _make_hetero_pool_objects()
    pool = PooledObjectPlacer(objects=[desk, hetero], placer_params=placer_params, pool_size=4, num_envs=2)

    initial_remaining = pool.remaining

    # Draw all layouts for env 0 (variant 0) and env 1 (variant 1)
    env_ids = torch.tensor([0, 1] * initial_remaining)
    pool.sample_without_replacement(len(env_ids), env_ids=env_ids)

    # Pool should be exhausted now; request more to trigger refill
    env_ids_more = torch.tensor([0, 1])
    draws = pool.sample_without_replacement(2, env_ids=env_ids_more)
    assert len(draws) == 2, "Pool should refill and return requested layouts"


def test_pooled_placer_homogeneous_unaffected_by_num_envs():
    """Homogeneous pool should work the same whether num_envs is passed or not."""
    desk = _make_desk()
    box = DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )
    box.add_relation(On(desk, clearance_m=0.01))

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3, verbose=False)
    placer_params = ObjectPlacerParams(solver_params=solver_params, placement_seed=None)

    pool = PooledObjectPlacer(objects=[desk, box], placer_params=placer_params, pool_size=10, num_envs=4)
    assert not pool.is_heterogeneous
    draws = pool.sample_without_replacement(3)
    assert len(draws) == 3


# ---------------------------------------------------------------------------
# Multi-set heterogeneous: different variant counts across objects
# ---------------------------------------------------------------------------


def test_pooled_placer_multi_set_different_variant_counts():
    """Pool with two heterogeneous objects having different variant counts.

    Bottles (3 variants) and boxes (2 variants) across 6 envs.
    Each env gets its own pool with layouts matching its object geometry.
    """
    desk = _make_desk()

    bottle_small = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.08, 0.08, 0.2))
    bottle_medium = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.25))
    bottle_large = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.12, 0.12, 0.3))
    bottles = HeterogeneousDummyObject(name="bottles", bboxes=[bottle_small, bottle_medium, bottle_large])
    bottles.add_relation(On(desk, clearance_m=0.01))

    box_small = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.15, 0.1, 0.1))
    box_large = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.15, 0.15))
    boxes = HeterogeneousDummyObject(name="boxes", bboxes=[box_small, box_large])
    boxes.add_relation(On(desk, clearance_m=0.01))

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3, verbose=False)
    placer_params = ObjectPlacerParams(solver_params=solver_params, placement_seed=None)

    pool = PooledObjectPlacer(objects=[desk, bottles, boxes], placer_params=placer_params, pool_size=50, num_envs=6)
    assert pool.is_heterogeneous
    assert pool.remaining > 0

    env_ids = torch.tensor([0, 1, 2, 3, 4, 5])
    draws = pool.sample_without_replacement(6, env_ids=env_ids)
    assert len(draws) == 6
    for d in draws:
        assert bottles in d.positions
        assert boxes in d.positions


def test_pooled_placer_multi_set_sample_with_replacement():
    """sample_with_replacement with multi-set heterogeneous objects."""
    desk = _make_desk()

    a_s = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.08, 0.08, 0.15))
    a_m = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.2))
    a_l = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.12, 0.12, 0.25))
    obj_a = HeterogeneousDummyObject(name="A", bboxes=[a_s, a_m, a_l])
    obj_a.add_relation(On(desk, clearance_m=0.01))

    b_s = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1))
    b_l = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.15, 0.12))
    obj_b = HeterogeneousDummyObject(name="B", bboxes=[b_s, b_l])
    obj_b.add_relation(On(desk, clearance_m=0.01))

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3, verbose=False)
    placer_params = ObjectPlacerParams(solver_params=solver_params, placement_seed=None)

    pool = PooledObjectPlacer(objects=[desk, obj_a, obj_b], placer_params=placer_params, pool_size=50, num_envs=6)

    initial_remaining = pool.remaining
    samples = pool.sample_with_replacement(6)
    assert len(samples) == 6
    assert pool.remaining == initial_remaining


def test_pooled_placer_multi_set_refill():
    """Exhausting a per-env pool should trigger refill with multi-set objects."""
    desk = _make_desk()

    v1 = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.15))
    v2 = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.12, 0.12, 0.2))
    v3 = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.08, 0.08, 0.18))
    obj_a = HeterogeneousDummyObject(name="A", bboxes=[v1, v2, v3])
    obj_a.add_relation(On(desk, clearance_m=0.01))

    w1 = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.15, 0.1, 0.1))
    w2 = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.18, 0.12, 0.12))
    obj_b = HeterogeneousDummyObject(name="B", bboxes=[w1, w2])
    obj_b.add_relation(On(desk, clearance_m=0.01))

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3, verbose=False)
    placer_params = ObjectPlacerParams(solver_params=solver_params, placement_seed=None)

    # 6 envs with 3+2 variant objects, small pool to force refill
    pool = PooledObjectPlacer(objects=[desk, obj_a, obj_b], placer_params=placer_params, pool_size=12, num_envs=6)

    # Drain pool then request more to trigger refill
    initial_remaining = pool.remaining
    env_ids = torch.tensor(list(range(6)) * initial_remaining)
    pool.sample_without_replacement(len(env_ids), env_ids=env_ids)

    env_ids_more = torch.tensor([0, 1, 2, 3, 4, 5])
    draws = pool.sample_without_replacement(6, env_ids=env_ids_more)
    assert len(draws) == 6


def test_pooled_placer_per_env_pools_isolated():
    """Each env_id should have its own independent pool of layouts."""
    desk, hetero, placer_params = _make_hetero_pool_objects()
    pool = PooledObjectPlacer(objects=[desk, hetero], placer_params=placer_params, pool_size=20, num_envs=4)

    initial_remaining = pool.remaining

    # Draw only from env 0 and env 1; env 2 and 3 should be unaffected.
    env_ids = torch.tensor([0, 1])
    pool.sample_without_replacement(2, env_ids=env_ids)

    # `remaining` reports the min across all envs.  Env 0 and 1 each lost one
    # layout, so the min should have decreased by 1.
    assert pool.remaining == initial_remaining - 1

    # Drawing from env 2 and 3 should still work from their full pools.
    env_ids_23 = torch.tensor([2, 3])
    draws = pool.sample_without_replacement(2, env_ids=env_ids_23)
    assert len(draws) == 2
    for d in draws:
        assert hetero in d.positions
