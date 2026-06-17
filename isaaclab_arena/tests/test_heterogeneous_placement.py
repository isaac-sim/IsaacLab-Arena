# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# pyright: reportArgumentType=false, reportPrivateUsage=false
# DummyObject is a lightweight test double for ObjectBase; a few pool tests also
# inspect internals directly to cover allocation edge cases.

"""Tests for heterogeneous object placement with per-env bounding boxes."""

import torch

import pytest

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.bounding_box_helpers import build_per_env_bounding_boxes, get_bounding_box_per_env
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_result import PlacementResult
from isaaclab_arena.relations.placement_validation import PlacementValidationResults
from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


def _checklist(passed: bool) -> PlacementValidationResults:
    """Single-item checklist standing in for a solved layout's validation verdict."""
    return PlacementValidationResults(validation_results={"valid": passed}, required_checks={"valid"})


# ---------------------------------------------------------------------------
# Fixture: let HeterogeneousDummyObject trigger the heterogeneous path
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_bounding_box_helpers_for_test_doubles(monkeypatch):
    """Let HeterogeneousDummyObject flow through the heterogeneous placement path.

    Production dispatch uses isinstance(RigidObjectSet), but these tests use
    lightweight DummyObject subclasses with get_bounding_box_per_env(...).
    Patch modules that bind the heterogeneous check by name at import time.
    """
    from isaaclab_arena.relations import bounding_box_helpers

    original_has_het = bounding_box_helpers.has_heterogeneous_objects
    original_bbox_per_env = bounding_box_helpers.get_bounding_box_per_env

    def has_het_with_doubles(objects):
        return original_has_het(objects) or any(hasattr(obj, "get_bounding_box_per_env") for obj in objects)

    def bbox_per_env_with_doubles(obj, num_envs):
        if hasattr(obj, "get_bounding_box_per_env"):
            return obj.get_bounding_box_per_env(num_envs)
        return original_bbox_per_env(obj, num_envs)

    has_het_sites = [
        "isaaclab_arena.relations.bounding_box_helpers.has_heterogeneous_objects",
        "isaaclab_arena.relations.pooled_object_placer.has_heterogeneous_objects",
    ]
    for site in has_het_sites:
        monkeypatch.setattr(site, has_het_with_doubles)
    monkeypatch.setattr(
        "isaaclab_arena.relations.bounding_box_helpers.get_bounding_box_per_env", bbox_per_env_with_doubles
    )


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

    def get_bounding_box_per_env(self, num_envs: int) -> AxisAlignedBoundingBox:
        """Return env-specific bbox variants for this test double."""
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
# get_bounding_box_per_env default behavior
# ---------------------------------------------------------------------------


def test_dummy_object_bbox_per_env_expands_single():
    """Default get_bounding_box_per_env should repeat the single bbox."""

    obj = DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )

    per_env = get_bounding_box_per_env(obj, 4)
    assert per_env.min_point.shape == (4, 3)
    assert per_env.max_point.shape == (4, 3)
    assert torch.allclose(per_env.min_point[0], per_env.min_point[3])


def test_per_env_bounding_boxes_formats_solver_and_env_views():
    """PerEnvBoundingBoxes should expose solver and one-env bbox formats."""
    obj = DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.3, 0.4)),
    )

    env_bboxes = build_per_env_bounding_boxes([obj], num_envs=3)
    solver_bboxes = env_bboxes.get_bounding_boxes_for_solver_candidates(candidates_per_env=2)
    per_env_bboxes = env_bboxes.get_bounding_boxes_for_all_envs()

    assert solver_bboxes[obj].min_point.shape == (6, 3)
    assert solver_bboxes[obj].max_point.shape == (6, 3)
    assert len(per_env_bboxes) == 3
    assert per_env_bboxes[1][obj].min_point.shape == (1, 3)
    assert torch.allclose(per_env_bboxes[1][obj].max_point[0], torch.tensor([0.2, 0.3, 0.4]))


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


def test_dummy_object_preserves_constructor_relations():
    """DummyObject should keep relations passed at construction time."""
    from isaaclab_arena.assets.object_set import RigidObjectSet

    anchor_relation = IsAnchor()
    obj = DummyObject(
        name="anchor",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1)),
        relations=[anchor_relation],
    )

    assert obj.get_relations() == [anchor_relation]
    assert not isinstance(obj, RigidObjectSet)


def test_object_preserves_constructor_relations():
    """Object should keep relations passed at construction time."""
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_set import RigidObjectSet

    anchor_relation = IsAnchor()
    obj = Object(
        name="rigid_object",
        object_type=ObjectType.RIGID,
        # Explicit object_type avoids USD inspection; the path is never opened in this constructor-relations test.
        usd_path="/tmp/rigid_object.usd",
        relations=[anchor_relation],
    )

    assert obj.get_relations() == [anchor_relation]
    assert not isinstance(obj, RigidObjectSet)


# ---------------------------------------------------------------------------
# Solver with per-env bboxes
# ---------------------------------------------------------------------------


def test_relation_solver_uses_env_bboxes():
    """Solver should accept env_bboxes and produce valid results."""

    desk = _make_desk()
    box = DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )
    box.add_relation(On(desk, clearance_m=0.01))

    objects = [desk, box]
    batch_size = 4

    initial_positions = [{desk: (0.0, 0.0, 0.0), box: (0.5, 0.5, 0.11)} for _ in range(batch_size)]

    # Create per-env bboxes with varying sizes across the batch.
    min_pts = torch.zeros(batch_size, 3)
    max_pts = torch.stack([torch.tensor([0.1 + 0.05 * i, 0.1 + 0.05 * i, 0.2]) for i in range(batch_size)])
    env_bbox = AxisAlignedBoundingBox(min_point=min_pts, max_point=max_pts)

    solver_params = RelationSolverParams(max_iters=100, convergence_threshold=1e-3, verbose=False)
    solver = RelationSolver(params=solver_params)
    result = solver.solve(objects, initial_positions, env_bboxes={box: env_bbox})

    assert len(result) == batch_size
    for pos_dict in result:
        assert box in pos_dict
        assert desk in pos_dict


# ---------------------------------------------------------------------------
# ObjectPlacer heterogeneous path
# ---------------------------------------------------------------------------


def test_object_placer_heterogeneous_produces_per_env_results():
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
    result = placer.place(objects, num_envs=num_envs)

    assert len(result) == num_envs
    for r in result:
        assert hetero_box in r.positions


def test_object_placer_heterogeneous_z_height_matches_variant():
    """Objects should be placed at z-height matching their env's variant bbox."""

    desk = _make_desk()

    # "tall" variant: height 0.4 -> bottom at z ~0.11 (desk top 0.1 + clearance 0.01)
    tall = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.4))
    # "short" variant: height 0.1 -> bottom at z ~0.11 (same clearance)
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
    result = placer.place(objects, num_envs=num_envs)

    assert len(result) == num_envs
    for env_idx, r in enumerate(result):
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
    result = placer.place(objects, num_envs=num_envs)

    assert len(result) == num_envs

    for env_idx, r in enumerate(result):
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


def test_heterogeneous_placement_always_returns_per_env_results():
    """Heterogeneous placement returns one layout per env solved against its variant geometry."""

    desk, hetero, _placer_params = _make_hetero_pool_objects()

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3, verbose=False)
    params = ObjectPlacerParams(
        solver_params=solver_params,
        apply_positions_to_objects=False,
        placement_seed=42,
    )

    placer = ObjectPlacer(params=params)
    result = placer.place([desk, hetero], num_envs=4)

    assert len(result) == 4


def test_object_placer_place_ranked_per_env_returns_sorted_env_lists():
    """place_ranked_per_env should return ranked candidate lists for each env."""

    desk, hetero, _placer_params = _make_hetero_pool_objects()
    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3, verbose=False)
    params = ObjectPlacerParams(
        solver_params=solver_params,
        apply_positions_to_objects=False,
        placement_seed=42,
    )

    placer = ObjectPlacer(params=params)
    ranked_results = placer.place_ranked_per_env([desk, hetero], num_envs=3, results_per_env=2)

    assert len(ranked_results) == 3
    for env_results in ranked_results:
        assert len(env_results) == 2
        assert all(hetero in result.positions for result in env_results)
        sort_keys = [(not result.success, result.final_loss) for result in env_results]
        assert sort_keys == sorted(sort_keys)

    with pytest.raises(AssertionError):
        placer.place_ranked_per_env([desk, hetero], num_envs=3, results_per_env=0)

    apply_params = ObjectPlacerParams(
        solver_params=solver_params,
        apply_positions_to_objects=True,
    )
    apply_results = ObjectPlacer(params=apply_params).place_ranked_per_env(
        [desk, hetero], num_envs=3, results_per_env=1
    )
    assert len(apply_results) == 3
    assert hetero.get_initial_pose() is None


def test_object_placer_homogeneous_objects_return_multi_env_result():
    """Homogeneous objects return one layout per env (bboxes identical across envs)."""

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
    result = placer.place([desk, box], num_envs=2)

    assert len(result) == 2
    assert all(r.success for r in result)


# ---------------------------------------------------------------------------
# PooledObjectPlacer env-specific variants
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


def test_pooled_placer_env_specific_layouts_sample_from_fixed_env_order():
    """PooledObjectPlacer should hide env routing behind the sampling strategy."""
    desk, hetero, placer_params = _make_hetero_pool_objects()
    pool = PooledObjectPlacer(objects=[desk, hetero], placer_params=placer_params, pool_size=20, num_envs=4)

    assert pool.remaining > 0
    draws = pool.sample_without_replacement(4)
    assert len(draws) == 4
    for draw in draws:
        assert hetero in draw.positions


def test_pooled_placer_heterogeneous_sample_without_replacement():
    """sample_without_replacement should return one layout per requested sample."""
    desk, hetero, placer_params = _make_hetero_pool_objects()
    pool = PooledObjectPlacer(objects=[desk, hetero], placer_params=placer_params, pool_size=20, num_envs=4)

    draws = pool.sample_without_replacement(4)
    assert len(draws) == 4
    for d in draws:
        assert hetero in d.positions


def test_pooled_placer_heterogeneous_sample_without_replacement_requires_complete_rounds():
    """sample_without_replacement should consume complete env rounds."""
    desk, hetero, placer_params = _make_hetero_pool_objects()
    pool = PooledObjectPlacer(objects=[desk, hetero], placer_params=placer_params, pool_size=20, num_envs=4)

    with pytest.raises(ValueError):
        pool.sample_without_replacement(2)


def test_pooled_placer_sample_for_envs_consumes_only_requested_envs():
    """sample_for_envs should advance only the requested absolute env pools."""
    desk, hetero, placer_params = _make_hetero_pool_objects()
    pool = PooledObjectPlacer(objects=[desk, hetero], placer_params=placer_params, pool_size=20, num_envs=4)

    for env_id in range(4):
        pool._env_pools[env_id].layouts = [
            PlacementResult(
                validation_results=_checklist(True),
                positions={hetero: (float(env_id), 0.0, 0.0)},
                final_loss=0.0,
                attempts=1,
            )
        ]
        pool._env_pools[env_id].cursor = 0

    results = pool.sample_for_envs([2, 0])

    assert list(results) == [2, 0]
    assert results[2].positions[hetero][0] == 2.0
    assert results[0].positions[hetero][0] == 0.0
    assert [env_pool.available for env_pool in pool._env_pools] == [0, 1, 0, 1]


def test_pooled_placer_heterogeneous_sample_with_replacement():
    """sample_with_replacement should return env-matched layouts without consuming."""
    desk, hetero, placer_params = _make_hetero_pool_objects()
    pool = PooledObjectPlacer(objects=[desk, hetero], placer_params=placer_params, pool_size=20, num_envs=4)

    for env_id in range(4):
        pool._env_pools[env_id].layouts = [
            PlacementResult(
                validation_results=_checklist(True),
                positions={hetero: (float(env_id), 0.0, 0.0)},
                final_loss=0.0,
                attempts=1,
            )
        ]
        pool._env_pools[env_id].cursor = 0
    initial_remaining = pool.remaining
    samples = pool.sample_with_replacement(8)
    assert [sample.positions[hetero][0] for sample in samples] == [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]
    assert pool.remaining == initial_remaining, "sample_with_replacement should not consume layouts"


def test_pooled_placer_heterogeneous_sample_with_replacement_reproducible_per_env_id():
    """sample_with_replacement should reproduce each env's layout under a fixed seed (env-specific branch)."""
    num_envs = 4
    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3, verbose=False)
    placer_params = ObjectPlacerParams(solver_params=solver_params, placement_seed=42)

    desk1, hetero1, _ = _make_hetero_pool_objects()
    desk2, hetero2, _ = _make_hetero_pool_objects()
    pool1 = PooledObjectPlacer(objects=[desk1, hetero1], placer_params=placer_params, pool_size=20, num_envs=num_envs)
    pool2 = PooledObjectPlacer(objects=[desk2, hetero2], placer_params=placer_params, pool_size=20, num_envs=num_envs)

    samples1 = pool1.sample_with_replacement(num_envs * 3)
    samples2 = pool2.sample_with_replacement(num_envs * 3)

    for i, (s1, s2) in enumerate(zip(samples1, samples2)):
        assert (
            s1.positions[hetero1] == s2.positions[hetero2]
        ), f"slot {i} (env {i % num_envs}) not reproducible: {s1.positions[hetero1]} != {s2.positions[hetero2]}"


def test_pooled_placer_heterogeneous_sample_without_replacement_triggers_refill():
    """Exhausting an env pool should trigger a refill."""
    desk, hetero, placer_params = _make_hetero_pool_objects()
    pool = PooledObjectPlacer(objects=[desk, hetero], placer_params=placer_params, pool_size=4, num_envs=2)

    initial_remaining = pool.remaining

    for _ in range(initial_remaining):
        pool.sample_without_replacement(2)

    # Pool should be exhausted now; request more to trigger refill
    draws = pool.sample_without_replacement(2)
    assert len(draws) == 2, "Pool should refill and return requested layouts"


def test_pooled_placer_seeded_refills_are_reproducible():
    """Two seeded pools should produce the same layout sequence across a refill."""
    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3, verbose=False)
    placer_params = ObjectPlacerParams(
        solver_params=solver_params,
        placement_seed=11,
        max_placement_attempts=2,
    )

    def _draw_sequence():
        desk, hetero, _placer_params = _make_hetero_pool_objects()
        pool = PooledObjectPlacer(objects=[desk, hetero], placer_params=placer_params, pool_size=2, num_envs=2)
        first_round = pool.sample_without_replacement(2)
        refill_round = pool.sample_without_replacement(2)
        return [result.positions[hetero] for result in first_round + refill_round]

    assert _draw_sequence() == _draw_sequence()


def test_pooled_placer_env_specific_fallbacks_are_reported(capsys):
    """Env-specific best-loss fallbacks should be reported to callers."""
    desk, hetero, placer_params = _make_hetero_pool_objects()
    pool = PooledObjectPlacer(objects=[desk, hetero], placer_params=placer_params, pool_size=2, num_envs=2)
    for env_pool in pool._env_pools:
        env_pool.layouts = []
        env_pool.cursor = 0
    pool._had_fallbacks = False

    fallback_results = [
        [
            PlacementResult(
                validation_results=_checklist(False),
                positions={hetero: (float(cur_env), 0.0, 0.0)},
                final_loss=1.0,
                attempts=1,
            )
        ]
        for cur_env in range(2)
    ]

    pool._store_env_matched_results(fallback_results, layouts_per_env=1, target_per_env=1, allow_fallback=True)
    captured = capsys.readouterr()

    assert pool.had_fallbacks
    assert "Falling back to best-loss layouts" in captured.out
    assert [env_pool.available for env_pool in pool._env_pools] == [1, 1]


def test_pooled_placer_env_specific_fallbacks_wait_for_final_retry(capsys):
    """Invalid env-specific candidates should not fill pools before fallback is allowed."""
    desk, hetero, placer_params = _make_hetero_pool_objects()
    pool = PooledObjectPlacer(objects=[desk, hetero], placer_params=placer_params, pool_size=2, num_envs=2)
    for env_pool in pool._env_pools:
        env_pool.layouts = []
        env_pool.cursor = 0
    pool._had_fallbacks = False

    fallback_results = [
        [
            PlacementResult(
                validation_results=_checklist(False),
                positions={hetero: (float(cur_env), 0.0, 0.0)},
                final_loss=1.0,
                attempts=1,
            )
        ]
        for cur_env in range(2)
    ]

    pool._store_env_matched_results(fallback_results, layouts_per_env=1, target_per_env=1)
    captured = capsys.readouterr()

    assert not pool.had_fallbacks
    assert "Falling back to best-loss layouts" not in captured.out
    assert [env_pool.available for env_pool in pool._env_pools] == [0, 0]


def test_pooled_placer_env_specific_fallback_only_fills_short_env(capsys):
    """Final-batch fallback should not overfill env pools that already met the target."""
    desk, hetero, placer_params = _make_hetero_pool_objects()
    pool = PooledObjectPlacer(objects=[desk, hetero], placer_params=placer_params, pool_size=2, num_envs=2)
    for env_pool in pool._env_pools:
        env_pool.layouts = []
        env_pool.cursor = 0
    pool._had_fallbacks = False

    existing_layout = PlacementResult(
        validation_results=_checklist(True),
        positions={hetero: (10.0, 0.0, 0.0)},
        final_loss=0.0,
        attempts=1,
    )
    pool._env_pools[0].append(existing_layout)

    fallback_results = [
        [
            PlacementResult(
                validation_results=_checklist(False),
                positions={hetero: (float(cur_env), 0.0, 0.0)},
                final_loss=1.0,
                attempts=1,
            )
        ]
        for cur_env in range(2)
    ]

    pool._store_env_matched_results(
        fallback_results,
        layouts_per_env=1,
        allow_fallback=True,
        target_per_env=1,
    )
    captured = capsys.readouterr()

    assert pool.had_fallbacks
    assert "envs: [1]" in captured.out
    assert [env_pool.available for env_pool in pool._env_pools] == [1, 1]
    assert pool._env_pools[0].layouts == [existing_layout]
    assert pool._env_pools[1].layouts[0] is fallback_results[1][0]


def test_pooled_placer_env_specific_valid_results_only_fill_short_envs():
    """Refills should not grow env pools that already met the target."""
    desk, hetero, placer_params = _make_hetero_pool_objects()
    pool = PooledObjectPlacer(objects=[desk, hetero], placer_params=placer_params, pool_size=2, num_envs=2)
    for env_pool in pool._env_pools:
        env_pool.layouts = []
        env_pool.cursor = 0

    existing_layout = PlacementResult(
        validation_results=_checklist(True),
        positions={hetero: (10.0, 0.0, 0.0)},
        final_loss=0.0,
        attempts=1,
    )
    pool._env_pools[0].append(existing_layout)

    ranked_results = [
        [
            PlacementResult(
                validation_results=_checklist(True),
                positions={hetero: (float(cur_env), float(candidate_idx), 0.0)},
                final_loss=0.0,
                attempts=1,
            )
            for candidate_idx in range(2)
        ]
        for cur_env in range(2)
    ]

    pool._store_env_matched_results(ranked_results, layouts_per_env=2, target_per_env=1)

    assert [env_pool.available for env_pool in pool._env_pools] == [1, 1]
    assert pool._env_pools[0].layouts == [existing_layout]
    assert pool._env_pools[1].layouts == [ranked_results[1][0]]


def test_pooled_placer_reusable_layouts_report_complete_env_rounds():
    """Reusable layouts should still expose equal without-replacement capacity per env."""
    desk = _make_desk()
    box = DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )
    box.add_relation(On(desk, clearance_m=0.01))

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3, verbose=False)
    placer_params = ObjectPlacerParams(solver_params=solver_params, placement_seed=None)

    pool = PooledObjectPlacer(objects=[desk, box], placer_params=placer_params, pool_size=10, num_envs=4)
    assert pool.remaining == 3
    draws = pool.sample_without_replacement(4)
    assert len(draws) == 4
    assert pool.remaining == 2


def test_pooled_placer_reusable_layouts_keep_partial_valid_results():
    """Reusable layouts should not be dropped when fewer than num_envs are valid."""
    desk = _make_desk()
    box = DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )
    box.add_relation(On(desk, clearance_m=0.01))

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3, verbose=False)
    placer_params = ObjectPlacerParams(solver_params=solver_params, placement_seed=None)

    pool = PooledObjectPlacer(objects=[desk, box], placer_params=placer_params, pool_size=4, num_envs=4)
    for env_pool in pool._env_pools:
        env_pool.layouts = []
        env_pool.cursor = 0

    layouts = [
        PlacementResult(
            validation_results=_checklist(True), positions={box: (float(i), 0.0, 0.0)}, final_loss=0.0, attempts=1
        )
        for i in range(3)
    ]
    pool._store_reusable_results(layouts)

    assert sum(len(env_pool.layouts) for env_pool in pool._env_pools) == 3
    assert pool.remaining == 0


def test_pooled_placer_mixed_heterogeneous_and_homogeneous_objects():
    """A pool with mixed object types should match only per-env geometry by env."""
    desk = _make_desk()
    small = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1))
    large = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.25, 0.25, 0.25))
    hetero = HeterogeneousDummyObject(name="hetero", bboxes=[small, large])
    hetero.add_relation(On(desk, clearance_m=0.01))

    box = DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.15, 0.15, 0.15)),
    )
    box.add_relation(On(desk, clearance_m=0.01))

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3, verbose=False)
    placer_params = ObjectPlacerParams(solver_params=solver_params, placement_seed=None)

    pool = PooledObjectPlacer(objects=[desk, hetero, box], placer_params=placer_params, pool_size=40, num_envs=4)

    draws = pool.sample_without_replacement(4)
    assert len(draws) == 4
    for draw in draws:
        assert hetero in draw.positions
        assert box in draw.positions


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
    assert pool.remaining > 0

    draws = pool.sample_without_replacement(6)
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


def test_pooled_placer_multi_set_sample_without_replacement_triggers_refill():
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
    for _ in range(initial_remaining):
        pool.sample_without_replacement(6)

    draws = pool.sample_without_replacement(6)
    assert len(draws) == 6


def test_pooled_placer_per_env_pools_advance_in_complete_rounds():
    """Every env pool cursor should advance together."""
    desk, hetero, placer_params = _make_hetero_pool_objects()
    pool = PooledObjectPlacer(objects=[desk, hetero], placer_params=placer_params, pool_size=20, num_envs=4)

    initial_remaining = pool.remaining

    pool.sample_without_replacement(4)

    # One complete round was consumed, so every env lost one layout.
    assert pool.remaining == initial_remaining - 1

    draws = pool.sample_without_replacement(4)
    assert len(draws) == 4
    for d in draws:
        assert hetero in d.positions


# ---------------------------------------------------------------------------
# End-to-end with real RigidObjectSet
# ---------------------------------------------------------------------------


def test_real_rigid_object_set_through_pooled_placer():
    """Real RigidObjectSet should flow through PooledObjectPlacer without monkey-patching.

    This is an integration test that verifies the actual isinstance(RigidObjectSet)
    dispatch in bounding_box_helpers.py triggers correctly, unlike the other tests
    in this file that monkey-patch has_heterogeneous_objects.
    """
    from unittest.mock import patch

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.relations.bounding_box_helpers import has_heterogeneous_objects

    desk = _make_desk()

    can_a = Object(name="can_a", object_type=ObjectType.RIGID, usd_path="/tmp/can_a.usd")
    can_b = Object(name="can_b", object_type=ObjectType.RIGID, usd_path="/tmp/can_b.usd")
    can_a.bounding_box = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.15))
    can_b.bounding_box = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.15, 0.15, 0.2))

    with (
        patch("isaaclab_arena.assets.object_set.detect_object_type", return_value=ObjectType.RIGID),
        patch("isaaclab_arena.assets.object_set.find_shallowest_rigid_body", return_value="/rigid"),
    ):
        obj_set = RigidObjectSet(name="cans", objects=[can_a, can_b])

    obj_set.add_relation(On(desk, clearance_m=0.01))

    assert has_heterogeneous_objects([desk, obj_set])

    num_envs = 4
    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3, verbose=False)
    placer_params = ObjectPlacerParams(solver_params=solver_params, placement_seed=42)

    pool = PooledObjectPlacer(objects=[desk, obj_set], placer_params=placer_params, pool_size=20, num_envs=num_envs)

    assert pool.requires_env_indexed_layouts
    assert pool.remaining > 0

    draws = pool.sample_without_replacement(num_envs)
    assert len(draws) == num_envs
    for draw in draws:
        assert obj_set in draw.positions
        z = draw.positions[obj_set][2]
        assert abs(z - 0.11) < 0.05, f"z={z:.4f}, expected ~0.11"
