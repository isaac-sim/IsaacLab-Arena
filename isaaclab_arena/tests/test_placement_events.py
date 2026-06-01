# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for placement-on-reset event: fresh layouts on successive resets."""

import torch
from unittest.mock import MagicMock

import pytest


def _create_test_objects():
    """Create a desk (anchor) with two boxes (On + NextTo)."""

    from isaaclab_arena.assets.dummy_object import DummyObject
    from isaaclab_arena.relations.relations import IsAnchor, NextTo, On, Side
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

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

    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

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

    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_result import MultiEnvPlacementResult
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

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
    positions_box1 = [result.results[env_idx].positions[box1] for env_idx in range(num_envs)]
    any_different = any(positions_box1[i] != positions_box1[j] for i in range(num_envs) for j in range(i + 1, num_envs))
    assert any_different, "Unseeded multi-env placement should produce different positions across environments"


def test_successive_seeded_placements_produce_same_layout():
    """Two place() calls with the same seed should produce identical positions."""

    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

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


def _make_mock_env(num_envs: int, device: str = "cpu") -> MagicMock:
    """Build a mock ManagerBasedEnv with scene assets that record write calls."""

    env = MagicMock()
    env.device = device
    env.scene.env_origins = torch.zeros(num_envs, 3, device=device)

    assets: dict[str, MagicMock] = {}

    def scene_getitem(self, name: str) -> MagicMock:
        if name not in assets:
            assets[name] = MagicMock()
        return assets[name]

    env.scene.__getitem__ = scene_getitem
    env._assets = assets
    return env


def test_solve_and_place_objects_writes_poses_to_sim():
    """solve_and_place_objects should call write_root_pose_to_sim for non-anchor objects."""

    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_events import solve_and_place_objects
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

    desk, box1, box2 = _create_test_objects()
    objects = [desk, box1, box2]

    env = _make_mock_env(num_envs=1)
    env_ids = torch.tensor([0])

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3)
    placer_params = ObjectPlacerParams(solver_params=solver_params)
    pool = PooledObjectPlacer(objects=objects, placer_params=placer_params, pool_size=10)

    solve_and_place_objects(env, env_ids, objects, pool)

    # Anchor (desk) should NOT have been written.
    assert "desk" not in env._assets, "Anchor pose should not be written to sim"

    # Non-anchor objects should each get a pose and zero velocity write.
    for name in ("box1", "box2"):
        asset = env._assets[name]
        asset.write_root_pose_to_sim.assert_called_once()
        asset.write_root_velocity_to_sim.assert_called_once()

        pose_arg = asset.write_root_pose_to_sim.call_args[0][0]
        assert pose_arg.shape == (1, 7), f"Expected (1,7) pose tensor for {name}, got {pose_arg.shape}"


def test_solve_and_place_objects_skips_empty_env_ids():
    """solve_and_place_objects should return immediately for an empty env_ids tensor."""

    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_events import solve_and_place_objects
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

    desk, box1, box2 = _create_test_objects()
    env = _make_mock_env(num_envs=1)

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3)
    placer_params = ObjectPlacerParams(solver_params=solver_params)
    pool = PooledObjectPlacer(objects=[desk, box1, box2], placer_params=placer_params, pool_size=10)

    solve_and_place_objects(env, torch.tensor([], dtype=torch.int64), [desk, box1, box2], pool)

    assert len(env._assets) == 0, "No writes should occur for empty env_ids"


def test_solve_and_place_objects_skips_none_env_ids():
    """solve_and_place_objects should return immediately when env_ids is None."""

    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_events import solve_and_place_objects
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

    desk, box1, box2 = _create_test_objects()
    env = _make_mock_env(num_envs=1)

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3)
    placer_params = ObjectPlacerParams(solver_params=solver_params)
    pool = PooledObjectPlacer(objects=[desk, box1, box2], placer_params=placer_params, pool_size=10)

    solve_and_place_objects(env, None, [desk, box1, box2], pool)

    assert len(env._assets) == 0, "No writes should occur for None env_ids"


def test_solve_and_place_objects_handles_multiple_env_ids():
    """solve_and_place_objects should write poses for each resetting environment."""

    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_events import solve_and_place_objects
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

    desk, box1, box2 = _create_test_objects()
    objects = [desk, box1, box2]
    num_envs = 4

    env = _make_mock_env(num_envs=num_envs)
    env_ids = torch.tensor([0, 2])

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3)
    placer_params = ObjectPlacerParams(solver_params=solver_params)
    pool = PooledObjectPlacer(objects=objects, placer_params=placer_params, pool_size=12, num_envs=num_envs)

    solve_and_place_objects(env, env_ids, objects, pool)

    assert "desk" not in env._assets, "Anchor pose should not be written to sim"

    for name in ("box1", "box2"):
        asset = env._assets[name]
        assert asset.write_root_pose_to_sim.call_count == 2, (
            f"Expected 2 write_root_pose_to_sim calls for {name} (one per reset env), "
            f"got {asset.write_root_pose_to_sim.call_count}"
        )


def test_solve_and_place_objects_partial_reset_reusable_pool_consumes_only_reset_envs():
    """Reusable layouts should not consume a full env round for a partial reset."""

    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_events import solve_and_place_objects
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

    desk, box1, box2 = _create_test_objects()
    objects = [desk, box1, box2]
    num_envs = 4
    env_ids = torch.tensor([2])

    env = _make_mock_env(num_envs=num_envs)
    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3)
    placer_params = ObjectPlacerParams(solver_params=solver_params)
    pool = PooledObjectPlacer(objects=objects, placer_params=placer_params, pool_size=12, num_envs=num_envs)

    available_before = pool.total_remaining
    solve_and_place_objects(env, env_ids, objects, pool)
    available_after = pool.total_remaining

    assert available_before - available_after == len(env_ids)


def test_solve_and_place_objects_writes_invalid_fallback_layout(capsys):
    """Invalid fallback layouts should still be written, matching pool fallback behavior."""

    from isaaclab_arena.relations.placement_events import solve_and_place_objects
    from isaaclab_arena.relations.placement_result import PlacementResult

    desk, box1, box2 = _create_test_objects()
    objects = [desk, box1, box2]
    env = _make_mock_env(num_envs=1)

    class InvalidPool:
        requires_env_indexed_layouts = False

        def sample_without_replacement(self, count: int) -> list[PlacementResult]:
            assert count == 1
            return [
                PlacementResult(
                    success=False,
                    positions={box1: (0.0, 0.0, 0.0), box2: (0.0, 0.0, 0.0)},
                    final_loss=float("nan"),
                    attempts=1,
                )
            ]

    solve_and_place_objects(env, torch.tensor([0]), objects, InvalidPool())
    captured = capsys.readouterr()

    assert set(env._assets) == {box1.name, box2.name}
    assert env._assets[box1.name].write_root_pose_to_sim.call_count == 1
    assert env._assets[box2.name].write_root_pose_to_sim.call_count == 1
    assert "Writing best-loss fallback placement" in captured.out


def test_solve_and_place_objects_partial_reset_env_indexed_uses_absolute_env_result():
    """Env-indexed partial resets should write the result for each absolute env id."""

    from isaaclab_arena.relations.placement_events import solve_and_place_objects
    from isaaclab_arena.relations.placement_result import PlacementResult

    desk, box1, box2 = _create_test_objects()
    objects = [desk, box1, box2]
    env = _make_mock_env(num_envs=4)

    class EnvIndexedPool:
        requires_env_indexed_layouts = True
        num_envs = 4
        requested_env_ids = None

        def sample_without_replacement(self, count: int) -> list[PlacementResult]:
            raise AssertionError(f"partial reset should not consume a full env round, got count={count}")

        def sample_for_envs(self, env_ids: list[int]) -> dict[int, PlacementResult]:
            self.requested_env_ids = env_ids
            return {
                cur_env: PlacementResult(
                    success=True,
                    positions={
                        box1: (float(cur_env), 0.0, 0.0),
                        box2: (float(cur_env), 1.0, 0.0),
                    },
                    final_loss=0.0,
                    attempts=1,
                )
                for cur_env in env_ids
            }

    pool = EnvIndexedPool()
    solve_and_place_objects(env, torch.tensor([2]), objects, pool)

    box1_pose = env._assets[box1.name].write_root_pose_to_sim.call_args[0][0]
    box2_pose = env._assets[box2.name].write_root_pose_to_sim.call_args[0][0]
    box1_env_id = env._assets[box1.name].write_root_pose_to_sim.call_args.kwargs["env_ids"]
    box2_env_id = env._assets[box2.name].write_root_pose_to_sim.call_args.kwargs["env_ids"]
    assert box1_pose[0, 0].item() == 2.0
    assert box2_pose[0, 0].item() == 2.0
    assert box1_env_id.tolist() == [2]
    assert box2_env_id.tolist() == [2]
    assert pool.requested_env_ids == [2]


def test_solve_and_place_objects_asserts_env_indexed_pool_size_matches_scene():
    """Env-indexed pool slots must line up with absolute Isaac Lab env ids."""

    from isaaclab_arena.relations.placement_events import solve_and_place_objects

    desk, box1, box2 = _create_test_objects()
    objects = [desk, box1, box2]
    env = _make_mock_env(num_envs=2)

    class MismatchedEnvIndexedPool:
        requires_env_indexed_layouts = True
        num_envs = 1

    with pytest.raises(ValueError, match="scene has 2 env origins"):
        solve_and_place_objects(env, torch.tensor([0]), objects, MismatchedEnvIndexedPool())


def test_pooled_placer_sample_without_replacement_returns_different_layouts():
    """sample_without_replacement() should return layouts (likely different across draws)."""

    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

    desk, box1, box2 = _create_test_objects()
    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3)
    placer_params = ObjectPlacerParams(solver_params=solver_params, placement_seed=None)
    pool = PooledObjectPlacer(objects=[desk, box1, box2], placer_params=placer_params, pool_size=20)

    assert pool.remaining == 20

    layouts = pool.sample_without_replacement(5)
    assert len(layouts) == 5
    positions = [layout.positions[box1] for layout in layouts]
    any_different = any(
        positions[i] != positions[j] for i in range(len(positions)) for j in range(i + 1, len(positions))
    )
    assert any_different, "Pool should produce different layouts"


def test_pooled_object_placer_sample_with_replacement_does_not_consume():
    """sample_with_replacement() should return layouts without consuming them."""

    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

    desk, box1, box2 = _create_test_objects()
    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3)
    placer_params = ObjectPlacerParams(solver_params=solver_params, placement_seed=None)
    pool = PooledObjectPlacer(objects=[desk, box1, box2], placer_params=placer_params, pool_size=10)

    initial_available = pool.remaining

    layouts = pool.sample_with_replacement(5)
    assert len(layouts) == 5
    assert pool.remaining == initial_available, "sample_with_replacement() should not consume from available queue"


def test_pooled_object_placer_sample_without_replacement_triggers_refill():
    """Exhausting the pool and requesting more should trigger a refill."""

    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

    desk, box1, box2 = _create_test_objects()
    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3)
    placer_params = ObjectPlacerParams(solver_params=solver_params, placement_seed=None)
    pool = PooledObjectPlacer(objects=[desk, box1, box2], placer_params=placer_params, pool_size=5)

    # Exhaust the pool, then request more
    pool.sample_without_replacement(5)
    assert pool.remaining == 0

    layouts = pool.sample_without_replacement(3)
    assert len(layouts) == 3, "sample_without_replacement() should refill and return the requested count"


def test_resolve_on_reset_false_applies_pose_per_env():
    """Simulates the resolve_on_reset=False path: sample layouts and build PosePerEnv per object."""

    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_events import get_rotation_xyzw
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import get_anchor_objects
    from isaaclab_arena.utils.pose import Pose, PosePerEnv

    desk, box1, box2 = _create_test_objects()
    objects = [desk, box1, box2]
    num_envs = 3

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3)
    placer_params = ObjectPlacerParams(solver_params=solver_params, placement_seed=None)
    pool = PooledObjectPlacer(objects=objects, placer_params=placer_params, pool_size=21, num_envs=num_envs)

    layouts = pool.sample_with_replacement(num_envs)
    assert len(layouts) == num_envs

    anchor_objects = set(get_anchor_objects(objects))
    for obj in objects:
        if obj in anchor_objects:
            continue
        rotation_xyzw = get_rotation_xyzw(obj)
        poses = [
            Pose(position_xyz=layouts[env_idx].positions[obj], rotation_xyzw=rotation_xyzw)
            for env_idx in range(num_envs)
        ]
        pose_per_env = PosePerEnv(poses=poses)
        assert len(pose_per_env.poses) == num_envs, f"Expected {num_envs} poses for {obj.name}"
        for p in pose_per_env.poses:
            assert p.position_xyz is not None, f"Position should not be None for {obj.name}"


def test_env_indexed_pool_seeds_init_state_before_reset_without_event():
    """Env-indexed resolve-on-reset path should seed non-anchor initial poses."""
    from types import SimpleNamespace

    from isaaclab_arena.environments.relation_solver_interface import _apply_dynamic_spawn_pose
    from isaaclab_arena.relations.placement_result import PlacementResult

    class MinimalObject:
        def __init__(self, name: str):
            self.name = name
            self.event_cfg = None
            self.object_cfg = SimpleNamespace(init_state=SimpleNamespace(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)))

        def get_relations(self):
            return []

        def set_initial_pose(self, pose):
            raise AssertionError("resolve_on_reset init seeding must not register per-object reset events")

    class EnvIndexedPool:
        requires_env_indexed_layouts = True
        num_envs = 3
        sample_count = None

        def sample_with_replacement(self, count: int):
            self.sample_count = count
            assert count == 1
            return [
                PlacementResult(
                    success=True,
                    positions={box: (float(env_id), 0.0, 0.1)},
                    final_loss=0.0,
                    attempts=1,
                )
                for env_id in range(self.num_envs)
            ]

    anchor = MinimalObject("desk")
    box = MinimalObject("box")
    pool = EnvIndexedPool()

    _apply_dynamic_spawn_pose(
        objects=[anchor, box],
        placement_pool=pool,
        anchor_objects_set={anchor},
    )

    assert pool.sample_count == 1
    assert anchor.object_cfg.init_state.pos == (0.0, 0.0, 0.0)
    assert box.object_cfg.init_state.pos == (0.0, 0.0, 0.1)
    assert box.event_cfg is None


def test_env_indexed_static_poses_apply_per_env_positions():
    """Static initial poses should apply per-env positions from env-indexed layouts."""
    from isaaclab_arena.assets.dummy_object import DummyObject
    from isaaclab_arena.environments.relation_solver_interface import _apply_static_initial_poses
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose, PosePerEnv

    desk = DummyObject(
        name="desk",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())

    box = DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )
    box.add_relation(On(desk, clearance_m=0.01))

    num_envs = 3

    class PerEnvPool:
        requires_env_indexed_layouts = True
        num_envs = 3

        def sample_with_replacement(self, count: int):
            return [
                PlacementResult(
                    success=True,
                    positions={box: (0.1 * env_id, 0.2 * env_id, 0.11)},
                    final_loss=0.0,
                    attempts=1,
                )
                for env_id in range(count)
            ]

    _apply_static_initial_poses(
        objects=[desk, box],
        placement_pool=PerEnvPool(),
        anchor_objects_set={desk},
        num_envs=num_envs,
    )

    pose = box.get_initial_pose()
    assert isinstance(pose, PosePerEnv)
    assert len(pose.poses) == num_envs
    for env_id in range(num_envs):
        assert pose.poses[env_id].position_xyz == (0.1 * env_id, 0.2 * env_id, 0.11)


def test_pooled_placer_falls_back_when_no_valid_layouts(capsys):
    """PooledObjectPlacer should keep best-loss fallback layouts when validation rejects all candidates."""

    from isaaclab_arena.assets.dummy_object import DummyObject
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    desk = DummyObject(
        name="desk",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.01, 0.01, 0.01)),
    )
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())

    # Two large boxes that cannot both fit On a tiny desk
    big1 = DummyObject(
        name="big1",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(5.0, 5.0, 5.0)),
    )
    big2 = DummyObject(
        name="big2",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(5.0, 5.0, 5.0)),
    )
    big1.add_relation(On(desk))
    big2.add_relation(On(desk))

    solver_params = RelationSolverParams(max_iters=50, convergence_threshold=1e-6)
    placer_params = ObjectPlacerParams(solver_params=solver_params, max_placement_attempts=1)

    pool = PooledObjectPlacer(objects=[desk, big1, big2], placer_params=placer_params, pool_size=5)
    captured = capsys.readouterr()

    assert pool.remaining == 5
    assert pool.had_fallbacks
    assert "Accepting best-loss layouts as fallback" in captured.out
    assert not pool.sample_without_replacement(1)[0].success
