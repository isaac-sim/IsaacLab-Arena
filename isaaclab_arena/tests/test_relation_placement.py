# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations import relation_placement
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_events import solve_and_place_objects
from isaaclab_arena.relations.relation_placement import (
    ArenaRelationSolver,
    ObjectRelationSolver,
    ValidatedPlacementPool,
    prepare_relation_placement,
)
from isaaclab_arena.relations.relations import IsAnchor, On, RotateAroundSolution, get_anchor_objects
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose, PosePerEnv


class _FakePooledObjectPlacer:
    """Fast stand-in for PooledObjectPlacer."""

    def __init__(self, objects, placer_params: ObjectPlacerParams, pool_size: int):
        self.objects = objects
        self.placer_params = placer_params
        self.pool_size = pool_size
        anchor_objects = set(get_anchor_objects(objects))
        solved_objects = [obj for obj in objects if obj not in anchor_objects]
        self.layouts = []
        for idx in range(max(pool_size, 1)):
            positions = {
                obj: (idx + obj_idx + 0.1, idx + obj_idx + 0.2, idx + obj_idx + 0.3)
                for obj_idx, obj in enumerate(solved_objects)
            }
            self.layouts.append(SimpleNamespace(success=True, positions=positions))

    def sample_with_replacement(self, count: int):
        return self.layouts[:count]

    def sample_without_replacement(self, count: int):
        return self.layouts[:count]

    @property
    def remaining(self):
        return len(self.layouts)


class _MissingPositionPooledObjectPlacer(_FakePooledObjectPlacer):
    def __init__(self, objects, placer_params: ObjectPlacerParams, pool_size: int):
        super().__init__(objects, placer_params, pool_size)
        self.layouts = [SimpleNamespace(success=True, positions={}) for _ in range(max(pool_size, 1))]


class _InvalidObjectPooledObjectPlacer(_FakePooledObjectPlacer):
    def __init__(self, objects, placer_params: ObjectPlacerParams, pool_size: int):
        super().__init__(objects, placer_params, pool_size)
        for layout in self.layouts:
            layout.success = False


class _RejectingObjectRelationSolver(ObjectRelationSolver):
    def validate_layout(self, layout):
        raise RuntimeError("Relation placement failed object validation.")


class _RejectingLaterLayoutObjectSolver(ObjectRelationSolver):
    def validate_layout(self, layout):
        if any(pos[0] > 1.0 for pos in layout.positions.values()):
            raise RuntimeError("Relation placement failed object validation.")


def _create_objects_with_configs() -> tuple[Any, Any]:
    table: Any = DummyObject(
        name="table",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())

    box: Any = DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1)),
    )
    box.add_relation(On(table))

    for obj in (table, box):
        obj.object_cfg = SimpleNamespace(init_state=SimpleNamespace(pos=None, rot=None))
        obj.event_cfg = None

    return table, box


def test_prepare_relation_placement_skips_empty_objects(capsys):
    plan = prepare_relation_placement(objects=[], num_envs=2)

    assert plan is None
    assert "No objects with relations found" in capsys.readouterr().out


def test_prepare_relation_placement_default_creates_reset_event(monkeypatch):
    monkeypatch.setattr(relation_placement, "PooledObjectPlacer", _FakePooledObjectPlacer)
    table, box = _create_objects_with_configs()
    embodiment: Any = object()

    plan = prepare_relation_placement(objects=[table, box], num_envs=1, embodiment=embodiment)

    assert plan is not None
    assert plan.placement_event_cfg is not None


def test_prepare_relation_placement_creates_reset_event(monkeypatch):
    monkeypatch.setattr(relation_placement, "PooledObjectPlacer", _FakePooledObjectPlacer)
    table, box = _create_objects_with_configs()
    box.add_relation(RotateAroundSolution(yaw_rad=math.pi / 2))

    plan = prepare_relation_placement(
        objects=[table, box],
        num_envs=3,
        placement_seed=123,
        resolve_on_reset=True,
    )

    assert plan is not None
    assert plan.placement_event_cfg is not None
    assert plan.placement_event_cfg.func is solve_and_place_objects
    event_objects = plan.placement_event_cfg.params["objects"]
    assert [obj.name for obj in event_objects] == ["table", "box"]
    assert [obj.name for obj in event_objects] == ["table", "box"]
    assert getattr(plan.placement_event_cfg.params["placement_pool"], "pool_size") == 15
    assert table.object_cfg.init_state.pos is None
    assert box.object_cfg.init_state.pos == (0.1, 0.2, 0.3)
    assert box.object_cfg.init_state.rot == pytest.approx((0.0, 0.0, 0.7071068, 0.7071068))


def test_prepare_relation_placement_applies_static_pose_per_env(monkeypatch):
    monkeypatch.setattr(relation_placement, "PooledObjectPlacer", _FakePooledObjectPlacer)
    table, box = _create_objects_with_configs()
    box.add_relation(RotateAroundSolution(yaw_rad=math.pi / 2))

    plan = prepare_relation_placement(
        objects=[table, box],
        num_envs=2,
        resolve_on_reset=False,
    )

    assert plan is not None
    assert plan.placement_event_cfg is None
    initial_pose = box.get_initial_pose()
    assert isinstance(initial_pose, PosePerEnv)
    assert [pose.position_xyz for pose in initial_pose.poses] == [(0.1, 0.2, 0.3), (1.1, 1.2, 1.3)]
    assert all(pose.rotation_xyzw == pytest.approx((0.0, 0.0, 0.7071068, 0.7071068)) for pose in initial_pose.poses)
    assert table.get_initial_pose().position_xyz == (0.0, 0.0, 0.0)
    assert table.object_cfg.init_state.pos is None


def test_prepare_relation_placement_static_pose_per_env_supports_single_env(monkeypatch):
    monkeypatch.setattr(relation_placement, "PooledObjectPlacer", _FakePooledObjectPlacer)
    table, box = _create_objects_with_configs()

    plan = prepare_relation_placement(objects=[table, box], num_envs=1, resolve_on_reset=False)

    assert plan is not None
    initial_pose = box.get_initial_pose()
    assert isinstance(initial_pose, PosePerEnv)
    assert [pose.position_xyz for pose in initial_pose.poses] == [(0.1, 0.2, 0.3)]


@pytest.mark.parametrize("resolve_on_reset", [True, False, None])
def test_prepare_relation_placement_rejects_non_anchor_pose_reset_event(monkeypatch, resolve_on_reset):
    monkeypatch.setattr(relation_placement, "PooledObjectPlacer", _FakePooledObjectPlacer)
    table, box = _create_objects_with_configs()
    box.event_cfg = object()

    with pytest.raises(RuntimeError, match="explicit pose-reset event"):
        prepare_relation_placement(objects=[table, box], num_envs=1, resolve_on_reset=resolve_on_reset)


def test_prepare_relation_placement_raises_when_reset_layout_missing_position(monkeypatch):
    monkeypatch.setattr(relation_placement, "PooledObjectPlacer", _MissingPositionPooledObjectPlacer)
    table, box = _create_objects_with_configs()

    with pytest.raises(RuntimeError, match="missing a solved position for 'box'"):
        prepare_relation_placement(objects=[table, box], num_envs=1, resolve_on_reset=True)


def test_prepare_relation_placement_raises_when_static_layout_missing_position(monkeypatch):
    monkeypatch.setattr(relation_placement, "PooledObjectPlacer", _MissingPositionPooledObjectPlacer)
    table, box = _create_objects_with_configs()

    with pytest.raises(RuntimeError, match="missing a solved position for 'box'"):
        prepare_relation_placement(objects=[table, box], num_envs=1, resolve_on_reset=False)


def test_prepare_relation_placement_accepts_object_placer_fallback_layouts(monkeypatch):
    monkeypatch.setattr(relation_placement, "PooledObjectPlacer", _InvalidObjectPooledObjectPlacer)
    table, box = _create_objects_with_configs()

    plan = prepare_relation_placement(objects=[table, box], num_envs=1, resolve_on_reset=True)

    assert plan is not None
    assert plan.placement_event_cfg is not None


def test_prepare_relation_placement_raises_when_object_cfg_missing(monkeypatch):
    monkeypatch.setattr(relation_placement, "PooledObjectPlacer", _FakePooledObjectPlacer)
    table, box = _create_objects_with_configs()
    box.object_cfg = None

    with pytest.raises(RuntimeError, match="object_cfg"):
        prepare_relation_placement(objects=[table, box], num_envs=1, resolve_on_reset=True)


def test_dynamic_reset_event_pool_validates_sampled_layouts(monkeypatch):
    monkeypatch.setattr(relation_placement, "PooledObjectPlacer", _FakePooledObjectPlacer)
    table, box = _create_objects_with_configs()
    solver = ArenaRelationSolver(
        num_envs=1,
        object_solver=_RejectingLaterLayoutObjectSolver(num_envs=1, resolve_on_reset=True),
    )

    plan = prepare_relation_placement(objects=[table, box], num_envs=1, solver=solver)

    assert plan is not None
    assert plan.placement_event_cfg is not None
    placement_pool: Any = plan.placement_event_cfg.params["placement_pool"]
    assert isinstance(placement_pool, ValidatedPlacementPool)
    with pytest.raises(RuntimeError, match="object validation"):
        placement_pool.sample_without_replacement(2)


def test_dynamic_reset_event_pool_validates_replacement_samples(monkeypatch):
    monkeypatch.setattr(relation_placement, "PooledObjectPlacer", _FakePooledObjectPlacer)
    table, box = _create_objects_with_configs()
    solver = ArenaRelationSolver(
        num_envs=1,
        object_solver=_RejectingLaterLayoutObjectSolver(num_envs=1, resolve_on_reset=True),
    )

    plan = prepare_relation_placement(objects=[table, box], num_envs=1, solver=solver)

    assert plan is not None
    assert plan.placement_event_cfg is not None
    placement_pool: Any = plan.placement_event_cfg.params["placement_pool"]
    with pytest.raises(RuntimeError, match="object validation"):
        placement_pool.sample_with_replacement(2)


def test_validated_placement_pool_deepcopy_preserves_validation(monkeypatch):
    monkeypatch.setattr(relation_placement, "PooledObjectPlacer", _FakePooledObjectPlacer)
    table, box = _create_objects_with_configs()
    solver = ArenaRelationSolver(
        num_envs=1,
        object_solver=_RejectingLaterLayoutObjectSolver(num_envs=1, resolve_on_reset=True),
    )
    plan = prepare_relation_placement(objects=[table, box], num_envs=1, solver=solver)

    assert plan is not None
    assert plan.placement_event_cfg is not None
    placement_pool: Any = deepcopy(plan.placement_event_cfg.params["placement_pool"])
    with pytest.raises(RuntimeError, match="object validation"):
        placement_pool.sample_without_replacement(2)


def test_validated_placement_pool_exposes_only_known_api(monkeypatch):
    monkeypatch.setattr(relation_placement, "PooledObjectPlacer", _FakePooledObjectPlacer)
    table, box = _create_objects_with_configs()
    plan = prepare_relation_placement(objects=[table, box], num_envs=1, resolve_on_reset=True)

    assert plan is not None
    assert plan.placement_event_cfg is not None
    placement_pool: Any = plan.placement_event_cfg.params["placement_pool"]
    assert placement_pool.pool_size == 5
    assert placement_pool.remaining == 5
    with pytest.raises(AttributeError):
        placement_pool.sample_batched


def test_relation_solver_rejects_conflicting_object_solver_config():
    object_solver = ObjectRelationSolver(num_envs=1, placement_seed=7)

    with pytest.raises(ValueError, match="owned by object_solver"):
        ArenaRelationSolver(num_envs=1, placement_seed=7, object_solver=object_solver)


def test_relation_solver_rejects_object_solver_num_env_mismatch():
    object_solver = ObjectRelationSolver(num_envs=2)

    with pytest.raises(ValueError, match="num_envs"):
        ArenaRelationSolver(num_envs=1, object_solver=object_solver)


def test_prepare_relation_placement_rejects_solver_owned_kwargs():
    solver = ArenaRelationSolver(num_envs=1)

    with pytest.raises(ValueError, match="owned by solver"):
        prepare_relation_placement(objects=[], num_envs=1, placement_seed=7, solver=solver)


def test_prepare_relation_placement_rejects_solver_num_env_mismatch():
    solver = ArenaRelationSolver(num_envs=2)

    with pytest.raises(ValueError, match="num_envs"):
        prepare_relation_placement(objects=[], num_envs=1, solver=solver)


def test_relation_solver_stores_current_objects(monkeypatch):
    monkeypatch.setattr(relation_placement, "PooledObjectPlacer", _FakePooledObjectPlacer)
    table, box = _create_objects_with_configs()
    embodiment: Any = object()
    solver = ArenaRelationSolver(num_envs=1)

    plan = prepare_relation_placement(objects=[table, box], num_envs=1, embodiment=embodiment, solver=solver)

    assert solver.objects == [table, box]
    assert solver.embodiment is embodiment
    assert plan is not None


def test_relation_solver_reuses_current_objects(monkeypatch):
    monkeypatch.setattr(relation_placement, "PooledObjectPlacer", _FakePooledObjectPlacer)
    table_a, box_a = _create_objects_with_configs()
    table_b, box_b = _create_objects_with_configs()
    solver = ArenaRelationSolver(num_envs=1)

    first_plan = prepare_relation_placement(objects=[table_a, box_a], num_envs=1, solver=solver)
    second_plan = prepare_relation_placement(objects=[table_b, box_b], num_envs=1, solver=solver)

    assert first_plan is not None
    assert second_plan is not None
    assert solver.objects == [table_b, box_b]


def test_relation_solver_subclass_can_reject_layout_with_object_hook(monkeypatch):
    monkeypatch.setattr(relation_placement, "PooledObjectPlacer", _FakePooledObjectPlacer)
    table, box = _create_objects_with_configs()
    solver = ArenaRelationSolver(
        num_envs=1,
        object_solver=_RejectingObjectRelationSolver(num_envs=1, resolve_on_reset=True),
    )

    with pytest.raises(RuntimeError, match="object validation"):
        prepare_relation_placement(objects=[table, box], num_envs=1, solver=solver)


def test_prepare_relation_placement_supports_anchor_only_scene(monkeypatch):
    monkeypatch.setattr(relation_placement, "PooledObjectPlacer", _FakePooledObjectPlacer)
    table, _ = _create_objects_with_configs()

    plan = prepare_relation_placement(objects=[table], num_envs=1, resolve_on_reset=True)

    assert plan is not None
    assert plan.placement_event_cfg is None
    assert table.object_cfg.init_state.pos is None
