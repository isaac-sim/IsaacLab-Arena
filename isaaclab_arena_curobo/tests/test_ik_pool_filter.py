# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reject-&-refill logic tests for filter_pool_by_ik_reachability, with the sim/cuRobo mocked out.

Exercises the pool-filtering control flow (stamp -> retain reachable -> refill geometry -> re-check)
against a real geometry-solved pool, so no SimulationApp, GPU, or cuRobo solve is needed. The three
sim-touching helpers are patched: object poses are never written, and a supplied verdict function
decides reachability per candidate layout.
"""

from __future__ import annotations

from unittest.mock import MagicMock


def _make_pool(num_envs: int, min_layouts_per_env: int):
    """Build a small valid desk+box pool for filtering tests."""
    from isaaclab_arena.assets.dummy_object import DummyObject
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

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

    params = ObjectPlacerParams(
        solver_params=RelationSolverParams(max_iters=200, convergence_threshold=1e-3),
        apply_positions_to_objects=False,
        min_unique_layouts_per_env=min_layouts_per_env,
        placement_seed=11,
    )
    return PooledObjectPlacer(
        objects=[desk, box],
        placer_params=params,
        pool_size=num_envs * min_layouts_per_env,
        num_envs=num_envs,
    )


def _patch_sim(monkeypatch, verdict_fn):
    """Replace the sim-touching helpers so the filter loop runs without a sim/cuRobo.

    ``write_layout_to_sim`` records the layout currently under test; ``_layout_is_ik_reachable`` returns
    ``verdict_fn(current_layout)``; the collision-world sync is a no-op.
    """
    import isaaclab_arena_curobo.placement_pool_ik_validation as mod

    state = {"current": None}
    monkeypatch.setattr(
        mod, "write_layout_to_sim", lambda env, env_id, layout, anchors, rots: state.__setitem__("current", layout)
    )
    monkeypatch.setattr(mod, "sync_object_poses_in_robot_base_frame", lambda planner: None)
    monkeypatch.setattr(mod, "_layout_is_ik_reachable", lambda *args, **kwargs: verdict_fn(state["current"]))
    return state


def test_filter_all_reachable_retains_every_layout(monkeypatch):
    """When every candidate is reachable, no layout is dropped and all are stamped reachable."""
    from isaaclab_arena.relations.placement_validation import PlacementCheck
    from isaaclab_arena_curobo.placement_pool_ik_validation import filter_pool_by_ik_reachability

    pool = _make_pool(num_envs=2, min_layouts_per_env=3)
    before = [len(layouts) for layouts in pool.layouts_per_env()]

    _patch_sim(monkeypatch, verdict_fn=lambda layout: True)
    filter_pool_by_ik_reachability(
        MagicMock(), MagicMock(), placement_pool=pool, target_reachable_per_env=2, max_refill_rounds=3
    )

    after = pool.layouts_per_env()
    assert [len(layouts) for layouts in after] == before
    assert all(
        layout.validation_results.validation_results[PlacementCheck.IK_REACHABLE]
        for layouts in after
        for layout in layouts
    )


def test_filter_rejects_unreachable_and_refills(monkeypatch):
    """Unreachable candidates are dropped and geometry is refilled until each env meets the target."""
    from isaaclab_arena.relations.placement_validation import PlacementCheck
    from isaaclab_arena_curobo.placement_pool_ik_validation import filter_pool_by_ik_reachability

    num_envs, target = 2, 2
    pool = _make_pool(num_envs=num_envs, min_layouts_per_env=target)

    # Reject every layout from the initial fill (num_envs * target of them), accept everything solved after,
    # forcing at least one refill-and-recheck round before the target can be met.
    reject_first = num_envs * target
    seen: dict[int, bool] = {}

    def verdict(layout) -> bool:
        key = id(layout)
        if key not in seen:
            seen[key] = len(seen) >= reject_first
        return seen[key]

    _patch_sim(monkeypatch, verdict_fn=verdict)
    filter_pool_by_ik_reachability(
        MagicMock(), MagicMock(), placement_pool=pool, target_reachable_per_env=target, max_refill_rounds=5
    )

    after = pool.layouts_per_env()
    assert all(len(layouts) >= target for layouts in after)
    assert all(
        layout.validation_results.validation_results[PlacementCheck.IK_REACHABLE]
        for layouts in after
        for layout in layouts
    )
    # More than the initially-solved candidates were validated, i.e. a refill actually happened.
    assert len(seen) > reject_first
