# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Logic tests for ReachabilityValidator, with cuRobo mocked out.

Exercises what the validator does around cuRobo -- reconstruct object poses from a layout, build one
collision cuboid per object, and IK-check one grasp per movable (non-anchor) object -- against a real
geometry-solved layout, asserting the per-layout ``validate_batch`` verdict. The cuRobo solver build
and the batched IK solve are patched, so no GPU or cuRobo install is needed; the pure-math grasp
reconstruction runs for real on CPU.
"""

from __future__ import annotations

import torch
from unittest.mock import MagicMock

import pytest


def _make_desk_box_pool(num_envs: int = 1, min_layouts_per_env: int = 2):
    """Build a small valid desk (anchor) + box (On desk) pool and return it."""
    from isaaclab_arena.assets.dummy_object import DummyObject
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import IsAnchor, On, RequiresReachability
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
    box.add_relation(RequiresReachability())

    params = ObjectPlacerParams(
        solver_params=RelationSolverParams(max_iters=200, convergence_threshold=1e-3),
        apply_positions_to_objects=False,
        min_unique_layouts_per_env=min_layouts_per_env,
        placement_seed=5,
    )
    return PooledObjectPlacer(
        objects=[desk, box],
        placer_params=params,
        pool_size=num_envs * min_layouts_per_env,
        num_envs=num_envs,
    )


def _patch_curobo(monkeypatch, feasible_fn):
    """Replace the cuRobo solver build and the batched IK solve; return the captured fake solver.

    ``feasible_fn(num_grasps) -> list[bool]`` decides per-grasp feasibility. The fake solver records the
    cuboids passed to ``update_world`` so a test can assert one obstacle per object.
    """
    import isaaclab_arena_curobo.ik_reachability_validator as mod

    class _FakeSolver:
        def __init__(self, *args, **kwargs):
            self.device = torch.device("cpu")
            self.world_cuboids = None

        def update_world(self, cuboids, base_pos, base_quat):
            self.world_cuboids = cuboids

    captured = {}

    def _make_solver(*args, **kwargs):
        captured["solver"] = _FakeSolver(*args, **kwargs)
        return captured["solver"]

    def _fake_ik(solver, target_poses, **kwargs):
        num = target_poses.shape[0]
        feasible = torch.tensor(feasible_fn(num), dtype=torch.bool)
        captured["num_grasps"] = num
        return feasible, torch.zeros(num), torch.zeros(num)

    monkeypatch.setattr(mod, "CuroboIKSolver", _make_solver)
    monkeypatch.setattr(mod, "solve_ik_feasibility", _fake_ik)
    monkeypatch.setattr(mod, "get_embodiment_curobo_cfg", lambda embodiment: None)
    return captured


def _fake_embodiment():
    """Embodiment stub reporting the env-local default base pose (origin, upright identity)."""
    from isaaclab_arena.utils.pose import Pose

    embodiment = MagicMock()
    embodiment.get_initial_pose.return_value = Pose.identity()
    return embodiment


def _make_two_box_pool(num_envs: int = 1, min_layouts_per_env: int = 2):
    """Build a desk (anchor) + two boxes (each On desk) pool; two movable objects to scope between."""
    from isaaclab_arena.assets.dummy_object import DummyObject
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import IsAnchor, On, RequiresReachability
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    desk = DummyObject(
        name="desk",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())
    boxes = []
    for box_name in ("box_a", "box_b"):
        box = DummyObject(
            name=box_name,
            bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
        )
        box.add_relation(On(desk, clearance_m=0.01))
        # Only box_a carries the RequiresReachability marker, so only it should be IK-checked.
        if box_name == "box_a":
            box.add_relation(RequiresReachability())
        boxes.append(box)

    params = ObjectPlacerParams(
        solver_params=RelationSolverParams(max_iters=200, convergence_threshold=1e-3),
        apply_positions_to_objects=False,
        min_unique_layouts_per_env=min_layouts_per_env,
        placement_seed=5,
    )
    return PooledObjectPlacer(
        objects=[desk, *boxes],
        placer_params=params,
        pool_size=num_envs * min_layouts_per_env,
        num_envs=num_envs,
    )


def _make_reachability_validator(embodiment):
    """Construct the registered ReachabilityValidator with ``embodiment`` set on its params."""
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena_curobo.ik_reachability_validator import ReachabilityValidator

    params = ObjectPlacerParams()
    params.reachability_config.embodiment = embodiment
    return ReachabilityValidator(params)


@pytest.mark.curobo_deps
def test_validator_accepts_when_all_grasps_feasible(monkeypatch):
    """A layout passes when every movable-object grasp is feasible."""
    captured = _patch_curobo(monkeypatch, feasible_fn=lambda n: [True] * n)
    validator = _make_reachability_validator(_fake_embodiment())

    layout = _make_desk_box_pool().layouts_per_env()[0][0]
    assert validator.validate_batch([layout.positions], [layout.orientations], [{}], []) == [True]
    # One collision cuboid per object (desk + box); one grasp per movable object (box only, desk is anchor).
    assert len(captured["solver"].world_cuboids) == 2
    assert captured["num_grasps"] == 1


@pytest.mark.curobo_deps
def test_validator_rejects_when_any_grasp_infeasible(monkeypatch):
    """A layout fails when any movable-object grasp is infeasible."""
    _patch_curobo(monkeypatch, feasible_fn=lambda n: [False] * n)
    validator = _make_reachability_validator(_fake_embodiment())

    layout = _make_desk_box_pool().layouts_per_env()[0][0]
    assert validator.validate_batch([layout.positions], [layout.orientations], [{}], []) == [False]


@pytest.mark.curobo_deps
def test_validator_checks_only_stamped_objects(monkeypatch):
    """Only movable objects stamped with a 'reachable' constraint are IK-checked, not every movable object."""
    captured = _patch_curobo(monkeypatch, feasible_fn=lambda n: [True] * n)
    validator = _make_reachability_validator(_fake_embodiment())

    layout = _make_two_box_pool().layouts_per_env()[0][0]
    assert validator.validate_batch([layout.positions], [layout.orientations], [{}], []) == [True]
    # Two movable boxes exist, but only the stamped one (box_a) contributes a grasp.
    assert captured["num_grasps"] == 1
