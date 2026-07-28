# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Logic tests for ReachabilityValidator, with cuRobo mocked out.

Exercises what the validator does around cuRobo -- reconstruct object poses from a layout, build one
collision cuboid per object, and IK-check one grasp per task-marked reachability target -- against a
real geometry-solved layout, asserting the per-layout ``validate_batch`` verdict. The cuRobo solver
build and batched IK solve are patched, so no GPU or cuRobo install is needed; pure-math grasp
reconstruction runs for real on CPU.
"""

from __future__ import annotations

import torch
from unittest.mock import MagicMock

import pytest


def _make_desk_box_pool(
    num_envs: int = 1,
    min_layouts_per_env: int = 2,
    desk_rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
):
    """Build a small valid desk (anchor) + box (On desk) pool and return it."""
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import IsAnchor, On, RequiresReachability
    from isaaclab_arena.tests.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import OrientedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    desk = DummyObject(
        name="desk",
        bounding_box=OrientedBoundingBox.from_min_max(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=desk_rotation))
    desk.add_relation(IsAnchor())
    box = DummyObject(
        name="box",
        bounding_box=OrientedBoundingBox.from_min_max(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
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

    def _fake_cuboid(obj, bbox, position, rotation):
        captured.setdefault("cuboid_bboxes", {})[obj.name] = bbox
        captured.setdefault("cuboid_poses", {})[obj.name] = (position, rotation)
        return obj.name

    def _fake_grasp(position, rotation, *args, **kwargs):
        captured.setdefault("grasp_poses", []).append((position, rotation))
        return torch.tensor((*position, *rotation), dtype=torch.float32)

    monkeypatch.setattr(mod, "CuroboIKSolver", _make_solver)
    monkeypatch.setattr(mod, "solve_ik_feasibility", _fake_ik)
    monkeypatch.setattr(mod, "get_obb_collision_cuboid_for_object", _fake_cuboid)
    monkeypatch.setattr(mod, "top_down_grasp_pose_from_world_poses", _fake_grasp)
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
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import IsAnchor, On, RequiresReachability
    from isaaclab_arena.tests.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import OrientedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    desk = DummyObject(
        name="desk",
        bounding_box=OrientedBoundingBox.from_min_max(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())
    boxes = []
    for box_name in ("box_a", "box_b"):
        box = DummyObject(
            name=box_name,
            bounding_box=OrientedBoundingBox.from_min_max(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
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


def _make_unstamped_desk_box_pool(num_envs: int = 1, min_layouts_per_env: int = 2):
    """Build a desk (anchor) + box (On desk) pool where the box carries NO RequiresReachability marker."""
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.tests.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import OrientedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    desk = DummyObject(
        name="desk",
        bounding_box=OrientedBoundingBox.from_min_max(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())
    box = DummyObject(
        name="box",
        bounding_box=OrientedBoundingBox.from_min_max(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )
    box.add_relation(On(desk, clearance_m=0.01))

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


def _make_reachability_validator(embodiment):
    """Construct the registered ReachabilityValidator with ``embodiment`` set on its params."""
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena_curobo.ik_reachability_validator import ReachabilityValidator

    params = ObjectPlacerParams()
    params.reachability_config.embodiment = embodiment
    return ReachabilityValidator(params)


def _layout_bboxes(layout):
    """Return the complete candidate bounding-box map for a solved layout."""
    return {obj: obj.get_bounding_box() for obj in layout.positions}


@pytest.mark.curobo_deps
def test_collision_cuboid_composes_intrinsic_bbox_rotation():
    """The cuRobo cuboid pose includes the OBB's asset-local orientation."""
    from isaaclab_arena.tests.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import OrientedBoundingBox
    from isaaclab_arena_curobo.utils.ik_solver_utils import get_obb_collision_cuboid_for_object

    bbox_rotation = (0.0, 0.0, 2**-0.5, 2**-0.5)
    obj = DummyObject(
        name="rotated_box",
        bounding_box=OrientedBoundingBox(
            center=(0.25, 0.0, 0.0),
            half_extents=(0.2, 0.1, 0.05),
            rotation_xyzw=bbox_rotation,
        ),
    )

    cuboid = get_obb_collision_cuboid_for_object(
        obj,
        obj.get_bounding_box(),
        pos_w=(1.0, 2.0, 3.0),
        quat_w_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    assert cuboid.pose_W_O.position_xyz == pytest.approx((1.25, 2.0, 3.0))
    assert cuboid.pose_W_O.rotation_xyzw == pytest.approx(bbox_rotation)


@pytest.mark.curobo_deps
def test_validator_accepts_when_all_grasps_feasible(monkeypatch):
    """Full world quaternions reach cuboid and grasp construction unchanged."""
    captured = _patch_curobo(monkeypatch, feasible_fn=lambda n: [True] * n)
    validator = _make_reachability_validator(_fake_embodiment())

    anchor_rotation = (0.5, 0.0, 0.0, 3**0.5 / 2)
    movable_rotation = (0.5, 0.5, 0.5, 0.5)
    layout = _make_desk_box_pool(desk_rotation=anchor_rotation).layouts_per_env()[0][0]
    box = next(obj for obj in layout.positions if obj.name == "box")
    layout.rotations[box] = movable_rotation
    assert validator.validate_batch([layout.positions], [layout.rotations], [_layout_bboxes(layout)], []) == [True]
    # One collision cuboid per object (desk + box); one grasp per movable object (box only, desk is anchor).
    assert len(captured["solver"].world_cuboids) == 2
    assert captured["num_grasps"] == 1
    assert captured["cuboid_poses"]["desk"][1] == pytest.approx(anchor_rotation)
    box_cuboid_position, box_cuboid_rotation = captured["cuboid_poses"]["box"]
    assert box_cuboid_position == pytest.approx(layout.positions[box])
    assert box_cuboid_rotation == pytest.approx(movable_rotation)
    assert len(captured["grasp_poses"]) == 1
    grasp_position, grasp_rotation = captured["grasp_poses"][0]
    assert grasp_position == pytest.approx(layout.positions[box])
    assert grasp_rotation == pytest.approx(movable_rotation)


@pytest.mark.curobo_deps
def test_validator_rejects_when_any_grasp_infeasible(monkeypatch):
    """Omitted movable rotations use identity before an infeasible grasp rejects the layout."""
    captured = _patch_curobo(monkeypatch, feasible_fn=lambda n: [False] * n)
    validator = _make_reachability_validator(_fake_embodiment())

    layout = _make_desk_box_pool().layouts_per_env()[0][0]
    layout.rotations.clear()
    assert validator.validate_batch([layout.positions], [layout.rotations], [_layout_bboxes(layout)], []) == [False]
    assert captured["cuboid_poses"]["box"][1] == pytest.approx((0.0, 0.0, 0.0, 1.0))


@pytest.mark.curobo_deps
def test_validator_checks_only_stamped_objects(monkeypatch):
    """Only movable objects stamped with a 'reachable' constraint are IK-checked, not every movable object."""
    captured = _patch_curobo(monkeypatch, feasible_fn=lambda n: [True] * n)
    validator = _make_reachability_validator(_fake_embodiment())

    layout = _make_two_box_pool().layouts_per_env()[0][0]
    assert validator.validate_batch([layout.positions], [layout.rotations], [_layout_bboxes(layout)], []) == [True]
    # Two movable boxes exist, but only the stamped one (box_a) contributes a grasp.
    assert captured["num_grasps"] == 1


@pytest.mark.curobo_deps
def test_validator_passes_trivially_and_warns_when_no_targets(monkeypatch, capsys):
    """No stamped target: the layout passes trivially, no IK solve runs, and a one-time warning is printed."""
    captured = _patch_curobo(monkeypatch, feasible_fn=lambda n: [True] * n)
    validator = _make_reachability_validator(_fake_embodiment())

    layout = _make_unstamped_desk_box_pool().layouts_per_env()[0][0]
    # Two layouts through the same validator: the warning must print once, not once per candidate.
    bboxes = _layout_bboxes(layout)
    assert validator.validate_batch(
        [layout.positions, layout.positions],
        [layout.rotations, layout.rotations],
        [bboxes, bboxes],
        [],
    ) == [True, True]

    # No grasp was ever solved (the IK path is skipped entirely when there are no targets).
    assert "num_grasps" not in captured
    assert capsys.readouterr().out.count("resolved zero reachability targets") == 1


@pytest.mark.curobo_deps
def test_validator_passes_candidate_bbox_to_cuboid_construction(monkeypatch):
    """The candidate's heterogeneous box, not the object's default box, defines its cuboid."""
    from isaaclab_arena.utils.bounding_box import OrientedBoundingBox

    captured = _patch_curobo(monkeypatch, feasible_fn=lambda n: [True] * n)
    validator = _make_reachability_validator(_fake_embodiment())
    layout = _make_desk_box_pool().layouts_per_env()[0][0]
    box = next(obj for obj in layout.positions if obj.name == "box")
    candidate_bbox = OrientedBoundingBox(
        center=(0.4, -0.2, 0.1),
        half_extents=(0.3, 0.2, 0.1),
        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    bboxes = _layout_bboxes(layout)
    bboxes[box] = candidate_bbox

    assert validator.validate_batch([layout.positions], [layout.rotations], [bboxes], []) == [True]
    assert captured["cuboid_bboxes"]["box"] is candidate_bbox
