# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Logic tests for ReachabilityValidator, with cuRobo mocked out.

Exercises what the validator does around cuRobo -- reconstruct object poses from a layout, build a
collision cuboid per non-target object, and IK-check one grasp per movable (non-anchor) object -- against
a real geometry-solved layout, asserting the per-layout ``validate_batch`` verdict. The cuRobo solver build
and the batched IK solve are patched, so no GPU or cuRobo install is needed; the pure-math grasp
reconstruction runs for real on CPU.
"""

from __future__ import annotations

import torch
from unittest.mock import MagicMock

import pytest

DOF = 7
"""Joint count the patched IK solve reports; only its width matters, no kinematics run here."""

NUM_SPHERES = 5
"""Collision-sphere count the patched kinematics reports, for the same reason."""


def _make_desk_box_pool(num_envs: int = 1, min_layouts_per_env: int = 2):
    """Build a small valid desk (anchor) + box (On desk) pool and return it."""
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import IsAnchor, On, RequiresReachability
    from isaaclab_arena.tests.dummy_object import DummyObject
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
        from isaaclab_arena_curobo.ik_solver import IKFeasibility

        num = target_poses.shape[0]
        feasible = torch.tensor(feasible_fn(num), dtype=torch.bool)
        captured["num_grasps"] = num
        captured["ik_kwargs"] = kwargs
        return IKFeasibility(feasible, torch.zeros(num), torch.zeros(num), torch.zeros(num, DOF))

    monkeypatch.setattr(mod, "CuroboIKSolver", _make_solver)
    monkeypatch.setattr(mod, "solve_ik_feasibility", _fake_ik)
    monkeypatch.setattr(mod, "get_embodiment_curobo_cfg", lambda embodiment: None)
    # The sphere helpers are cuRobo kinematics calls; they are covered in test_ik_solver_utils.py.
    monkeypatch.setattr(mod, "robot_collision_spheres", lambda solver, q: torch.zeros(q.shape[0], NUM_SPHERES, 4))
    monkeypatch.setattr(mod, "hand_sphere_mask", lambda solver: torch.zeros(NUM_SPHERES, dtype=torch.bool))
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


def _make_unstamped_desk_box_pool(num_envs: int = 1, min_layouts_per_env: int = 2):
    """Build a desk (anchor) + box (On desk) pool where the box carries NO RequiresReachability marker."""
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.tests.dummy_object import DummyObject
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
        placement_seed=5,
    )
    return PooledObjectPlacer(
        objects=[desk, box],
        placer_params=params,
        pool_size=num_envs * min_layouts_per_env,
        num_envs=num_envs,
    )


def _make_reachability_validator(embodiment, visualizer=None, require_collision_free=None):
    """Construct the registered ReachabilityValidator with ``embodiment`` set on its params.

    ``require_collision_free`` overrides that config default when given.
    """
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena_curobo.ik_reachability_validator import ReachabilityValidator

    params = ObjectPlacerParams()
    params.reachability_config.embodiment = embodiment
    if require_collision_free is not None:
        params.reachability_config.require_collision_free = require_collision_free
    return ReachabilityValidator(params, visualizer)


@pytest.mark.curobo_deps
def test_validator_skips_visualization_by_default(monkeypatch):
    """The debug view is opt-in: placement without one means the check adds no layer."""
    _patch_curobo(monkeypatch, feasible_fn=lambda n: [True] * n)
    validator = _make_reachability_validator(_fake_embodiment())

    assert validator._rerun_layer is None


@pytest.mark.curobo_deps
def test_validator_draws_each_candidate_on_its_own_frame(monkeypatch):
    """The check draws on the frame of the candidate it was handed, not on its position in the batch.

    Expensive checks only see the candidates that passed the cheap ones, so the two differ.
    """
    _patch_curobo(monkeypatch, feasible_fn=lambda n: [False] * n)
    visualizer = MagicMock()
    # The batch the check is given is candidates 7 and 9 of the run.
    visualizer.get_layout_index_across_batch.side_effect = [7, 9]
    validator = _make_reachability_validator(_fake_embodiment(), visualizer=visualizer)
    drawn: list[dict] = []
    monkeypatch.setattr(validator._rerun_layer, "log_layout", lambda **kwargs: drawn.append(kwargs))

    layout = _make_desk_box_pool().layouts_per_env()[0][0]
    assert validator.validate_batch(
        [layout.positions, layout.positions], [layout.orientations, layout.orientations], [{}, {}], []
    ) == [False, False]

    assert [entry["layout_index_across_batch"] for entry in drawn] == [7, 9]
    assert [entry["target_names"] for entry in drawn] == [["box"], ["box"]]
    # The arm's pose at each solved grasp is drawn with it, one set of collision spheres per grasp,
    # with the hand spheres marked as the ones the solve muted.
    assert [tuple(entry["robot_spheres"].shape) for entry in drawn] == [(1, NUM_SPHERES, 4), (1, NUM_SPHERES, 4)]
    assert all(entry["muted_sphere_mask"] is not None for entry in drawn)


def test_reachability_layer_records_to_rrd(tmp_path):
    """The layer's grasps and verdicts reach a recording without a viewer or cuRobo."""
    from isaaclab_arena.relations.placement_visualizer import PlacementRerunVisualizer
    from isaaclab_arena_curobo.reachability_visualizer import ReachabilityRerunLayer

    rrd_path = tmp_path / "placement.rrd"
    visualizer = PlacementRerunVisualizer(app_id="arena_test", spawn=False, output_path=str(rrd_path))
    layer = ReachabilityRerunLayer(visualizer)

    layer.log_layout(
        layout_index_across_batch=0,
        robot_base_pos_w=(0.0, 0.0, 0.0),
        robot_base_quat_w_xyzw=(0.0, 0.0, 0.0, 1.0),
        target_names=["box"],
        grasp_poses_base_frame=torch.eye(4).unsqueeze(0),
        feasible=torch.tensor([False]),
        position_error=torch.tensor([0.3]),
        rotation_error=torch.tensor([0.1]),
        robot_spheres=torch.tensor([[[0.1, 0.0, 0.2, 0.05], [0.2, 0.0, 0.3, 0.04]]]),
        muted_sphere_mask=torch.tensor([False, True]),
    )
    visualizer.close()

    assert rrd_path.is_file() and rrd_path.stat().st_size > 0


@pytest.mark.curobo_deps
def test_validator_accepts_when_all_grasps_feasible(monkeypatch):
    """A layout passes when every movable-object grasp is feasible."""
    captured = _patch_curobo(monkeypatch, feasible_fn=lambda n: [True] * n)
    validator = _make_reachability_validator(_fake_embodiment())

    layout = _make_desk_box_pool().layouts_per_env()[0][0]
    assert validator.validate_batch([layout.positions], [layout.orientations], [{}], []) == [True]
    # One collision cuboid per non-target object (the desk); one grasp per target (the box).
    assert [cuboid.name for cuboid in captured["solver"].world_cuboids] == ["desk"]
    assert captured["num_grasps"] == 1


@pytest.mark.curobo_deps
def test_validator_collision_checks_grasps_by_default(monkeypatch):
    """Grasps are collision-checked unless the task's reachability config opts out."""
    captured = _patch_curobo(monkeypatch, feasible_fn=lambda n: [True] * n)
    layout = _make_desk_box_pool().layouts_per_env()[0][0]

    _make_reachability_validator(_fake_embodiment()).validate_batch([layout.positions], [layout.orientations], [{}], [])
    assert captured["ik_kwargs"]["require_collision_free"] is True

    _make_reachability_validator(_fake_embodiment(), require_collision_free=False).validate_batch(
        [layout.positions], [layout.orientations], [{}], []
    )
    assert captured["ik_kwargs"]["require_collision_free"] is False


@pytest.mark.curobo_deps
def test_validator_rejects_when_any_grasp_infeasible(monkeypatch):
    """A layout fails when any movable-object grasp is infeasible."""
    _patch_curobo(monkeypatch, feasible_fn=lambda n: [False] * n)
    validator = _make_reachability_validator(_fake_embodiment())

    layout = _make_desk_box_pool().layouts_per_env()[0][0]
    assert validator.validate_batch([layout.positions], [layout.orientations], [{}], []) == [False]


@pytest.mark.curobo_deps
def test_grasp_targets_do_not_obstruct_their_own_grasp(monkeypatch):
    """Targets are what the gripper closes on, so only the other objects go into the collision world."""
    captured = _patch_curobo(monkeypatch, feasible_fn=lambda n: [True] * n)
    validator = _make_reachability_validator(_fake_embodiment())

    # Of the three objects, only box_a is a target; the desk and the unstamped box_b stay obstacles.
    layout = _make_two_box_pool().layouts_per_env()[0][0]
    validator.validate_batch([layout.positions], [layout.orientations], [{}], [])

    assert sorted(cuboid.name for cuboid in captured["solver"].world_cuboids) == ["box_b", "desk"]


@pytest.mark.curobo_deps
def test_validator_checks_only_stamped_objects(monkeypatch):
    """Only movable objects stamped with a 'reachable' constraint are IK-checked, not every movable object."""
    captured = _patch_curobo(monkeypatch, feasible_fn=lambda n: [True] * n)
    validator = _make_reachability_validator(_fake_embodiment())

    layout = _make_two_box_pool().layouts_per_env()[0][0]
    assert validator.validate_batch([layout.positions], [layout.orientations], [{}], []) == [True]
    # Two movable boxes exist, but only the stamped one (box_a) contributes a grasp.
    assert captured["num_grasps"] == 1


@pytest.mark.curobo_deps
def test_validator_passes_trivially_and_warns_when_no_targets(monkeypatch, capsys):
    """No stamped target: the layout passes trivially, no IK solve runs, and a one-time warning is printed."""
    captured = _patch_curobo(monkeypatch, feasible_fn=lambda n: [True] * n)
    validator = _make_reachability_validator(_fake_embodiment())

    layout = _make_unstamped_desk_box_pool().layouts_per_env()[0][0]
    # Two layouts through the same validator: the warning must print once, not once per candidate.
    assert validator.validate_batch(
        [layout.positions, layout.positions], [layout.orientations, layout.orientations], [{}, {}], []
    ) == [True, True]

    # No grasp was ever solved (the IK path is skipped entirely when there are no targets).
    assert "num_grasps" not in captured
    assert capsys.readouterr().out.count("resolved zero reachability targets") == 1
