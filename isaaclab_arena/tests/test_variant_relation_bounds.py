# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check anchored variant geometry in relation solving and its diagnostics."""

import math
import torch

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _make_variant_object(name, sizes, **kwargs):
    from isaaclab.sim import CuboidCfg

    from isaaclab_arena.assets.object_choice import ObjectChoice

    return ObjectChoice(name=name, objects=[CuboidCfg(size=size) for size in sizes], **kwargs)


def _make_rotated_anchor():
    from isaaclab_arena.relations.relations import IsAnchor
    from isaaclab_arena.utils.pose import Pose

    return _make_variant_object(
        "support",
        [(2.0, 4.0, 1.0), (6.0, 8.0, 3.0)],
        initial_pose=Pose(
            position_xyz=(10.0, 20.0, 1.0),
            rotation_xyzw=(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)),
        ),
        relations=[IsAnchor()],
    )


def _test_heterogeneous_anchor_uses_assigned_bounds_and_fixed_pose(simulation_app):
    from isaaclab_arena.relations.bounding_box_helpers import build_per_env_bounding_boxes
    from isaaclab_arena.relations.relation_solver_state import RelationSolverState
    from isaaclab_arena.scene.object_variant_assignment import assign_object_variants
    from isaaclab_arena.tests.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    anchor = _make_rotated_anchor()
    assign_object_variants([anchor], num_envs=2)
    obstacle = DummyObject(
        name="passive_obstacle",
        bounding_box=AxisAlignedBoundingBox((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        initial_pose=Pose(position_xyz=(-5.0, 0.0, 2.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
    )
    bounds = build_per_env_bounding_boxes([anchor], 2).object_bboxes
    state = RelationSolverState(
        [anchor],
        [{anchor: (10.0, 20.0, 1.0)}, {anchor: (10.0, 20.0, 1.0)}],
        env_bboxes=bounds,
        collision_objects=[obstacle],
    )

    anchor_bounds = state.get_fixed_obstacle_world_bbox(anchor)
    assert torch.allclose(anchor_bounds.min_point, torch.tensor([[8.0, 19.0, 0.5], [6.0, 17.0, -0.5]]))
    assert torch.allclose(anchor_bounds.max_point, torch.tensor([[12.0, 21.0, 1.5], [14.0, 23.0, 2.5]]))
    obstacle_bounds = state.get_fixed_obstacle_world_bbox(obstacle)
    assert torch.equal(obstacle_bounds.min_point, torch.tensor([[-5.0, 0.0, 2.0]]))
    assert torch.equal(obstacle_bounds.max_point, torch.tensor([[-4.0, 1.0, 3.0]]))
    return True


def _test_debug_losses_reuses_variant_bounds_without_position_history(simulation_app, monkeypatch, capsys):
    from isaaclab_arena.relations.bounding_box_helpers import build_per_env_bounding_boxes
    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import AtPosition, On
    from isaaclab_arena.scene.object_variant_assignment import assign_object_variants

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    anchor = _make_rotated_anchor()
    child = _make_variant_object(
        "pickup",
        [(0.1, 0.2, 0.3), (0.2, 0.4, 0.6)],
        relations=[On(anchor), AtPosition(x=10.0, y=20.0)],
    )
    objects = [anchor, child]
    assign_object_variants(objects, num_envs=2)
    bounds = build_per_env_bounding_boxes(objects, 2).object_bboxes
    initial_positions = [
        {anchor: (10.0, 20.0, 1.0), child: (10.0, 20.0, 1.66)},
        {anchor: (10.0, 20.0, 1.0), child: (10.0, 20.0, 2.81)},
    ]
    solver = RelationSolver(RelationSolverParams(max_iters=0, verbose=False, save_position_history=False))
    solver.solve(objects, initial_positions, env_bboxes=bounds)
    losses_before_debug = solver.last_loss_per_env.clone()
    assert solver.last_position_history == []

    solver.debug_losses(objects)

    output = capsys.readouterr().out
    assert "pickup -> On(support)" in output
    assert "pickup -> AtPosition" in output
    assert "Parent world bbox: min=[8.0, 19.0, 0.5]" in output
    assert torch.equal(solver.last_loss_per_env, losses_before_debug)
    return True


def test_heterogeneous_anchor_uses_assigned_bounds_and_fixed_pose():
    assert run_function_with_persistent_simulation_app(_test_heterogeneous_anchor_uses_assigned_bounds_and_fixed_pose)


def test_debug_losses_reuses_variant_bounds_without_position_history(monkeypatch, capsys):
    assert run_function_with_persistent_simulation_app(
        _test_debug_losses_reuses_variant_bounds_without_position_history, monkeypatch=monkeypatch, capsys=capsys
    )
