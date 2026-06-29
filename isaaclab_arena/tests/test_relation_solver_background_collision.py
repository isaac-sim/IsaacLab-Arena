# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for passive background-mesh collision in the relation solver.

Background obstacles carry no relations (so they are absent from the relation graph)
but should still be avoided by placed objects.
"""

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def _make_desk():
    """Anchor desk, 2m x 1m wide so a box has room to relocate along X away from the obstacle."""
    from isaaclab_arena.assets.dummy_object import DummyObject
    from isaaclab_arena.relations.relations import IsAnchor
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    desk = DummyObject(
        name="desk",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(2.0, 1.0, 0.1)),
    )
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())
    return desk


def _make_box(name: str = "box"):
    """A 0.3m cube to place On the desk (smaller than each desk half so a valid spot exists)."""
    from isaaclab_arena.assets.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    return DummyObject(
        name=name,
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.3, 0.3, 0.3)),
    )


def _make_background():
    """Tall obstacle on the desk's left strip (x in [0, 0.5]).

    Narrower than the desk so a straddling box has a non-zero escape gradient: the
    overlap-volume loss is flat when one box is fully enclosed by the other.
    """
    from isaaclab_arena.assets.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    background = DummyObject(
        name="cabinet",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.5, 1.0, 1.0)),
    )
    background.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    return background


def _mesh_box(name: str, extents: tuple[float, float, float], position: tuple[float, float, float]):
    """Dummy object with a box collision mesh and fixed pose."""
    import trimesh

    from isaaclab_arena.assets.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    half = tuple(e / 2.0 for e in extents)
    obj = DummyObject(
        name=name,
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-half[0], -half[1], -half[2]),
            max_point=(half[0], half[1], half[2]),
        ),
        collision_mesh=trimesh.creation.box(extents=extents),
    )
    obj.set_initial_pose(Pose(position_xyz=position, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    return obj


def test_background_collision_object_combines_meshes():
    """Multiple fixed background meshes are represented as one collision-only object."""
    from isaaclab_arena.relations.background_collision_object import make_background_collision_object

    left = _mesh_box("left_cabinet", (0.2, 0.2, 0.2), (-1.0, 0.0, 0.0))
    right = _mesh_box("right_cabinet", (0.2, 0.2, 0.2), (1.0, 0.0, 0.0))

    background = make_background_collision_object([left, right])

    assert background is not None
    assert background.name == "__background_collision_mesh__"
    assert len(background.get_collision_mesh().split()) == 2
    bounds = background.get_collision_mesh().bounds
    assert bounds[0][0] < -1.09
    assert bounds[1][0] > 1.09


def test_collision_objects_add_no_overlap_loss():
    """Box overlapping the obstacle incurs positive loss; zero loss when the obstacle is absent."""
    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relation_solver_state import RelationSolverState
    from isaaclab_arena.relations.relations import On

    desk = _make_desk()
    box = _make_box()
    box.add_relation(On(desk))
    background = _make_background()

    # Box on the desk top, inside the background footprint.
    initial_positions = [{desk: (0.0, 0.0, 0.0), box: (0.3, 0.3, 0.1)}]
    solver = RelationSolver(RelationSolverParams(verbose=False))

    state_with = RelationSolverState([desk, box], initial_positions, collision_objects=[background])
    state_without = RelationSolverState([desk, box], initial_positions)

    loss_with = solver._compute_no_overlap_loss(state_with).sum().item()
    loss_without = solver._compute_no_overlap_loss(state_without).sum().item()

    # Box-desk is an On pair (skipped), so the obstacle is the only overlap source.
    assert loss_without == 0.0
    assert loss_with > 0.0


def test_solver_pushes_object_off_background_obstacle():
    """Full solve moves the box onto the clear (right) half of the desk."""
    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import On

    desk = _make_desk()
    box = _make_box()
    box.add_relation(On(desk))
    background = _make_background()

    initial_positions = [{desk: (0.0, 0.0, 0.0), box: (0.3, 0.3, 0.1)}]
    solver = RelationSolver(RelationSolverParams(verbose=False, save_position_history=False))

    result = solver.solve([desk, box], initial_positions, collision_objects=[background])[0]

    box_world = box.get_bounding_box().translated(result[box])
    assert not box_world.overlaps(background.get_world_bounding_box(), margin=0.0).item()


def test_solve_without_collision_objects_is_a_noop():
    """Omitting collision objects must not raise and still solves the On placement."""
    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import On

    desk = _make_desk()
    box = _make_box()
    box.add_relation(On(desk))

    initial_positions = [{desk: (0.0, 0.0, 0.0), box: (0.3, 0.3, 0.1)}]
    solver = RelationSolver(RelationSolverParams(verbose=False, save_position_history=False))

    result = solver.solve([desk, box], initial_positions)[0]
    box_bottom_z = box.get_bounding_box().min_point[0, 2].item() + result[box][2]
    assert abs(box_bottom_z - 0.1) < 0.05


def test_validate_no_overlap_rejects_background_overlap():
    """ObjectPlacer validation flags a placed object overlapping a fixed background obstacle."""
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.relations import On

    desk = _make_desk()
    box = _make_box()
    box.add_relation(On(desk))
    background = _make_background()

    placer = ObjectPlacer()
    env_bboxes = {desk: desk.get_bounding_box(), box: box.get_bounding_box()}

    overlapping = {desk: (0.0, 0.0, 0.0), box: (0.3, 0.3, 0.1)}
    assert not placer._validate_no_overlap(overlapping, env_bboxes, [background])

    clear = {desk: (0.0, 0.0, 0.0), box: (1.5, 0.3, 0.1)}
    assert placer._validate_no_overlap(clear, env_bboxes, [background])


def _test_scene_get_collision_objects_filters(simulation_app) -> bool:
    """Only relation-free objects with a USD path and a fixed Pose are returned."""
    from unittest.mock import MagicMock

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.relations.relations import IsAnchor
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose, PoseRange

    def fake_object(name, relations, usd_path, pose):
        obj = MagicMock(spec=Object)
        obj.name = name
        obj.get_relations.return_value = relations
        obj.usd_path = usd_path
        obj.get_initial_pose.return_value = pose
        return obj

    fixed_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    pose_range = PoseRange(
        position_xyz_min=(0.0, 0.0, 0.0),
        position_xyz_max=(1.0, 1.0, 1.0),
        rpy_min=(0.0, 0.0, 0.0),
        rpy_max=(0.0, 0.0, 0.0),
    )
    background = fake_object("background", [], "bg.usd", fixed_pose)
    anchored = fake_object("anchored", [IsAnchor()], "a.usd", fixed_pose)
    no_usd = fake_object("no_usd", [], None, fixed_pose)
    no_pose = fake_object("no_pose", [], "n.usd", None)
    ranged = fake_object("ranged", [], "r.usd", pose_range)

    scene = Scene()
    scene.assets = {o.name: o for o in [background, anchored, no_usd, no_pose, ranged]}

    return scene.get_collision_objects() == [background]


def test_scene_get_collision_objects_filters():
    result = run_simulation_app_function(_test_scene_get_collision_objects_filters, headless=HEADLESS)
    assert result, "Scene.get_collision_objects() returned the wrong subset"


def test_object_placer_place_forwards_collision_objects():
    """ObjectPlacer.place threads collision_objects into the solve and validation, yielding a clear layout."""
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import On

    desk = _make_desk()
    box = _make_box()
    box.add_relation(On(desk))
    background = _make_background()

    params = ObjectPlacerParams(
        placement_seed=0,
        solver_params=RelationSolverParams(verbose=False, save_position_history=False),
    )
    result = ObjectPlacer(params=params).place([desk, box], num_envs=1, collision_objects=[background])[0]

    assert result.success
    box_world = box.get_bounding_box().translated(result.positions[box])
    assert not box_world.overlaps(background.get_world_bounding_box(), margin=0.0).item()


def test_pooled_object_placer_forwards_collision_objects():
    """PooledObjectPlacer (the production reset path) avoids background obstacles in pooled layouts."""
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import On

    desk = _make_desk()
    box = _make_box()
    box.add_relation(On(desk))
    background = _make_background()

    params = ObjectPlacerParams(
        placement_seed=0,
        apply_positions_to_objects=False,
        solver_params=RelationSolverParams(verbose=False, save_position_history=False),
    )
    pool = PooledObjectPlacer(
        objects=[desk, box],
        placer_params=params,
        pool_size=4,
        num_envs=1,
        collision_objects=[background],
    )
    layout = pool.sample_with_replacement(1)[0]

    box_world = box.get_bounding_box().translated(layout.positions[box])
    assert not box_world.overlaps(background.get_world_bounding_box(), margin=0.0).item()
