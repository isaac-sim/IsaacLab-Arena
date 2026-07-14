# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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
    from isaaclab_arena.relations.background_collision_object import FixedCollisionObject, make_fixed_collision_objects
    from isaaclab_arena.relations.collision_mode import CollisionMode

    left = _mesh_box("left_cabinet", (0.2, 0.2, 0.2), (-1.0, 0.0, 0.0))
    right = _mesh_box("right_cabinet", (0.2, 0.2, 0.2), (1.0, 0.0, 0.0))
    meshless = _make_background()

    collision_objects = make_fixed_collision_objects([left, right])
    background = collision_objects[0]

    assert len(collision_objects) == 1
    assert background.name == "fixed_collision_mesh"
    assert not background.repair_collision_mesh_non_watertight
    assert len(background.get_collision_mesh().split()) == 2
    bounds = background.get_collision_mesh().bounds
    assert bounds[0][0] < -1.09
    assert bounds[1][0] > 1.09

    collision_objects = make_fixed_collision_objects([left, meshless])
    assert isinstance(collision_objects[0], FixedCollisionObject)
    assert collision_objects[1] is meshless

    bbox_only = _mesh_box("bbox_only", (0.2, 0.2, 0.2), (2.0, 0.0, 0.0))
    bbox_only.collision_mode = CollisionMode.BBOX
    collision_objects = make_fixed_collision_objects([left, bbox_only])
    assert isinstance(collision_objects[0], FixedCollisionObject)
    assert collision_objects[1] is bbox_only
    assert len(collision_objects[0].get_collision_mesh().split()) == 1


def test_background_collision_objects_reject_failed_whole_background(monkeypatch):
    """Whole-scene Background mesh extraction failure raises."""
    import pytest

    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.relations.background_collision_object import make_fixed_collision_objects
    from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache
    from isaaclab_arena.utils.pose import Pose

    left = _mesh_box("left_cabinet", (0.2, 0.2, 0.2), (-1.0, 0.0, 0.0))
    kitchen = Background.__new__(Background)
    kitchen.name = "kitchen"
    kitchen.collision_mode = None
    kitchen.initial_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    monkeypatch.setattr(
        WarpMeshAndSphereCache,
        "get_collision_mesh",
        lambda self, obj: left.get_collision_mesh() if obj is left else None,
    )

    with pytest.raises(AssertionError, match="whole-scene Background"):
        make_fixed_collision_objects([left, kitchen])


def test_warp_mesh_cache_caches_unsupported_usd_geometry(monkeypatch):
    """Unsupported USD geometry degrades to cached meshless collision."""
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache
    from isaaclab_arena.utils.usd_helpers import UnsupportedCollisionGeometryError

    obj = Object.__new__(Object)
    obj.name = "kitchen"
    obj.usd_path = "/tmp/kitchen.usd"
    obj.scale = (1.0, 1.0, 1.0)
    obj.object_type = ObjectType.BASE
    obj.repair_collision_mesh_non_watertight = True
    calls = {"count": 0}

    def fail_extract(usd_path, scale):
        calls["count"] += 1
        raise UnsupportedCollisionGeometryError("Unsupported non-mesh geometry in /tmp/kitchen.usd: /World/cube")

    monkeypatch.setattr("isaaclab_arena.utils.usd_helpers.extract_trimesh_from_usd", fail_extract)
    manager = WarpMeshAndSphereCache(device="cpu")

    assert manager.get_collision_mesh(obj) is None
    assert manager.get_collision_mesh(obj) is None
    assert calls["count"] == 1


def test_background_collision_objects_treat_background_none_pose_as_identity(monkeypatch):
    """A Background with no initial pose is fixed at the USD origin for mesh aggregation."""
    import torch

    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.relations.background_collision_object import make_fixed_collision_objects
    from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache

    kitchen = Background.__new__(Background)
    kitchen.name = "kitchen"
    kitchen.collision_mode = None
    kitchen.initial_pose = None
    mesh_source = _mesh_box("source", (0.2, 0.2, 0.2), (0.0, 0.0, 0.0))
    monkeypatch.setattr(
        WarpMeshAndSphereCache, "get_collision_mesh", lambda self, obj: mesh_source.get_collision_mesh()
    )

    collision_objects = make_fixed_collision_objects([kitchen])

    assert len(collision_objects) == 1
    assert torch.allclose(collision_objects[0].get_bounding_box().min_point, torch.tensor([[-0.1, -0.1, -0.1]]))
    assert torch.allclose(collision_objects[0].get_bounding_box().max_point, torch.tensor([[0.1, 0.1, 0.1]]))


def test_background_collision_objects_reject_bbox_whole_background():
    """Whole-scene Backgrounds cannot use BBOX collision because their AABBs are room-scale."""
    import pytest

    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.relations.background_collision_object import make_fixed_collision_objects
    from isaaclab_arena.relations.collision_mode import CollisionMode
    from isaaclab_arena.utils.pose import Pose

    kitchen = Background.__new__(Background)
    kitchen.name = "kitchen"
    kitchen.collision_mode = CollisionMode.BBOX
    kitchen.initial_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))

    with pytest.raises(AssertionError, match="Whole-scene Background assets cannot use explicit BBOX"):
        make_fixed_collision_objects([kitchen])


def test_mesh_in_world_frame_applies_pose_rotation():
    """Background aggregate meshes are transformed by the source object's fixed pose."""
    import torch
    import trimesh

    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.utils.trimesh import mesh_in_world_frame

    mesh = trimesh.creation.box(extents=(2.0, 1.0, 1.0))
    yaw_90 = (0.0, 0.0, 2**-0.5, 2**-0.5)

    transformed = mesh_in_world_frame(
        mesh,
        Pose(position_xyz=(3.0, 4.0, 0.0), rotation_xyzw=yaw_90),
    )

    assert torch.allclose(
        torch.tensor(transformed.bounds[0], dtype=torch.float32), torch.tensor([2.5, 3.0, -0.5]), atol=1e-6
    )
    assert torch.allclose(
        torch.tensor(transformed.bounds[1], dtype=torch.float32), torch.tensor([3.5, 5.0, 0.5]), atol=1e-6
    )


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


def test_fixed_collision_object_from_factory_adds_mesh_loss():
    """Factory-built fixed meshes participate in MESH no-overlap loss."""
    from isaaclab_arena.relations.background_collision_object import FixedCollisionObject, make_fixed_collision_objects
    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.relation_solver_params import CollisionMode, RelationSolverParams
    from isaaclab_arena.relations.relation_solver_state import RelationSolverState
    from isaaclab_arena.relations.relations import On

    desk = _make_desk()
    box = _make_box()
    box.add_relation(On(desk))
    fixture = _mesh_box("fixture", (0.4, 0.4, 0.4), (0.35, 0.35, 0.2))
    collision_objects = make_fixed_collision_objects([fixture])
    assert isinstance(collision_objects[0], FixedCollisionObject)

    initial_positions = [{desk: (0.0, 0.0, 0.0), box: (0.3, 0.3, 0.1)}]
    solver = RelationSolver(RelationSolverParams(collision_mode=CollisionMode.MESH, verbose=False))
    state = RelationSolverState([desk, box], initial_positions, collision_objects=collision_objects)

    assert solver._compute_no_overlap_loss(state).sum().item() > 0.0


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
    solver_params = RelationSolverParams(verbose=False, save_position_history=False)
    solver = RelationSolver(solver_params)

    result = solver.solve([desk, box], initial_positions, collision_objects=[background])[0]

    box_world = box.get_bounding_box().translated(result[box])
    assert not box_world.overlaps(background.get_world_bounding_box(), margin=solver_params.clearance_m).item()


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


def test_relation_solver_state_rejects_object_as_collision_object():
    """An object cannot be both optimized and a fixed collision obstacle."""
    import pytest

    from isaaclab_arena.relations.relation_solver_state import RelationSolverState

    desk = _make_desk()
    box = _make_box()
    initial_positions = [{desk: (0.0, 0.0, 0.0), box: (0.3, 0.3, 0.1)}]

    with pytest.raises(AssertionError, match="disjoint"):
        RelationSolverState([desk, box], initial_positions, collision_objects=[box])


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


def _test_get_passive_collision_objects_filters(simulation_app) -> bool:
    """Only relation-free objects with a USD path and a fixed Pose are returned; Background is excluded."""
    from unittest.mock import MagicMock

    import isaaclab_arena.relations.passive_collision_objects as passive_collision_module
    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.relations.relations import IsAnchor
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose, PoseRange

    def fake_object(name, relations, usd_path, pose, spec=Object):
        obj = MagicMock(spec=spec)
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
    furniture = fake_object("furniture", [], "bg.usd", fixed_pose)
    # A Background qualifies on every other check but must still be excluded from AABB fallback.
    kitchen = fake_object("kitchen", [], "kitchen.usd", fixed_pose, spec=Background)
    kitchen_no_pose = fake_object("kitchen_no_pose", [], "kitchen.usd", None, spec=Background)
    counter_ref = fake_object("counter_ref", [], None, fixed_pose, spec=ObjectReference)
    counter_ref.parent_asset = kitchen
    anchored = fake_object("anchored", [IsAnchor()], "a.usd", fixed_pose)
    no_usd = fake_object("no_usd", [], None, fixed_pose)
    no_pose = fake_object("no_pose", [], "n.usd", None)
    ranged = fake_object("ranged", [], "r.usd", pose_range)

    scene = Scene()
    scene.assets = {
        o.name: o for o in [furniture, counter_ref, kitchen, kitchen_no_pose, anchored, no_usd, no_pose, ranged]
    }

    original = passive_collision_module.make_fixed_collision_objects
    passive_collision_module.make_fixed_collision_objects = lambda objects: list(objects)
    try:
        no_combine = passive_collision_module.get_passive_collision_objects(scene.assets.values())
        combined = passive_collision_module.get_passive_collision_objects(
            scene.assets.values(), include_background=True
        )
    finally:
        passive_collision_module.make_fixed_collision_objects = original

    return no_combine == [furniture, counter_ref] and combined == [furniture, kitchen, kitchen_no_pose]


def test_get_passive_collision_objects_filters():
    result = run_simulation_app_function(_test_get_passive_collision_objects_filters, headless=HEADLESS)
    assert result, "get_passive_collision_objects() returned the wrong subset"


def test_background_with_pose_range_rejected_for_aggregate_collision():
    from unittest.mock import MagicMock

    import pytest

    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.relations.passive_collision_objects import get_passive_collision_objects
    from isaaclab_arena.utils.pose import PoseRange

    background = MagicMock(spec=Background)
    background.name = "varying_background"
    background.usd_path = "background.usd"
    background.get_relations.return_value = []
    background.get_initial_pose.return_value = PoseRange(
        position_xyz_min=(0.0, 0.0, 0.0),
        position_xyz_max=(1.0, 0.0, 0.0),
        rpy_min=(0.0, 0.0, 0.0),
        rpy_max=(0.0, 0.0, 0.0),
    )

    with pytest.raises(AssertionError, match="must have a fixed Pose or no initial_pose"):
        get_passive_collision_objects([background], include_background=True)


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

    solver_params = RelationSolverParams(verbose=False, save_position_history=False)
    params = ObjectPlacerParams(
        placement_seed=0,
        solver_params=solver_params,
    )
    result = ObjectPlacer(params=params).place([desk, box], num_envs=1, collision_objects=[background])[0]

    assert result.success
    box_world = box.get_bounding_box().translated(result.positions[box])
    assert not box_world.overlaps(background.get_world_bounding_box(), margin=solver_params.clearance_m).item()


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

    solver_params = RelationSolverParams(verbose=False, save_position_history=False)
    params = ObjectPlacerParams(
        placement_seed=0,
        apply_positions_to_objects=False,
        solver_params=solver_params,
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
    assert not box_world.overlaps(background.get_world_bounding_box(), margin=solver_params.clearance_m).item()


def test_pooled_object_placer_multi_env_avoids_obstacle():
    """With num_envs > 1, every per-env pooled layout still avoids the background obstacle."""
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import On

    desk = _make_desk()
    box = _make_box()
    box.add_relation(On(desk))
    background = _make_background()

    num_envs = 3
    solver_params = RelationSolverParams(verbose=False, save_position_history=False)
    params = ObjectPlacerParams(
        placement_seed=0,
        apply_positions_to_objects=False,
        solver_params=solver_params,
    )
    pool = PooledObjectPlacer(
        objects=[desk, box],
        placer_params=params,
        pool_size=4,
        num_envs=num_envs,
        collision_objects=[background],
    )
    # sample_with_replacement maps slot i to env i % num_envs, so count == num_envs covers every env.
    layouts = pool.sample_with_replacement(num_envs)

    assert len(layouts) == num_envs
    for layout in layouts:
        box_world = box.get_bounding_box().translated(layout.positions[box])
        assert not box_world.overlaps(background.get_world_bounding_box(), margin=solver_params.clearance_m).item()


def test_arena_env_builder_forwards_background_collisions_by_default(monkeypatch):
    """Relation solving forwards scene assets for passive collision discovery."""
    from types import SimpleNamespace

    import isaaclab_arena.environments.arena_env_builder as builder_module
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver_params import CollisionMode, RelationSolverParams

    objects_with_relations = [object()]
    background_collision = object()
    calls = {}

    class Scene:
        assets = {"background_collision": background_collision}

        def get_objects_with_relations(self):
            return objects_with_relations

    def fake_solve_and_apply_relation_placement(
        objects, num_envs, placer_params, collision_objects=None, scene_assets=None
    ):
        calls["objects"] = objects
        calls["num_envs"] = num_envs
        calls["placer_params"] = placer_params
        calls["scene_assets"] = list(scene_assets)
        calls["collision_objects"] = collision_objects
        return "placement_event"

    monkeypatch.setattr(builder_module, "solve_and_apply_relation_placement", fake_solve_and_apply_relation_placement)
    placer_params = ObjectPlacerParams(solver_params=RelationSolverParams(collision_mode=CollisionMode.MESH))
    arena_env = SimpleNamespace(scene=Scene(), placer_params=placer_params)
    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2))

    builder._solve_relations()

    assert calls["objects"] == objects_with_relations
    assert calls["num_envs"] == 2
    assert calls["placer_params"] is placer_params
    assert calls["scene_assets"] == [background_collision]
    assert calls["collision_objects"] is None
    assert builder._placement_event_cfg == "placement_event"


def test_arena_env_builder_forwards_empty_relation_graph(monkeypatch):
    """Builder keeps one relation-placement call even when the relation graph is empty."""
    from types import SimpleNamespace

    import isaaclab_arena.environments.arena_env_builder as builder_module
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    calls = {}

    class Scene:
        assets = {}

        def get_objects_with_relations(self):
            return []

    def fake_solve_and_apply_relation_placement(
        objects, num_envs, placer_params, collision_objects=None, scene_assets=None
    ):
        calls["objects"] = objects
        calls["scene_assets"] = list(scene_assets)
        calls["collision_objects"] = collision_objects

    monkeypatch.setattr(builder_module, "solve_and_apply_relation_placement", fake_solve_and_apply_relation_placement)
    arena_env = SimpleNamespace(scene=Scene(), placer_params=None)
    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg())

    builder._solve_relations()

    assert calls["objects"] == []
    assert calls["scene_assets"] == []
    assert calls["collision_objects"] is None


def test_relation_placement_includes_background_mesh_for_object_mesh_override(monkeypatch):
    """Object-level MESH override enables aggregate background meshes."""
    import isaaclab_arena.environments.relation_solver_interface as interface_module
    from isaaclab_arena.assets.dummy_object import DummyObject
    from isaaclab_arena.environments.relation_solver_interface import solve_and_apply_relation_placement
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver_params import CollisionMode, RelationSolverParams
    from isaaclab_arena.relations.relations import IsAnchor
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    mesh_object = DummyObject(
        "mesh_object",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.1, -0.1, -0.1), max_point=(0.1, 0.1, 0.1)),
    )
    mesh_object.add_relation(IsAnchor())
    mesh_object.collision_mode = CollisionMode.MESH
    calls = {}

    def fake_get_passive_collision_objects(assets, include_background: bool = False):
        calls["assets"] = list(assets)
        calls["include_background"] = include_background
        return []

    class FakePooledObjectPlacer:
        had_fallbacks = False

        def __init__(self, objects, placer_params, pool_size, num_envs, collision_objects):
            calls["objects"] = objects
            calls["collision_objects"] = collision_objects

    monkeypatch.setattr(interface_module, "_get_passive_collision_objects", fake_get_passive_collision_objects)
    monkeypatch.setattr(interface_module, "PooledObjectPlacer", FakePooledObjectPlacer)
    placer_params = ObjectPlacerParams(solver_params=RelationSolverParams(collision_mode=CollisionMode.BBOX))

    solve_and_apply_relation_placement([mesh_object], num_envs=1, placer_params=placer_params, scene_assets=[])

    assert calls["assets"] == []
    assert calls["include_background"] is True
    assert calls["objects"] == [mesh_object]
    assert calls["collision_objects"] == []


def test_relation_placement_includes_background_mesh_for_background_override(monkeypatch):
    """A passive Background can opt into mesh collision when the solver default is BBOX."""
    import isaaclab_arena.environments.relation_solver_interface as interface_module
    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.assets.dummy_object import DummyObject
    from isaaclab_arena.environments.relation_solver_interface import solve_and_apply_relation_placement
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver_params import CollisionMode, RelationSolverParams
    from isaaclab_arena.relations.relations import IsAnchor
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    background = Background.__new__(Background)
    background.collision_mode = CollisionMode.MESH
    placed_object = DummyObject(
        "placed_object",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.1, -0.1, -0.1), max_point=(0.1, 0.1, 0.1)),
    )
    placed_object.add_relation(IsAnchor())
    calls = {}

    def fake_get_passive_collision_objects(assets, include_background: bool = False):
        calls["assets"] = list(assets)
        calls["include_background"] = include_background
        return []

    class FakePooledObjectPlacer:
        had_fallbacks = False

        def __init__(self, objects, placer_params, pool_size, num_envs, collision_objects):
            calls["objects"] = objects
            calls["collision_objects"] = collision_objects

    monkeypatch.setattr(interface_module, "_get_passive_collision_objects", fake_get_passive_collision_objects)
    monkeypatch.setattr(interface_module, "PooledObjectPlacer", FakePooledObjectPlacer)
    placer_params = ObjectPlacerParams(solver_params=RelationSolverParams(collision_mode=CollisionMode.BBOX))

    solve_and_apply_relation_placement(
        [placed_object], num_envs=1, placer_params=placer_params, scene_assets=[background]
    )

    assert calls["assets"] == [background]
    assert calls["include_background"] is True
    assert calls["objects"] == [placed_object]
    assert calls["collision_objects"] == []


def test_relation_placement_skips_background_mesh_for_default_bbox(monkeypatch):
    """Default BBOX mode uses individual passive objects, not aggregate whole-scene meshes."""
    import isaaclab_arena.environments.relation_solver_interface as interface_module
    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.assets.dummy_object import DummyObject
    from isaaclab_arena.environments.relation_solver_interface import solve_and_apply_relation_placement
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver_params import CollisionMode, RelationSolverParams
    from isaaclab_arena.relations.relations import IsAnchor
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    background = Background.__new__(Background)
    background.collision_mode = None
    placed_object = DummyObject(
        "placed_object",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.1, -0.1, -0.1), max_point=(0.1, 0.1, 0.1)),
    )
    placed_object.add_relation(IsAnchor())
    calls = {}

    def fake_get_passive_collision_objects(assets, include_background: bool = False):
        calls["assets"] = list(assets)
        calls["include_background"] = include_background
        return []

    class FakePooledObjectPlacer:
        had_fallbacks = False

        def __init__(self, objects, placer_params, pool_size, num_envs, collision_objects):
            calls["objects"] = objects
            calls["collision_objects"] = collision_objects

    monkeypatch.setattr(interface_module, "_get_passive_collision_objects", fake_get_passive_collision_objects)
    monkeypatch.setattr(interface_module, "PooledObjectPlacer", FakePooledObjectPlacer)
    placer_params = ObjectPlacerParams(solver_params=RelationSolverParams(collision_mode=CollisionMode.BBOX))

    solve_and_apply_relation_placement(
        [placed_object], num_envs=1, placer_params=placer_params, scene_assets=[background]
    )

    assert calls["assets"] == [background]
    assert calls["include_background"] is False
    assert calls["objects"] == [placed_object]
    assert calls["collision_objects"] == []
