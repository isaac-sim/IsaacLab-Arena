# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check geometry shared by ordinary objects and assigned object-set members."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _write_nested_box_usd(source_path, object_type_name):
    import trimesh

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(source_path))
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    if object_type_name == "ARTICULATION":
        UsdPhysics.ArticulationRootAPI.Apply(root)
    body = UsdGeom.Xform.Define(stage, "/Asset/Body")
    body.AddTranslateOp().Set(Gf.Vec3d(3.0, 0.0, 0.0))
    if object_type_name != "BASE":
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    box = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Body/Box")
    mesh.CreatePointsAttr().Set(box.vertices.tolist())
    mesh.CreateFaceVertexCountsAttr().Set([3] * len(box.faces))
    mesh.CreateFaceVertexIndicesAttr().Set(box.faces.reshape(-1).tolist())
    stage.GetRootLayer().Save()


def _test_object_geometry_preserves_pose_frame(simulation_app, tmp_path, object_type_name, expected_center_x):
    import numpy as np
    import torch

    from isaaclab.sim import UsdFileCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType

    source_path = tmp_path / f"{object_type_name.lower()}.usda"
    _write_nested_box_usd(source_path, object_type_name)
    scene_object = Object(
        name="box",
        object_type=ObjectType[object_type_name],
        spawn_cfg=UsdFileCfg(usd_path=str(source_path)),
    )
    expected_min = torch.tensor([[expected_center_x - 1.0, -2.0, -3.0]])
    expected_max = torch.tensor([[expected_center_x + 1.0, 2.0, 3.0]])
    for bounds in (scene_object.get_bounding_box(), scene_object.get_bounding_box_for_env(2)):
        torch.testing.assert_close(bounds.min_point, expected_min)
        torch.testing.assert_close(bounds.max_point, expected_max)
    bounds_per_env = scene_object.get_bounding_box_per_env(3)
    torch.testing.assert_close(bounds_per_env.min_point, expected_min.expand(3, 3))
    torch.testing.assert_close(bounds_per_env.max_point, expected_max.expand(3, 3))
    np.testing.assert_allclose(
        scene_object.get_collision_mesh().bounds,
        [[expected_center_x - 1.0, -2.0, -3.0], [expected_center_x + 1.0, 2.0, 3.0]],
    )
    return True


@pytest.mark.parametrize(
    ("object_type_name", "expected_center_x"), [("RIGID", 0.0), ("BASE", 3.0), ("ARTICULATION", 3.0)]
)
def test_object_geometry_preserves_pose_frame(tmp_path, object_type_name, expected_center_x):
    assert run_function_with_persistent_simulation_app(
        _test_object_geometry_preserves_pose_frame,
        tmp_path=tmp_path,
        object_type_name=object_type_name,
        expected_center_x=expected_center_x,
    )


def _test_object_custom_bounds_survive_spawn_replacement(simulation_app):
    import torch

    from isaaclab.sim import CuboidCfg, SphereCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    scene_object = Object(name="box", object_type=ObjectType.RIGID, spawn_cfg=CuboidCfg(size=(1.0, 2.0, 3.0)))
    torch.testing.assert_close(scene_object.get_bounding_box().size, torch.tensor([[1.0, 2.0, 3.0]]))
    custom_bounds = AxisAlignedBoundingBox((1.0, 2.0, 3.0), (4.0, 6.0, 8.0))
    scene_object.bounding_box = custom_bounds
    scene_object.spawn_cfg = SphereCfg(radius=3.0)

    for bounds in (scene_object.get_bounding_box(), scene_object.get_bounding_box_for_env(1)):
        torch.testing.assert_close(bounds.min_point, custom_bounds.min_point)
        torch.testing.assert_close(bounds.max_point, custom_bounds.max_point)
    bounds_per_env = scene_object.get_bounding_box_per_env(2)
    torch.testing.assert_close(bounds_per_env.min_point, custom_bounds.min_point.expand(2, 3))
    torch.testing.assert_close(bounds_per_env.max_point, custom_bounds.max_point.expand(2, 3))

    scene_object.bounding_box = None
    torch.testing.assert_close(scene_object.get_bounding_box_per_env(2).size, torch.full((2, 3), 6.0))
    return True


def test_object_custom_bounds_survive_spawn_replacement():
    assert run_function_with_persistent_simulation_app(_test_object_custom_bounds_survive_spawn_replacement)


def _test_assigned_geometry_tracks_member_configurations(simulation_app):
    import torch

    from isaaclab.sim import CuboidCfg, SphereCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.object_type import ObjectType

    members = [
        Object(name="small_box", object_type=ObjectType.RIGID, spawn_cfg=CuboidCfg(size=(1.0, 2.0, 3.0))),
        Object(name="large_box", object_type=ObjectType.RIGID, spawn_cfg=CuboidCfg(size=(2.0, 3.0, 4.0))),
    ]
    object_set = RigidObjectSet(name="boxes", objects=members)
    object_set.bind_asset_assignment((1, 0, 1))
    torch.testing.assert_close(
        object_set.get_bounding_box_per_env(3).size,
        torch.tensor([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
    )

    object_set.spawn_cfg.assets_cfg[0].size = (4.0, 5.0, 6.0)
    torch.testing.assert_close(
        object_set.get_bounding_box_per_env(3).size,
        torch.tensor([[2.0, 3.0, 4.0], [4.0, 5.0, 6.0], [2.0, 3.0, 4.0]]),
    )
    object_set.spawn_cfg.assets_cfg[1] = SphereCfg(radius=3.0)
    torch.testing.assert_close(
        object_set.get_bounding_box_per_env(3).size,
        torch.tensor([[6.0, 6.0, 6.0], [4.0, 5.0, 6.0], [6.0, 6.0, 6.0]]),
    )
    torch.testing.assert_close(object_set.get_bounding_box_for_env(1).size, torch.tensor([[4.0, 5.0, 6.0]]))
    assert object_set.asset_indices_by_env == (1, 0, 1)
    return True


def test_assigned_geometry_tracks_member_configurations():
    assert run_function_with_persistent_simulation_app(_test_assigned_geometry_tracks_member_configurations)


def _test_singleton_geometry_tracks_native_config_replacement(simulation_app, tmp_path):
    import torch

    from isaaclab.sim import SphereCfg, UsdFileCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.object_type import ObjectType

    source_path = tmp_path / "nested.usda"
    _write_nested_box_usd(source_path, "RIGID")
    member = Object(name="box", object_type=ObjectType.RIGID, spawn_cfg=UsdFileCfg(usd_path=str(source_path)))
    object_set = RigidObjectSet(name="singleton", objects=[member])
    object_set.bind_asset_assignment((0, 0))
    torch.testing.assert_close(object_set.get_bounding_box().size, torch.tensor([[2.0, 4.0, 6.0]]))
    assert object_set.get_collision_mesh() is not None
    assert object_set.get_contact_sensor_prim_path() == object_set.get_prim_path() + "/Body"

    object_set.object_cfg.spawn = SphereCfg(radius=2.0)
    torch.testing.assert_close(object_set.get_bounding_box_per_env(2).size, torch.full((2, 3), 4.0))
    assert object_set.get_collision_mesh() is None
    assert object_set.get_contact_sensor_prim_path() == object_set.get_prim_path()
    assert object_set.asset_indices_by_env == (0, 0)
    return True


def test_singleton_geometry_tracks_native_config_replacement(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_singleton_geometry_tracks_native_config_replacement, tmp_path=tmp_path
    )
