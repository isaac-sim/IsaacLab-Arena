# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _make_drawer_reference(tmp_path, nested):
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_reference import ArticulationBodyReference, ArticulationObjectReference
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.tests.test_reference_objects import _write_scaled_drawer_articulation
    from isaaclab_arena.utils.pose import Pose

    usd_path = tmp_path / "drawer.usda"
    _write_scaled_drawer_articulation(
        usd_path, "prismatic", default_prim_name="Workbench", drawer_physics_in_variant=True
    )
    source = Object(
        name="workbench" if nested else "cabinet",
        usd_path=str(usd_path),
        object_type=ObjectType.BASE if nested else ObjectType.ARTICULATION,
        initial_pose=Pose(position_xyz=(0.2, -0.3, 1.0)),
        scale=(0.5, 0.5, 0.5),
        spawn_cfg_addon={"variants": {"Physics": "physics"}},
    )
    if nested:
        cabinet = ArticulationObjectReference(
            name="cabinet",
            parent_asset=source,
            prim_path=source.get_prim_path() + "/Cabinet",
            initial_joint_positions={"drawer_joint": 0.3},
        )
    else:
        cabinet = source
        cabinet.object_cfg.init_state.joint_pos = {"drawer_joint": 0.3}
    drawer = ArticulationBodyReference(
        name="drawer", articulation=cabinet, body_prim_path="Drawer" if nested else "Cabinet/Drawer"
    )
    return source, cabinet, drawer


def _test_body_reference_geometry_and_graph(simulation_app, tmp_path, nested):
    from isaaclab_arena.assets.object_reference import ArticulationBodyReference, ArticulationBodyReferenceCfg
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import _instantiate_object_reference
    from isaaclab_arena.environment_spec.arena_env_graph_types import ObjectReferenceSpec

    source, cabinet, drawer = _make_drawer_reference(tmp_path, nested)
    assert drawer.articulation is cabinet
    assert drawer.root_asset is source
    assert drawer.object_type == ObjectType.BASE
    assert isinstance(drawer.object_cfg, ArticulationBodyReferenceCfg)
    assert drawer.object_cfg.articulation_name == "cabinet"
    assert drawer.object_cfg.body_name == "Drawer"
    assert drawer.object_cfg.spawn is None
    assert drawer.get_event_cfgs() == []
    assert drawer.prim_path_in_parent_usd == "/Workbench/Cabinet/Drawer"
    assert drawer.get_initial_pose().position_xyz == pytest.approx((0.6, -0.3, 1.0))
    torch.testing.assert_close(drawer.get_bounding_box().min_point, torch.tensor([[-0.2, -0.1, -0.05]]))
    torch.testing.assert_close(drawer.get_bounding_box().max_point, torch.tensor([[0.4, 0.1, 0.05]]))

    for prim_path in ("Drawer" if nested else "Cabinet/Drawer", drawer.get_prim_path()):
        spec = ObjectReferenceSpec(id="drawer", kind="articulation_body", parent_id="cabinet", prim_path=prim_path)
        converted = _instantiate_object_reference(spec, cabinet)
        assert isinstance(converted, ArticulationBodyReference)
        assert converted.get_initial_pose() == drawer.get_initial_pose()
        assert converted.root_asset is source
    return True


@pytest.mark.parametrize("nested", [False, True], ids=["standalone", "nested"])
def test_body_reference_geometry_and_graph(tmp_path, nested):
    assert run_function_with_persistent_simulation_app(
        _test_body_reference_geometry_and_graph, tmp_path=tmp_path, nested=nested
    )


def _test_body_reference_tracks_parallel_motion_and_partial_reset(simulation_app, tmp_path, nested):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    source, cabinet, drawer = _make_drawer_reference(tmp_path, nested)
    assets = [source, cabinet, drawer] if nested else [cabinet, drawer]
    args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "4"])
    definition = IsaacLabArenaEnvironment(name="body_reference", scene=Scene(assets=assets))
    env = ArenaEnvBuilder(definition, arena_env_builder_cfg_from_argparse(args)).make_registered()
    try:
        env.reset()
        scene = env.unwrapped.scene
        world = env.unwrapped.arena_world
        assert set(scene.articulations) == {cabinet.name}
        assert drawer.name in scene.extras and drawer.name not in scene.rigid_objects
        articulation = scene[cabinet.name]
        joint_id = articulation.data.joint_names.index("drawer_joint")
        joint_pos = torch.tensor([[0.0], [0.1], [0.2], [0.25]], device=env.unwrapped.device)
        articulation.write_joint_position_to_sim_index(position=joint_pos, joint_ids=[joint_id])

        def check_body_poses(expected_joint_pos):
            T_W_D = world.get_pose_w(drawer.name)
            torch.testing.assert_close(T_W_D, world.get_body_pose_w(cabinet.name, "Drawer"))
            t_W_B = world.get_body_pose_w(cabinet.name, "Base")[:, :3]
            expected_t_B_D = torch.zeros((4, 3), device=env.unwrapped.device)
            expected_t_B_D[:, 0] = 0.1 + expected_joint_pos[:, 0]
            torch.testing.assert_close(T_W_D[:, :3] - t_W_B, expected_t_B_D, atol=1e-4, rtol=0)
            torch.testing.assert_close(
                world.get_pose_e(drawer.name)[:, :3], T_W_D[:, :3] - scene.env_origins, atol=1e-5, rtol=0
            )

        check_body_poses(joint_pos)
        surviving_poses = world.get_pose_w(drawer.name)[[0, 2]].clone()
        env.unwrapped._reset_idx(torch.tensor([1, 3], device=env.unwrapped.device))
        joint_pos[[1, 3]] = 0.3
        check_body_poses(joint_pos)
        torch.testing.assert_close(world.get_pose_w(drawer.name)[[0, 2]], surviving_poses)
    finally:
        env.close()
    return True


@pytest.mark.parametrize("nested", [False, True], ids=["standalone", "nested"])
def test_body_reference_tracks_parallel_motion_and_partial_reset(tmp_path, nested):
    assert run_function_with_persistent_simulation_app(
        _test_body_reference_tracks_parallel_motion_and_partial_reset, tmp_path=tmp_path, nested=nested
    )


def _test_body_reference_rejects_invalid_selection(simulation_app, tmp_path):
    from isaaclab_arena.assets.object_reference import ArticulationBodyReference

    source, cabinet, _ = _make_drawer_reference(tmp_path, nested=True)
    with pytest.raises(AssertionError, match="requires an articulation parent"):
        ArticulationBodyReference(name="bad", articulation=source, body_prim_path="Cabinet/Drawer")
    for body_path in ("../Sibling/Drawer", "/Cabinet/Drawer", "Drawer.xformOp:translate"):
        with pytest.raises(AssertionError, match="must be a relative prim path"):
            ArticulationBodyReference(name="bad", articulation=cabinet, body_prim_path=body_path)
    with pytest.raises(AssertionError, match="must select a rigid body"):
        ArticulationBodyReference(name="bad", articulation=cabinet, body_prim_path="Drawer/Collider")
    with pytest.raises(ValueError, match="No prim found"):
        ArticulationBodyReference(name="bad", articulation=cabinet, body_prim_path="MissingBody")
    return True


def test_body_reference_rejects_invalid_selection(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_body_reference_rejects_invalid_selection, tmp_path=tmp_path
    )


def _test_body_reference_to_articulation_root(simulation_app, tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_reference import ArticulationBodyReference
    from isaaclab_arena.assets.object_type import ObjectType

    usd_path = tmp_path / "root_body.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    root = UsdGeom.Xform.Define(stage, "/AuthoredRoot").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    stage.GetRootLayer().Save()
    articulation = Object(name="renamed_root", usd_path=str(usd_path), object_type=ObjectType.ARTICULATION)
    body = ArticulationBodyReference(name="body", articulation=articulation, body_prim_path=".")
    assert body.get_prim_path() == articulation.get_prim_path()
    assert body.object_cfg.body_name == "renamed_root"
    return True


def test_body_reference_to_articulation_root(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_body_reference_to_articulation_root, tmp_path=tmp_path)
