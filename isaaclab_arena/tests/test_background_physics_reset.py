# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test discovery and reset of dynamic entities nested in background USDs."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _create_background_usds(
    background_path: str,
    online_asset_path: str,
    include_joint_network: bool = True,
) -> None:
    """Create composed loose, jointed, articulated, and guide content."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    online_stage = Usd.Stage.CreateNew(online_asset_path)
    online_root = UsdGeom.Xform.Define(online_stage, "/OnlineAsset")
    online_stage.SetDefaultPrim(online_root.GetPrim())
    online_body = UsdGeom.Cube.Define(online_stage, "/OnlineAsset/body")
    online_body.CreateSizeAttr(0.1)
    UsdPhysics.CollisionAPI.Apply(online_body.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(online_body.GetPrim())
    online_stage.GetRootLayer().Save()

    stage = Usd.Stage.CreateNew(background_path)
    root = UsdGeom.Xform.Define(stage, "/Background")
    stage.SetDefaultPrim(root.GetPrim())

    def add_cube(prim_path: str, position: tuple[float, float, float]) -> Usd.Prim:
        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.CreateSizeAttr(0.2)
        cube.AddTranslateOp().Set(Gf.Vec3d(*position))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(1.0)
        return cube.GetPrim()

    add_cube("/Background/free_body", (-0.6, 0.0, 1.0))
    if include_joint_network:
        add_cube("/Background/hinged/base", (-0.2, 0.0, 1.0))
        add_cube("/Background/hinged/door", (0.0, 0.0, 1.0))
        hinge = UsdPhysics.RevoluteJoint.Define(stage, "/Background/hinged/joint")
        hinge.CreateBody0Rel().SetTargets(["/Background/hinged/base"])
        hinge.CreateBody1Rel().SetTargets(["/Background/hinged/door"])
        hinge.CreateAxisAttr("Z")

    def add_articulation(name: str, x: float) -> None:
        prim_path = f"/Background/{name}"
        articulation = UsdGeom.Xform.Define(stage, prim_path)
        UsdPhysics.ArticulationRootAPI.Apply(articulation.GetPrim())
        add_cube(f"{prim_path}/base", (x, 0.0, 1.0))
        add_cube(f"{prim_path}/link", (x + 0.2, 0.0, 1.0))
        fixed_joint = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/world_joint")
        fixed_joint.CreateBody1Rel().SetTargets([f"{prim_path}/base"])
        articulation_joint = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/joint")
        articulation_joint.CreateBody0Rel().SetTargets([f"{prim_path}/base"])
        articulation_joint.CreateBody1Rel().SetTargets([f"{prim_path}/link"])
        articulation_joint.CreateAxisAttr("Z")

    add_articulation("articulation", 0.3)
    add_articulation("referenced_articulation", 0.8)

    online_visual = stage.DefinePrim("/Background/Prop_OnlineVisual", "Xform")
    online_visual.GetReferences().AddReference(online_asset_path)
    online_visual.SetInstanceable(True)
    placement_envelope = UsdGeom.Cube.Define(stage, "/Background/Prop_PlacementEnvelope")
    placement_envelope.CreatePurposeAttr("guide")
    stage.GetRootLayer().Save()


def _test_background_physics_discovery_and_reset(
    _,
    cli_options: tuple[str, ...] = (),
    check_interactivity: bool = True,
    include_joint_network: bool = True,
) -> bool:
    import tempfile
    import torch

    import pytest
    from pxr import UsdPhysics

    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.terms.events import ResetBackgroundPhysics, reset_articulation_pose_and_joints
    from isaaclab_arena.utils.usd_prim_tree import load_usd_physics_roots

    assert ResetBackgroundPhysics._is_unavailable_backend_error(
        RuntimeError("Failed to create rigid body at: /World/body. Please check PhysX logs.")
    )
    assert ResetBackgroundPhysics._is_unavailable_backend_error(
        KeyError("No articulations matching pattern '/World/body'")
    )
    assert not ResetBackgroundPhysics._is_unavailable_backend_error(RuntimeError("No prim found at '/World/body'."))
    assert not ResetBackgroundPhysics._is_unavailable_backend_error(torch.cuda.OutOfMemoryError())

    with tempfile.TemporaryDirectory() as temp_dir:
        background_path = f"{temp_dir}/background.usd"
        online_asset_path = f"{temp_dir}/online_asset.usd"
        _create_background_usds(background_path, online_asset_path, include_joint_network)

        records = load_usd_physics_roots(background_path)
        expected_records = {
            "Prop_OnlineVisual/body": ObjectType.RIGID,
            "articulation": ObjectType.ARTICULATION,
            "free_body": ObjectType.RIGID,
            "referenced_articulation": ObjectType.ARTICULATION,
        }
        if include_joint_network:
            expected_records.update({
                "hinged/base": ObjectType.RIGID,
                "hinged/door": ObjectType.RIGID,
            })
        assert records == expected_records
        assert "Prop_PlacementEnvelope" not in records
        assert not any(path.startswith("articulation/") for path in records)

        background = Background(
            "background",
            background_path,
            object_min_z=0.0,
            reset_nested_physics=True,
        )
        referenced_free_body = ObjectReference(
            name="referenced_free_body",
            prim_path="{ENV_REGEX_NS}/background/free_body",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        referenced_articulation = ObjectReference(
            name="referenced_articulation",
            prim_path="{ENV_REGEX_NS}/background/referenced_articulation",
            parent_asset=background,
            object_type=ObjectType.ARTICULATION,
        )
        assert referenced_free_body.get_event_cfg()[1] is not None
        referenced_articulation_event = referenced_articulation.get_event_cfg()[1]
        assert referenced_articulation_event is not None
        assert referenced_articulation_event.func is reset_articulation_pose_and_joints
        physics_paths = background.get_nested_physics_prim_paths({
            "{ENV_REGEX_NS}/background/free_body": ObjectType.RIGID,
            "{ENV_REGEX_NS}/background/referenced_articulation": ObjectType.ARTICULATION,
        })
        assert len(physics_paths) == (4 if include_joint_network else 2)
        assert sum(object_type == ObjectType.ARTICULATION for object_type in physics_paths.values()) == 1
        assert sum(object_type == ObjectType.RIGID for object_type in physics_paths.values()) == (
            3 if include_joint_network else 1
        )
        base_reference_paths = background.get_nested_physics_prim_paths(
            {"{ENV_REGEX_NS}/background/free_body": ObjectType.BASE}
        )
        assert "{ENV_REGEX_NS}/background/free_body" in base_reference_paths
        with pytest.raises(AssertionError, match="lies inside articulation root"):
            background.get_nested_physics_prim_paths({"{ENV_REGEX_NS}/background/articulation/link": ObjectType.RIGID})

        scene = Scene(assets=[background, referenced_free_body, referenced_articulation])
        arena_env = IsaacLabArenaEnvironment(name="background_physics_reset", scene=scene)
        args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "2", *cli_options])
        builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args))
        env_cfg, _ = builder.compose_manager_cfg()
        reset_paths = scene.get_background_physics_paths()["background"]
        assert len(reset_paths) == (4 if include_joint_network else 2)
        assert list(vars(env_cfg.events))[0] == "reset_background_physics"

        env = builder.make_registered(env_cfg)
        try:
            env.reset()
            base_env = env.unwrapped
            reset_term = base_env.event_manager.get_term_cfg("reset_background_physics").func
            rigid_reset = next(
                reset
                for reset in reset_term._rigid_resets
                if reset.asset.cfg.prim_path.endswith("/Prop_OnlineVisual/body")
            )
            articulation_reset = reset_term._articulation_resets[0]
            rigid = rigid_reset.asset
            articulation = articulation_reset.asset
            referenced_rigid = base_env.scene["referenced_free_body"]
            referenced_articulation_asset = base_env.scene["referenced_articulation"]
            initial_pose = rigid.data.root_pose_w.torch.clone()
            initial_referenced_pose = referenced_rigid.data.root_pose_w.torch.clone()
            initial_joint_position = articulation.data.joint_pos.torch.clone()
            initial_referenced_joint_position = referenced_articulation_asset.data.joint_pos.torch.clone()
            assert rigid_reset.root_pose_local.shape == (7,)
            assert articulation_reset.root_pose_local.shape == (7,)
            assert articulation_reset.joint_position.shape == initial_joint_position.shape[1:]
            env_ids = torch.arange(2, device=base_env.device)
            if check_interactivity:
                probe_velocity = torch.zeros_like(rigid.data.root_vel_w.torch)
                probe_velocity[0, 0] = 2.0
                rigid.write_root_velocity_to_sim_index(root_velocity=probe_velocity, env_ids=env_ids)
                base_env.sim.step()
                base_env.scene.update(base_env.step_dt)
                rigid.update(base_env.step_dt)
                assert rigid.data.root_pose_w.torch[0, 0] > initial_pose[0, 0]
                base_env.reset(env_ids=torch.arange(2, device=base_env.device))

            moved_pose = initial_pose.clone()
            moved_pose[:, 0] += 0.4
            rigid.write_root_pose_to_sim_index(root_pose=moved_pose, env_ids=env_ids)
            rigid.write_root_velocity_to_sim_index(
                root_velocity=torch.ones((2, 6), device=base_env.device),
                env_ids=env_ids,
            )
            moved_referenced_pose = initial_referenced_pose.clone()
            moved_referenced_pose[:, 1] += 0.3
            referenced_rigid.write_root_pose_to_sim_index(root_pose=moved_referenced_pose, env_ids=env_ids)
            moved_joint_position = initial_joint_position.clone()
            moved_joint_position[:, 0] += 0.2
            moved_referenced_joint_position = initial_referenced_joint_position.clone()
            moved_referenced_joint_position[:, 0] += 0.2
            articulation.write_root_velocity_to_sim_index(
                root_velocity=torch.ones((2, 6), device=base_env.device),
                env_ids=env_ids,
            )
            articulation.write_joint_position_to_sim_index(position=moved_joint_position, env_ids=env_ids)
            articulation.write_joint_velocity_to_sim_index(
                velocity=torch.ones_like(articulation.data.joint_vel.torch),
                env_ids=env_ids,
            )
            referenced_articulation_asset.write_joint_position_to_sim_index(
                position=moved_referenced_joint_position,
                env_ids=env_ids,
            )
            referenced_articulation_asset.write_joint_velocity_to_sim_index(
                velocity=torch.ones_like(referenced_articulation_asset.data.joint_vel.torch),
                env_ids=env_ids,
            )

            runtime_rigid_prim = base_env.scene.stage.GetPrimAtPath(
                rigid.cfg.prim_path.replace(base_env.scene.env_regex_ns, base_env.scene.env_prim_paths[0])
            )
            assert runtime_rigid_prim.IsValid()
            kinematic_attr = UsdPhysics.RigidBodyAPI(runtime_rigid_prim).GetKinematicEnabledAttr()
            assert not kinematic_attr.IsValid() or not kinematic_attr.Get()

            # Reset env 1 to verify the env-0 snapshot is translated through env-local coordinates.
            base_env.reset(env_ids=torch.tensor([1], device=base_env.device))
            reset_pose = rigid.data.root_pose_w.torch
            reset_velocity = rigid.data.root_vel_w.torch
            reset_articulation_velocity = articulation.data.root_vel_w.torch
            reset_joint_position = articulation.data.joint_pos.torch
            reset_joint_velocity = articulation.data.joint_vel.torch
            assert torch.allclose(reset_pose[0], moved_pose[0])
            assert torch.allclose(reset_pose[1], initial_pose[1], atol=1.0e-5)
            assert torch.count_nonzero(reset_velocity[1]) == 0
            assert torch.count_nonzero(reset_articulation_velocity[1]) == 0
            assert torch.allclose(reset_joint_position[1], initial_joint_position[1], atol=1.0e-5)
            assert torch.count_nonzero(reset_joint_velocity[1]) == 0
            assert torch.allclose(referenced_rigid.data.root_pose_w.torch[0], moved_referenced_pose[0])
            assert torch.allclose(referenced_rigid.data.root_pose_w.torch[1], initial_referenced_pose[1], atol=1.0e-5)
            assert torch.allclose(
                referenced_articulation_asset.data.joint_pos.torch[0],
                moved_referenced_joint_position[0],
            )
            assert torch.allclose(
                referenced_articulation_asset.data.joint_pos.torch[1],
                initial_referenced_joint_position[1],
                atol=1.0e-5,
            )
            assert torch.count_nonzero(referenced_articulation_asset.data.joint_vel.torch[1]) == 0
        finally:
            env.close()
    return True


def test_background_physics_discovery_and_reset():
    assert run_function_with_persistent_simulation_app(_test_background_physics_discovery_and_reset)


def test_background_physics_reset_without_fabric():
    assert run_function_with_persistent_simulation_app(
        _test_background_physics_discovery_and_reset,
        cli_options=("--disable_fabric",),
    )


def test_background_physics_reset_with_newton():
    assert run_function_with_persistent_simulation_app(
        _test_background_physics_discovery_and_reset,
        cli_options=("--presets", "newton"),
        check_interactivity=False,
        include_joint_network=False,
    )
