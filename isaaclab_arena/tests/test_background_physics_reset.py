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

    articulation = UsdGeom.Xform.Define(stage, "/Background/articulation")
    UsdPhysics.ArticulationRootAPI.Apply(articulation.GetPrim())
    add_cube("/Background/articulation/base", (0.3, 0.0, 1.0))
    add_cube("/Background/articulation/link", (0.5, 0.0, 1.0))
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/Background/articulation/world_joint")
    fixed_joint.CreateBody1Rel().SetTargets(["/Background/articulation/base"])
    articulation_joint = UsdPhysics.RevoluteJoint.Define(stage, "/Background/articulation/joint")
    articulation_joint.CreateBody0Rel().SetTargets(["/Background/articulation/base"])
    articulation_joint.CreateBody1Rel().SetTargets(["/Background/articulation/link"])
    articulation_joint.CreateAxisAttr("Z")

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

    from isaaclab.assets import ArticulationCfg, RigidObjectCfg
    from pxr import UsdPhysics

    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.usd_prim_tree import load_usd_physics_roots

    with tempfile.TemporaryDirectory() as temp_dir:
        background_path = f"{temp_dir}/background.usd"
        online_asset_path = f"{temp_dir}/online_asset.usd"
        _create_background_usds(background_path, online_asset_path, include_joint_network)

        records = {record.relative_path: record.object_type for record in load_usd_physics_roots(background_path)}
        expected_records = {
            "Prop_OnlineVisual/body": ObjectType.RIGID,
            "articulation": ObjectType.ARTICULATION,
            "free_body": ObjectType.RIGID,
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
        configs = background.get_nested_physics_cfgs({"{ENV_REGEX_NS}/background/free_body": ObjectType.RIGID})
        assert len(configs) == (4 if include_joint_network else 2)
        assert sum(isinstance(cfg, ArticulationCfg) for cfg in configs.values()) == 1
        assert sum(isinstance(cfg, RigidObjectCfg) for cfg in configs.values()) == (3 if include_joint_network else 1)
        assert all(cfg.spawn is None for cfg in configs.values())

        scene = Scene(
            assets=[
                Background(
                    "background",
                    background_path,
                    object_min_z=0.0,
                    reset_nested_physics=True,
                )
            ]
        )
        arena_env = IsaacLabArenaEnvironment(name="background_physics_reset", scene=scene)
        args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "2", *cli_options])
        builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args))
        env_cfg, _ = builder.compose_manager_cfg()
        hidden_entities = scene.get_background_physics_entities()["background"]
        assert len(hidden_entities) == (5 if include_joint_network else 3)
        assert list(vars(env_cfg.events))[0] == "reset_nested_background_physics"

        env = builder.make_registered(env_cfg)
        try:
            env.reset()
            base_env = env.unwrapped
            initial_state = base_env.scene.get_state()
            rigid_name = next(name for name, path in hidden_entities.items() if path.endswith("/free_body"))
            articulation_name = next(name for name in hidden_entities if name in base_env.scene.articulations)

            rigid = base_env.scene.rigid_objects[rigid_name]
            initial_pose = initial_state["rigid_object"][rigid_name]["root_pose"].clone()
            initial_joint_position = initial_state["articulation"][articulation_name]["joint_position"].clone()
            if check_interactivity:
                probe_velocity = torch.zeros((1, 6), device=base_env.device)
                probe_velocity[:, 0] = 2.0
                rigid.write_root_velocity_to_sim_index(
                    root_velocity=probe_velocity,
                    env_ids=torch.tensor([0], device=base_env.device),
                )
                base_env.sim.step()
                base_env.scene.update(base_env.step_dt)
                assert rigid.data.root_pose_w.torch[0, 0] > initial_pose[0, 0]
                base_env.reset(env_ids=torch.arange(2, device=base_env.device))

            moved_pose = initial_pose.clone()
            moved_pose[:, 0] += 0.4
            env_ids = torch.arange(2, device=base_env.device)
            rigid.write_root_pose_to_sim_index(root_pose=moved_pose, env_ids=env_ids)
            rigid.write_root_velocity_to_sim_index(
                root_velocity=torch.ones((2, 6), device=base_env.device),
                env_ids=env_ids,
            )

            articulation_asset = base_env.scene.articulations[articulation_name]
            moved_joint_position = initial_joint_position.clone()
            moved_joint_position[:, 0] += 0.2
            articulation_asset.write_joint_position_to_sim_index(position=moved_joint_position, env_ids=env_ids)

            runtime_rigid_prim = base_env.scene.stage.GetPrimAtPath(
                hidden_entities[rigid_name].replace("{ENV_REGEX_NS}", base_env.scene.env_prim_paths[0])
            )
            kinematic_attr = UsdPhysics.RigidBodyAPI(runtime_rigid_prim).GetKinematicEnabledAttr()
            assert not kinematic_attr or not kinematic_attr.Get()

            base_env.reset(env_ids=torch.tensor([0], device=base_env.device))
            reset_state = base_env.scene.get_state()
            assert torch.allclose(
                reset_state["rigid_object"][rigid_name]["root_pose"][0],
                initial_pose[0],
                atol=1.0e-5,
            )
            assert torch.allclose(
                reset_state["rigid_object"][rigid_name]["root_velocity"][0],
                initial_state["rigid_object"][rigid_name]["root_velocity"][0],
                atol=1.0e-5,
            )
            assert torch.allclose(
                reset_state["articulation"][articulation_name]["joint_position"][0],
                initial_joint_position[0],
                atol=1.0e-5,
            )
            assert torch.allclose(reset_state["rigid_object"][rigid_name]["root_pose"][1], moved_pose[1])
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
