# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise asset selection with local USD geometry, cloning, contacts, and resets."""

import torch

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _write_rigid_asset(path, nested, mass):
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    body = UsdGeom.Xform.Define(stage, "/Asset/Body").GetPrim() if nested else root
    if nested:
        UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3d(0.4, 0.2, 0.1))
    UsdPhysics.RigidBodyAPI.Apply(body)
    PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateDisableGravityAttr(True)
    UsdPhysics.MassAPI.Apply(body).CreateMassAttr(mass)
    cube = UsdGeom.Cube.Define(stage, str(body.GetPath()) + "/Cube")
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(0.1, 0.2, 0.3))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.GetRootLayer().Save()


def _make_usd_definition(name, path, scale=1.0):
    from isaaclab.sim import UsdFileCfg

    return name, UsdFileCfg(usd_path=str(path), scale=(scale,) * 3, activate_contact_sensors=True)


def _make_usd_object(name, path, scale=1.0, **kwargs):
    from isaaclab_arena.assets.object import Object

    return Object(name=name, asset=_make_usd_definition(name, path, scale), **kwargs)


def _test_selected_assets_match_native_physics_and_remain_fixed_across_resets(simulation_app, tmp_path):
    import warp as wp
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdPhysics

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation, AssetSelectionVariationCfg

    small_path = tmp_path / "small.usda"
    large_path = tmp_path / "large.usda"
    _write_rigid_asset(small_path, nested=False, mass=0.2)
    _write_rigid_asset(large_path, nested=True, mass=0.5)
    initial_pose = Pose(position_xyz=(0.0, 0.0, 2.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    pickup = Object(name="pick_up_object", initial_pose=initial_pose)
    variation = AssetSelectionVariation(
        asset_candidates=[
            _make_usd_definition("small", small_path),
            _make_usd_definition("large", large_path, scale=2.0),
        ],
        cfg=AssetSelectionVariationCfg(enabled=True),
    )
    pickup.add_variation(variation)

    def configure_contacts(env_cfg):
        env_cfg.scene.pickup_contacts = pickup.get_contact_sensor_cfg()
        return env_cfg

    arena_environment = IsaacLabArenaEnvironment(
        name="test_asset_selection_scene",
        scene=Scene(assets=[pickup]),
        env_cfg_callback=configure_contacts,
    )
    builder = ArenaEnvBuilder(
        arena_environment,
        ArenaEnvBuilderCfg(num_envs=4, env_spacing=4.0, seed=19, disable_fabric=True),
    )
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    assignments = pickup.asset_indices_by_env
    assert assignments == (0, 1, 0, 1)
    assert pickup.get_contact_sensor_prim_path().endswith("/rigid_body")
    expected_sizes = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]] * 2)
    torch.testing.assert_close(pickup.get_bounding_box_per_env(4).size, expected_sizes)
    record = env_kwargs["variation_recorder"][f"{pickup.name}.{variation.name}"]

    env = builder.make_registered(env_cfg, env_kwargs)
    try:
        runtime_scene = env.unwrapped.scene
        for _ in range(2):
            env.reset()
            with torch.inference_mode():
                env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
            assert pickup.asset_indices_by_env == assignments
            assert [record.sample_for_episode(env_id, 6) for env_id in range(4)] == ["small", "large"] * 2
            runtime_asset = runtime_scene[pickup.name]
            masses = wp.to_torch(runtime_asset.data.body_mass).cpu().flatten()
            torch.testing.assert_close(masses, torch.tensor([0.2, 0.5] * 2))
            squared_sizes = expected_sizes.square()
            expected_diagonal = masses[:, None] * (squared_sizes.sum(dim=1, keepdim=True) - squared_sizes) / 12
            inertias = wp.to_torch(runtime_asset.data.body_inertia).cpu().reshape(4, 3, 3)
            torch.testing.assert_close(inertias, torch.diag_embed(expected_diagonal), atol=1e-6, rtol=1e-4)
            positions = wp.to_torch(runtime_asset.data.root_pos_w) - runtime_scene.env_origins
            torch.testing.assert_close(
                positions,
                torch.tensor(initial_pose.position_xyz, device=positions.device).expand(4, 3),
                atol=1e-5,
                rtol=1e-5,
            )

        plan = runtime_scene.clone_plan
        rows = plan.cfg_rows[id(runtime_scene.cfg.pick_up_object)]
        assert tuple(plan.clone_mask[list(rows)].long().argmax(dim=0).tolist()) == assignments
        stage = get_current_stage()
        for env_id in range(4):
            body_path = f"/World/envs/env_{env_id}/pick_up_object/rigid_body"
            assert stage.GetPrimAtPath(body_path).HasAPI(UsdPhysics.RigidBodyAPI)
        sensor_forces = wp.to_torch(runtime_scene.sensors["pickup_contacts"].data.net_forces_w)
        assert sensor_forces.shape == (4, 1, 3)
    finally:
        env.close()
    return True


def test_selected_assets_match_native_physics_and_remain_fixed_across_resets(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_selected_assets_match_native_physics_and_remain_fixed_across_resets, tmp_path=tmp_path
    )


def _test_reference_to_selected_parent_is_rejected_after_hydra_enable(simulation_app, tmp_path):
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation

    asset_path = tmp_path / "parent.usda"
    _write_rigid_asset(asset_path, nested=True, mass=0.2)
    parent = _make_usd_object("parent", asset_path)
    reference = ObjectReference(
        name="body_reference",
        parent_asset=parent,
        object_type=ObjectType.RIGID,
        prim_path="{ENV_REGEX_NS}/parent/Body",
    )
    variation = AssetSelectionVariation(asset_candidates=[_make_usd_definition("candidate", asset_path)])
    parent.add_variation(variation)
    arena_environment = IsaacLabArenaEnvironment(
        name="test_reference_asset_selection", scene=Scene(assets=[parent, reference])
    )
    builder = ArenaEnvBuilder(
        arena_environment,
        ArenaEnvBuilderCfg(num_envs=2, solve_relations=False),
        hydra_overrides=[f"parent.{variation.name}.enabled=true"],
    )
    with pytest.raises(AssertionError, match="(?i)reference"):
        builder.compose_manager_cfg()
    return True


def test_reference_to_selected_parent_is_rejected_after_hydra_enable(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_reference_to_selected_parent_is_rejected_after_hydra_enable, tmp_path=tmp_path
    )


def _test_selected_geometry_controls_relation_placement_and_supports_runtime_mass(simulation_app):
    import warp as wp
    from isaaclab.sim import CollisionPropertiesCfg, CuboidCfg, MassPropertiesCfg, RigidBodyPropertiesCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose, PosePerEnv
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation, AssetSelectionVariationCfg
    from isaaclab_arena.variations.object_mass_variation import ObjectMassVariationCfg
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

    support = Object(
        name="support",
        object_type=ObjectType.BASE,
        spawn_cfg=CuboidCfg(size=(2.0, 2.0, 0.2), collision_props=CollisionPropertiesCfg()),
        initial_pose=Pose(position_xyz=(0.0, 0.0, 0.5)),
        relations=[IsAnchor()],
    )

    def make_pickup_definition(name, height):
        return name, CuboidCfg(
            size=(0.1, 0.1, height),
            mass_props=MassPropertiesCfg(mass=0.2),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=CollisionPropertiesCfg(),
        )

    pickup = Object(name="pick_up_object")
    pickup.add_relation(On(support, clearance_m=0.01))
    selection = AssetSelectionVariation(
        asset_candidates=[make_pickup_definition("short", 0.1), make_pickup_definition("tall", 0.4)],
        cfg=AssetSelectionVariationCfg(enabled=True),
    )
    pickup.add_variation(selection)
    pickup.get_variation("mass").apply_cfg(
        ObjectMassVariationCfg(enabled=True, sampler_cfg=UniformSamplerCfg(low=[0.25], high=[0.75]))
    )
    placer_params = ObjectPlacerParams(
        placement_seed=7,
        resolve_on_reset=False,
        min_unique_layouts_per_env=1,
        allow_best_loss_fallbacks=False,
        solver_params=RelationSolverParams(verbose=False, save_position_history=False),
    )
    builder = ArenaEnvBuilder(
        IsaacLabArenaEnvironment(
            name="test_asset_selection_placement",
            scene=Scene(assets=[support, pickup]),
            placer_params=placer_params,
        ),
        ArenaEnvBuilderCfg(num_envs=4, env_spacing=4.0, seed=19, disable_fabric=True),
    )
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    assigned_bounds = pickup.get_bounding_box_per_env(4)
    initial_poses = pickup.get_initial_pose()
    assert isinstance(initial_poses, PosePerEnv)
    solved_positions = torch.tensor([pose.position_xyz for pose in initial_poses.poses])
    expected_heights = torch.tensor([0.1, 0.4, 0.1, 0.4])
    torch.testing.assert_close(assigned_bounds.size[:, 2], expected_heights)
    support_top = support.initial_pose.position_xyz[2] + support.get_bounding_box().max_point[0, 2]
    object_bottoms = solved_positions[:, 2] + assigned_bounds.min_point[:, 2]
    tolerance = placer_params.on_relation_z_tolerance_m + 1e-5
    assert torch.all(object_bottoms >= support_top - tolerance)
    assert torch.all(object_bottoms <= support_top + 0.01 + tolerance)

    env = builder.make_registered(env_cfg, env_kwargs)
    try:
        recorder = env.unwrapped.variation_recorder
        selection_record = recorder[f"{pickup.name}.{selection.name}"]
        mass_record = recorder[f"{pickup.name}.mass"]
        masses_by_reset = []
        for reset_index in range(2):
            env.reset()
            runtime_asset = env.unwrapped.scene[pickup.name]
            live_positions = wp.to_torch(runtime_asset.data.root_pos_w) - env.unwrapped.scene.env_origins
            torch.testing.assert_close(live_positions.cpu(), solved_positions, atol=1e-4, rtol=0)
            masses = wp.to_torch(runtime_asset.data.body_mass).cpu().flatten().clone()
            masses_by_reset.append(masses)
            for env_id, candidate_name in enumerate(["short", "tall"] * 2):
                episode_index = env.unwrapped.get_episode_index(env_id)
                assert episode_index == reset_index
                assert selection_record.sample_for_episode(env_id, episode_index) == candidate_name
                recorded_mass = mass_record.sample_for_episode(env_id, episode_index)
                assert recorded_mass is not None
                assert 0.25 <= recorded_mass.item() <= 0.75
                torch.testing.assert_close(masses[env_id], recorded_mass.reshape(()))
            assert pickup.asset_indices_by_env == (0, 1, 0, 1)
        assert not torch.equal(masses_by_reset[0], masses_by_reset[1])
    finally:
        env.close()
    return True


def test_selected_geometry_controls_relation_placement_and_supports_runtime_mass():
    assert run_function_with_persistent_simulation_app(
        _test_selected_geometry_controls_relation_placement_and_supports_runtime_mass
    )
