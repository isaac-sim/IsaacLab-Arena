# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise native heterogeneous spawning, geometry, and reset with local USD assets."""

import torch

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _write_rigid_asset(path, nested: bool, mass: float) -> None:
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    body = root
    if nested:
        group = UsdGeom.Xform.Define(stage, "/Asset/Group")
        group.AddTranslateOp().Set(Gf.Vec3d(0.4, -0.2, 0.3))
        body = UsdGeom.Xform.Define(stage, "/Asset/Group/Body").GetPrim()
        UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3d(0.1, 0.2, 0.3))
        UsdGeom.Xformable(body).AddRotateZOp().Set(90.0)
    UsdPhysics.RigidBodyAPI.Apply(body)
    PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateDisableGravityAttr(True)
    UsdPhysics.MassAPI.Apply(body).CreateMassAttr(mass)
    cube = UsdGeom.Cube.Define(stage, str(body.GetPath()) + "/Cube")
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(0.12, 0.24, 0.36))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.GetRootLayer().Save()


def _test_native_variant_scene(simulation_app, tmp_path):
    import warp as wp
    from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings, replace_strings_with_env_cfg_spaces
    from isaaclab.sim import UsdFileCfg
    from isaaclab.sim.utils.stage import get_current_stage
    from isaaclab.utils import replace_slices_with_strings, replace_strings_with_slices
    from omegaconf import OmegaConf
    from pxr import UsdPhysics

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.bounding_box_helpers import build_per_env_bounding_boxes
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    root_path = tmp_path / "root.usda"
    nested_path = tmp_path / "nested.usda"
    _write_rigid_asset(root_path, nested=False, mass=0.2)
    _write_rigid_asset(nested_path, nested=True, mass=0.5)

    def make_member(path, scale):
        return Object(
            name=path.stem,
            spawn_cfg=UsdFileCfg(usd_path=str(path), scale=(scale,) * 3, activate_contact_sensors=True),
            object_type=ObjectType.RIGID,
        )

    pickup = RigidObjectSet(
        name="pickup",
        objects=[make_member(root_path, 1.0), make_member(nested_path, 1.5)],
        initial_pose=Pose(position_xyz=(0.0, 0.0, 2.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
    )
    destination = RigidObjectSet(
        name="destination",
        objects=[make_member(nested_path, scale) for scale in (0.5, 1.0, 2.0)],
        random_choice=True,
        initial_pose=Pose(position_xyz=(1.0, 0.0, 2.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
    )
    assert pickup.get_contact_sensor_prim_path().endswith("/rigid_body")
    assert destination.get_contact_sensor_prim_path().endswith("/Group/Body")
    assert all(cfg.usd_path == str(nested_path) for cfg in destination.spawn_cfg.assets_cfg)

    def configure_contacts(env_cfg):
        env_cfg.scene.pickup_contacts = pickup.get_contact_sensor_cfg(destination)
        return env_cfg

    arena_environment = IsaacLabArenaEnvironment(
        name="test_native_variant_scene",
        scene=Scene(assets=[pickup, destination]),
        env_cfg_callback=configure_contacts,
    )
    num_envs = 9
    builder = ArenaEnvBuilder(
        arena_environment,
        ArenaEnvBuilderCfg(num_envs=num_envs, env_spacing=4.0, seed=19, disable_fabric=True),
    )
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    # Isaac Lab's training entry points serialize through OmegaConf before constructing the scene.
    env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)
    serialized = OmegaConf.create(replace_slices_with_strings(env_cfg.to_dict()))
    env_cfg.from_dict(replace_strings_with_slices(OmegaConf.to_container(serialized, resolve=True)))
    env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)
    env = builder.make_registered(env_cfg, env_kwargs)
    try:
        env.reset()
        with torch.inference_mode():
            env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
        runtime_scene = env.unwrapped.scene
        stage = get_current_stage()
        bounds_by_object = build_per_env_bounding_boxes([pickup, destination], num_envs).object_bboxes
        assignments = {obj.name: obj.asset_indices_by_env for obj in (pickup, destination)}
        expected_scales = {pickup: (1.0, 1.5), destination: (0.5, 1.0, 2.0)}
        expected_masses = {pickup: (0.2, 0.5), destination: (0.5, 0.5, 0.5)}
        for obj in (pickup, destination):
            runtime_asset = runtime_scene[obj.name]
            assert runtime_asset.num_instances == num_envs
            assert runtime_asset.num_bodies == 1
            masses = wp.to_torch(runtime_asset.data.body_mass).cpu().flatten()
            torch.testing.assert_close(
                masses,
                torch.tensor([expected_masses[obj][index] for index in assignments[obj.name]]),
            )
            assigned_scales = torch.tensor([expected_scales[obj][index] for index in assignments[obj.name]])
            expected_sizes = assigned_scales[:, None] * torch.tensor((0.12, 0.24, 0.36))
            torch.testing.assert_close(bounds_by_object[obj].size, expected_sizes)
            torch.testing.assert_close(bounds_by_object[obj].center, torch.zeros((num_envs, 3)), atol=1e-6, rtol=0)
            # Live inertias verify that PhysX cooked the same scaled geometry used by placement.
            squared_sizes = expected_sizes.square()
            expected_diagonal = masses[:, None] * (squared_sizes.sum(dim=1, keepdim=True) - squared_sizes) / 12
            inertias = wp.to_torch(runtime_asset.data.body_inertia).cpu().reshape(num_envs, 3, 3)
            torch.testing.assert_close(inertias, torch.diag_embed(expected_diagonal), atol=1e-6, rtol=1e-4)

            # Native clone planning must use exactly the choices used by placement geometry.
            plan = runtime_scene.clone_plan
            rows = plan.cfg_rows[id(getattr(runtime_scene.cfg, obj.name))]
            clone_mask = plan.clone_mask[list(rows)]
            assert tuple(clone_mask.long().argmax(dim=0).tolist()) == assignments[obj.name]
            assert torch.equal(clone_mask.sum(dim=0).cpu(), torch.ones(num_envs, dtype=torch.long))

            body_suffix = obj.get_contact_sensor_prim_path().removeprefix(obj.prim_path)
            for env_id in range(num_envs):
                body_path = f"/World/envs/env_{env_id}/{obj.name}{body_suffix}"
                body = stage.GetPrimAtPath(body_path)
                assert body.HasAPI(UsdPhysics.RigidBodyAPI)
        force_matrix = runtime_scene.sensors["pickup_contacts"].data.force_matrix_w
        assert force_matrix is not None and wp.to_torch(force_matrix).shape == (num_envs, 1, 1, 3)
        env.reset()
        assert assignments == {obj.name: obj.asset_indices_by_env for obj in (pickup, destination)}
        for obj in (pickup, destination):
            root_positions = wp.to_torch(runtime_scene[obj.name].data.root_pos_w) - runtime_scene.env_origins
            expected_positions = torch.tensor(obj.initial_pose.position_xyz, device=root_positions.device)
            torch.testing.assert_close(root_positions, expected_positions.expand(num_envs, 3), atol=1e-5, rtol=1e-5)
            root_orientations = wp.to_torch(runtime_scene[obj.name].data.root_quat_w)
            expected_orientations = torch.tensor((0.0, 0.0, 0.0, 1.0), device=root_orientations.device)
            torch.testing.assert_close(
                root_orientations, expected_orientations.expand(num_envs, 4), atol=1e-5, rtol=1e-5
            )
    finally:
        env.close()
    return True


def test_native_variant_scene(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_native_variant_scene, tmp_path=tmp_path)
