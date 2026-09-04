# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration and backend smoke tests for deformables."""

import importlib.util

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True
PYTETWILD_AVAILABLE = importlib.util.find_spec("pytetwild") is not None


def _make_soft_cube(physics_preset, initial_pose=None):
    import isaaclab.sim as sim_utils
    from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
    from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxDeformableBodyPropertiesCfg
    from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg

    from isaaclab_arena.assets.deformable_object import DeformableObject

    if physics_preset == "physx":
        deformable_props = PhysxDeformableBodyPropertiesCfg()
        collision_props = [PhysxCollisionCfg(rest_offset=0.0025, contact_offset=0.01)]
        physics_material = PhysxDeformableBodyMaterialCfg(
            youngs_modulus=8.0e4,
            poissons_ratio=0.4,
            density=300.0,
        )
    else:
        deformable_props = NewtonDeformableBodyPropertiesCfg()
        collision_props = None
        physics_material = NewtonDeformableBodyMaterialCfg(
            k_mu=8.0e4 / (2.0 * (1.0 + 0.4)),
            k_lambda=8.0e4 * 0.4 / ((1.0 + 0.4) * (1.0 - 2.0 * 0.4)),
            density=300.0,
            particle_radius=0.01,
        )
    return DeformableObject(
        name="soft_cube",
        spawner_cfg=sim_utils.MeshCuboidCfg(
            size=(0.1, 0.1, 0.1),
            deformable_props=deformable_props,
            collision_props=collision_props,
            physics_material=physics_material,
        ),
        initial_pose=initial_pose,
    )


def _test_backend_specific_deformable_config(simulation_app) -> bool:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import DeformableObjectCfg
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
    from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg
    from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
    from isaaclab_physx.sim.spawners.materials import (
        PhysxDeformableBodyMaterialCfg,
        PhysxSurfaceDeformableBodyMaterialCfg,
    )

    from isaaclab_arena.assets.deformable_object import DeformableObject
    from isaaclab_arena.assets.deformable_object_library import DeformableCube, DeformableSurface, DeformableTeddyBear
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.relations.relations import IsAnchor
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose, PoseRange

    pose = Pose(position_xyz=(0.1, -0.2, 0.3), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    soft_cube = _make_soft_cube("physx", initial_pose=pose)
    cfg_name, physx_cfg = soft_cube.get_object_cfg()

    assert cfg_name == soft_cube.name
    assert isinstance(physx_cfg, DeformableObjectCfg)
    assert isinstance(physx_cfg.spawn.deformable_props, PhysxDeformableBodyPropertiesCfg)
    assert isinstance(physx_cfg.spawn.physics_material, PhysxDeformableBodyMaterialCfg)
    assert physx_cfg.spawn.physics_material.youngs_modulus == pytest.approx(8.0e4)
    assert physx_cfg.spawn.physics_material.poissons_ratio == pytest.approx(0.4)
    assert physx_cfg.spawn.physics_material.density == pytest.approx(300.0)
    assert physx_cfg.init_state.pos == pose.position_xyz
    assert physx_cfg.init_state.rot == pose.rotation_xyzw
    assert soft_cube.object_type is ObjectType.DEFORMABLE
    assert soft_cube.get_object_cfg()[1] is physx_cfg
    assert soft_cube.get_event_cfg()[1] is not None
    assert not hasattr(soft_cube, "reset_pose")
    assert not hasattr(soft_cube, "disable_reset_pose")
    assert not hasattr(soft_cube, "enable_reset_pose")

    newton_cube = _make_soft_cube("newton", initial_pose=pose)
    _, newton_cfg = newton_cube.get_object_cfg()
    assert isinstance(newton_cfg.spawn.deformable_props, NewtonDeformableBodyPropertiesCfg)
    assert isinstance(newton_cfg.spawn.physics_material, NewtonDeformableBodyMaterialCfg)
    expected_mu = 8.0e4 / (2.0 * (1.0 + 0.4))
    expected_lambda = 8.0e4 * 0.4 / ((1.0 + 0.4) * (1.0 - 2.0 * 0.4))
    assert newton_cfg.spawn.physics_material.k_mu == pytest.approx(expected_mu)
    assert newton_cfg.spawn.physics_material.k_lambda == pytest.approx(expected_lambda)
    assert newton_cfg.spawn.physics_material.density == pytest.approx(300.0)
    assert newton_cube.get_object_cfg()[1] is newton_cfg

    with pytest.raises(AssertionError, match="backend-specific deformable_props"):
        DeformableObject(
            name="authored_soft_body",
            spawner_cfg=UsdFileCfg(usd_path="/tmp/authored_soft_body.usd"),
        )

    for asset_type, asset_name, material_type in (
        (DeformableCube, "deformable_cube", PhysxDeformableBodyMaterialCfg),
        (DeformableSurface, "deformable_surface", PhysxSurfaceDeformableBodyMaterialCfg),
        (DeformableTeddyBear, "deformable_teddy_bear", PhysxDeformableBodyMaterialCfg),
    ):
        library_object = asset_type()
        assert AssetRegistry().get_asset_by_name(asset_name) is asset_type
        assert library_object.physics_preset == "physx"
        assert isinstance(library_object.spawner_cfg.deformable_props, PhysxDeformableBodyPropertiesCfg)
        assert isinstance(library_object.spawner_cfg.physics_material, material_type)

    library_cube = DeformableCube()
    assert library_cube.spawner_cfg.size == (0.15, 0.04, 0.04)
    assert library_cube.spawner_cfg.collision_props[0].rest_offset == 0.0
    assert library_cube.spawner_cfg.collision_props[0].contact_offset == pytest.approx(0.0025)
    assert library_cube.spawner_cfg.deformable_props.linear_damping == 0.0
    assert library_cube.spawner_cfg.physics_material.youngs_modulus == pytest.approx(8.0e4)
    assert library_cube.spawner_cfg.physics_material.poissons_ratio == pytest.approx(0.25)
    assert library_cube.spawner_cfg.physics_material.density == pytest.approx(300.0)
    assert library_cube.spawner_cfg.visual_material.diffuse_color == (0.95, 0.85, 0.1)
    assert library_cube.get_bounding_box().size[0].tolist() == pytest.approx([0.15, 0.04, 0.04])

    library_surface = DeformableSurface()
    assert library_surface.spawner_cfg.size == (0.2, 0.2)
    assert library_surface.spawner_cfg.resolution == (30, 30)
    assert library_surface.spawner_cfg.visual_material.diffuse_color == (0.95, 0.85, 0.1)
    assert library_surface.get_bounding_box().min_point[0].tolist() == pytest.approx([-0.1, -0.1, 0.0])
    assert library_surface.get_bounding_box().max_point[0].tolist() == pytest.approx([0.1, 0.1, 0.0])

    library_teddy_bear = DeformableTeddyBear()
    assert library_teddy_bear._bounding_box is None
    teddy_bear_bbox = library_teddy_bear.get_bounding_box()
    assert library_teddy_bear._bounding_box is teddy_bear_bbox
    assert (teddy_bear_bbox.size[0] > 0.0).all()
    assert (teddy_bear_bbox.size[0] < 1.0).all()

    updated_pose = Pose(position_xyz=(0.4, 0.0, 0.5))
    soft_cube.set_initial_pose(updated_pose)
    assert soft_cube.get_object_cfg()[1] is physx_cfg
    assert physx_cfg.init_state.pos == updated_pose.position_xyz
    with pytest.raises(AssertionError, match="fixed Pose or PosePerEnv"):
        soft_cube.set_initial_pose(PoseRange())

    soft_cube.add_relation(IsAnchor())
    scene = Scene(assets=[soft_cube])
    assert scene.get_objects_with_relations() == [soft_cube]

    rigid = Object(
        name="rigid",
        object_type=ObjectType.RIGID,
        spawner_cfg=sim_utils.MeshCuboidCfg(size=(0.1, 0.1, 0.1)),
    )
    with pytest.raises(NotImplementedError, match="does not support contact sensors"):
        soft_cube.get_contact_sensor_cfg(rigid)
    with pytest.raises(AssertionError, match="against deformable objects"):
        rigid.get_contact_sensor_cfg(soft_cube)
    return True


def _test_builder_validates_deformable_preset(simulation_app) -> bool:
    from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
    from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    for preset, properties_type in (
        ("physx", PhysxDeformableBodyPropertiesCfg),
        ("newton", NewtonDeformableBodyPropertiesCfg),
    ):
        soft_cube = _make_soft_cube(preset)
        arena_env = IsaacLabArenaEnvironment(
            name=f"{preset}_deformable_config",
            scene=Scene(assets=[soft_cube]),
        )
        builder = ArenaEnvBuilder(
            arena_env,
            ArenaEnvBuilderCfg(num_envs=1, presets=preset, solve_relations=False),
        )
        env_cfg, _ = builder.compose_manager_cfg()
        assert isinstance(env_cfg.scene.soft_cube.spawn.deformable_props, properties_type)

    mismatched_builder = ArenaEnvBuilder(
        IsaacLabArenaEnvironment(
            name="mismatched_deformable_config",
            scene=Scene(assets=[_make_soft_cube("physx")]),
        ),
        ArenaEnvBuilderCfg(num_envs=1, presets="newton", solve_relations=False),
    )
    with pytest.raises(ValueError, match="configured for 'physx'.*selected preset 'newton'"):
        mismatched_builder.compose_manager_cfg()
    return True


def _test_deformable_nodal_reset_terms(simulation_app) -> bool:
    import torch
    from types import SimpleNamespace

    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.utils import math as math_utils

    from isaaclab_arena.terms.events import set_deformable_object_pose, set_deformable_object_pose_per_env
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.utils.velocity import Velocity

    class FakeAsset:
        def __init__(self, usd_backed: bool = False):
            default_state = torch.zeros((2, 2, 6))
            default_state[0, :, :3] = torch.tensor([[-0.05, 0.0, 0.5], [0.05, 0.0, 0.5]])
            default_state[1, :, :3] = torch.tensor([[9.95, 0.0, 0.5], [10.05, 0.0, 0.5]])
            self.data = SimpleNamespace(default_nodal_state_w=SimpleNamespace(torch=default_state))
            spawn = UsdFileCfg(usd_path="/tmp/fake.usd") if usd_backed else SimpleNamespace()
            self.cfg = SimpleNamespace(
                init_state=SimpleNamespace(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
                spawn=spawn,
            )
            self.written_state = default_state.clone()
            self.reset_env_ids = None

        def write_nodal_state_to_sim_index(self, nodal_state, env_ids):
            self.written_state[env_ids] = nodal_state

        def transform_nodal_pos(self, nodal_pos, pos, quat):
            mean_nodal_pos = nodal_pos.mean(dim=1, keepdim=True)
            return math_utils.transform_points(nodal_pos - mean_nodal_pos, pos, quat) + mean_nodal_pos

        def reset(self, env_ids):
            self.reset_env_ids = env_ids.clone()

    class FakeScene(dict):
        def __init__(self, asset):
            super().__init__(soft_cube=asset)
            self.env_origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    asset = FakeAsset()
    env = SimpleNamespace(scene=FakeScene(asset), device=torch.device("cpu"))
    asset_cfg = SimpleNamespace(name="soft_cube")
    env_ids = torch.tensor([0, 1])

    fixed_pose = Pose(position_xyz=(1.0, 2.0, 3.0))
    set_deformable_object_pose(
        env,
        env_ids,
        asset_cfg,
        fixed_pose,
        Velocity(linear_xyz=(0.1, 0.2, 0.3)),
    )
    torch.testing.assert_close(
        asset.written_state[..., :3].mean(dim=1),
        torch.tensor([[1.0, 2.0, 3.0], [11.0, 2.0, 3.0]]),
    )
    torch.testing.assert_close(asset.written_state[..., 3:], torch.tensor([0.1, 0.2, 0.3]).expand(2, 2, 3))

    poses = [Pose(position_xyz=(0.0, 1.0, 2.0)), Pose(position_xyz=(3.0, 4.0, 5.0))]
    set_deformable_object_pose_per_env(env, env_ids, asset_cfg, poses)
    torch.testing.assert_close(
        asset.written_state[..., :3].mean(dim=1),
        torch.tensor([[0.0, 1.0, 2.0], [13.0, 4.0, 5.0]]),
    )
    torch.testing.assert_close(asset.written_state[..., 3:], torch.zeros((2, 2, 3)))
    torch.testing.assert_close(asset.reset_env_ids, env_ids)

    usd_asset = FakeAsset(usd_backed=True)
    usd_env = SimpleNamespace(scene=FakeScene(usd_asset), device=torch.device("cpu"))
    set_deformable_object_pose(usd_env, env_ids, asset_cfg, fixed_pose)
    torch.testing.assert_close(
        usd_asset.written_state[..., :3].mean(dim=1),
        torch.tensor([[1.0, 2.0, 3.5], [11.0, 2.0, 3.5]]),
    )
    return True


def _test_deformable_reset_and_initial_pose(simulation_app) -> bool:
    import torch

    from isaaclab.assets import DeformableObjectCfg

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose, PosePerEnv

    poses = PosePerEnv(
        poses=[
            Pose(position_xyz=(0.0, 0.0, 0.5), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
            Pose(position_xyz=(0.2, 0.0, 0.6), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        ]
    )
    physics_preset = "physx"
    soft_cube = AssetRegistry().get_asset_by_name("deformable_cube")(instance_name="soft_cube")
    soft_cube.set_initial_pose(poses)
    arena_env = IsaacLabArenaEnvironment(
        name=f"{physics_preset}_deformable_reset_and_initial_pose",
        scene=Scene(assets=[soft_cube]),
    )
    builder = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(num_envs=2, presets=physics_preset, solve_relations=False),
    )
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    assert isinstance(env_cfg.scene.soft_cube, DeformableObjectCfg)
    env = builder.make_registered(env_cfg, env_kwargs)
    env.reset()

    try:
        deformable_asset = env.unwrapped.scene[soft_cube.name]
        initial_nodal_state = deformable_asset.data.nodal_state_w.torch.clone()
        assert torch.isfinite(initial_nodal_state).all()
        expected = torch.tensor([pose.position_xyz for pose in poses.poses], device=env.unwrapped.device)
        aggregate_positions = deformable_asset.data.root_pos_w.torch - env.unwrapped.scene.env_origins
        torch.testing.assert_close(aggregate_positions, expected, atol=2.0e-3, rtol=0.0)

        displaced = initial_nodal_state.clone()
        displaced[..., 0] += 0.25
        displaced[..., 1] += torch.linspace(
            0.0,
            0.05,
            displaced.shape[1],
            device=displaced.device,
        )
        displaced[..., 3:] = 0.5
        deformable_asset.write_nodal_state_to_sim_index(displaced)
        env.reset()
        restored_nodal_state = deformable_asset.data.nodal_state_w.torch.clone()
        torch.testing.assert_close(restored_nodal_state, initial_nodal_state)
    finally:
        env.close()
    return True


def test_backend_specific_deformable_config():
    assert run_function_with_persistent_simulation_app(_test_backend_specific_deformable_config, headless=HEADLESS)


def test_builder_validates_deformable_preset():
    assert run_function_with_persistent_simulation_app(_test_builder_validates_deformable_preset, headless=HEADLESS)


def test_deformable_nodal_reset_terms():
    assert run_function_with_persistent_simulation_app(_test_deformable_nodal_reset_terms, headless=HEADLESS)


@pytest.mark.skipif(not PYTETWILD_AVAILABLE, reason="requires Isaac Lab's optional tetrahedralization dependencies")
def test_deformable_reset_and_initial_pose():
    assert run_function_with_persistent_simulation_app(
        _test_deformable_reset_and_initial_pose,
        headless=HEADLESS,
    )
