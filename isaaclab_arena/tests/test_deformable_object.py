# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Focused configuration and PhysX smoke tests for volume deformables."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True


def _make_soft_cube(initial_pose=None):
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.deformable_object import DeformableObject
    from isaaclab_arena.assets.deformable_spawn import VolumeDeformableMaterial

    return DeformableObject(
        name="soft_cube",
        spawner_cfg=sim_utils.MeshCuboidCfg(size=(0.1, 0.1, 0.1)),
        material=VolumeDeformableMaterial(
            youngs_modulus=8.0e4,
            poissons_ratio=0.4,
            density=300.0,
            particle_radius=0.01,
        ),
        initial_pose=initial_pose,
    )


def _test_volume_deformable_config(simulation_app) -> bool:
    from isaaclab.assets import DeformableObjectCfg
    from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
    from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg
    from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
    from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg

    from isaaclab_arena.assets.deformable_spawn import lame_parameters
    from isaaclab_arena.utils.pose import Pose

    pose = Pose(position_xyz=(0.1, -0.2, 0.3), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    soft_cube = _make_soft_cube(initial_pose=pose)
    cfg = soft_cube.resolve_object_cfg()

    assert isinstance(cfg, DeformableObjectCfg)
    assert isinstance(cfg.spawn.deformable_props, PhysxDeformableBodyPropertiesCfg)
    assert isinstance(cfg.spawn.physics_material, PhysxDeformableBodyMaterialCfg)
    assert cfg.spawn.physics_material.youngs_modulus == pytest.approx(8.0e4)
    assert cfg.spawn.physics_material.poissons_ratio == pytest.approx(0.4)
    assert cfg.spawn.physics_material.density == pytest.approx(300.0)
    assert cfg.init_state.pos == pose.position_xyz
    assert cfg.init_state.rot == pose.rotation_xyzw
    assert soft_cube.object_type.value == "deformable"

    cfg = soft_cube.resolve_object_cfg("newton_mjwarp_vbd")
    assert isinstance(cfg.spawn.deformable_props, NewtonDeformableBodyPropertiesCfg)
    assert isinstance(cfg.spawn.physics_material, NewtonDeformableBodyMaterialCfg)
    expected_mu, expected_lambda = lame_parameters(8.0e4, 0.4)
    assert cfg.spawn.physics_material.k_mu == pytest.approx(expected_mu)
    assert cfg.spawn.physics_material.k_lambda == pytest.approx(expected_lambda)
    assert cfg.spawn.physics_material.density == pytest.approx(300.0)
    with pytest.raises(ValueError, match="does not support volume deformables"):
        soft_cube.resolve_object_cfg("newton")
    return True


def _test_physx_deformable_smoke(simulation_app) -> bool:
    import torch

    from isaaclab.assets import DeformableObjectCfg

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose, PosePerEnv

    poses = PosePerEnv(
        poses=[
            Pose(position_xyz=(0.0, 0.0, 0.5), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
            Pose(position_xyz=(0.2, 0.0, 0.6), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        ]
    )
    soft_cube = _make_soft_cube(initial_pose=poses)
    arena_env = IsaacLabArenaEnvironment(
        name="physx_deformable_smoke",
        scene=Scene(assets=[soft_cube]),
    )
    args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "2"])
    builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args))
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    assert isinstance(env_cfg.scene.soft_cube, DeformableObjectCfg)
    env = builder.make_registered(env_cfg, env_kwargs)
    env.reset()

    try:
        asset = env.unwrapped.scene[soft_cube.name]
        nodal_state = asset.data.nodal_state_w.torch.clone()
        assert torch.isfinite(nodal_state).all()
        expected = torch.tensor([pose.position_xyz for pose in poses.poses], device=env.unwrapped.device)
        centroids = nodal_state[..., :3].mean(dim=1) - env.unwrapped.scene.env_origins
        torch.testing.assert_close(centroids, expected, atol=2.0e-3, rtol=0.0)

        displaced = nodal_state.clone()
        displaced[..., 0] += 0.25
        asset.write_nodal_state_to_sim_index(displaced)
        env.reset()
        restored = asset.data.nodal_state_w.torch
        restored_centroids = restored[..., :3].mean(dim=1) - env.unwrapped.scene.env_origins
        torch.testing.assert_close(restored_centroids, expected, atol=2.0e-3, rtol=0.0)
    finally:
        env.close()
    return True


def test_volume_deformable_config():
    assert run_function_with_persistent_simulation_app(_test_volume_deformable_config, headless=HEADLESS)


def test_physx_deformable_smoke():
    assert run_function_with_persistent_simulation_app(_test_physx_deformable_smoke, headless=HEADLESS)
