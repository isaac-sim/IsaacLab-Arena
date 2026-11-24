# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reset selection contracts with local primitives, including parallel Newton physics."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _make_builder(resolve_on_reset=False, replay_path=None, backend="newton", num_envs=4):
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.register import register_asset_factory
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relations import On
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tests.test_rigid_asset_selection_variation import make_table
    from isaaclab_arena.variations.rigid_asset_selection_variation import RigidAssetSelectionVariationCfg

    def register(name, size, mass):
        if AssetRegistry().is_registered(name):
            return

        @register_asset_factory(registry_name=name, object_type=ObjectType.RIGID)
        def make(name=name, **kwargs):
            return Object(
                name=name,
                object_type=ObjectType.RIGID,
                spawner_cfg=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                    collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                    mass_props=sim_utils.MassCfg(mass=mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
                ),
                **kwargs,
            )

    register("runtime_test_small", (0.04, 0.06, 0.08), 0.2)
    register("runtime_test_large", (0.10, 0.12, 0.16), 0.8)
    table = make_table()
    objects = []
    for name in ("pickup", "destination"):
        obj = AssetRegistry().get_asset_by_name("runtime_test_small")(name=name)
        obj.add_relation(On(table, clearance_m=0.01))
        obj.get_variation("asset_selection").apply_cfg(
            RigidAssetSelectionVariationCfg(
                enabled=True,
                mode="reset",
                registry_names=["runtime_test_small", "runtime_test_large"],
            )
        )
        objects.append(obj)
    arena = IsaacLabArenaEnvironment(
        name="runtime_asset_selection_test",
        scene=Scene(assets=[table, *objects]),
        placer_params=ObjectPlacerParams(min_unique_layouts_per_env=1, max_placement_attempts=3, placement_seed=3),
    )
    return ArenaEnvBuilder(
        arena,
        ArenaEnvBuilderCfg(
            num_envs=num_envs,
            presets=backend,
            resolve_on_reset=resolve_on_reset,
            recorded_variation_samples_path=replay_path,
        ),
    )


def _test_selection_and_placement(simulation_app):
    import torch

    from isaaclab_arena.assets.deformable_object_library import DeformableCubeNewton
    from isaaclab_arena.relations.runtime.placement_events import PLACEMENT_RESET_EVENT_NAME
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

    builder = _make_builder()
    cfg, kwargs = builder.build_env_cfg()
    selection = builder.runtime_asset_selections["pickup"]
    assert selection.selected_names == ["runtime_test_small", "runtime_test_large"] * 2
    assert cfg.scene.pickup.spawn.static is False
    assert len(cfg.scene.pickup.spawn.assets_cfg) == 2
    assets = builder.arena_env.scene.assets
    task = PickAndPlaceTask(assets["pickup"], assets["destination"], assets["table"])
    assert task.get_task_description() == "Pick up the object and place it in the destination."
    deformable_task = PickAndPlaceTask(DeformableCubeNewton(), assets["table"], assets["table"])
    assert deformable_task.get_task_description() == "Pick up the deformable_cube_newton, and place it into the table"
    sampler = getattr(cfg.events, PLACEMENT_RESET_EVENT_NAME).params["placement_sampler"].sampler
    # Supply a minimal recorder owner, as in a normal environment reset.
    # Recorder listeners are tested through the live env below; pure sampling must not call them without an env.
    for variation in builder.runtime_asset_selections.values():
        variation.sampler._listeners.clear()
    sampler._listeners.clear()
    ids = torch.arange(4)
    for variation in builder.runtime_asset_selections.values():
        variation.select(ids)
    samples = sampler.sample(4, ids)
    old_names = selection.selected_names.copy()
    old_layouts = [sample.poses for sample in samples]
    for variation in builder.runtime_asset_selections.values():
        variation.select(torch.tensor([3, 1]))
    assert selection.selected_names[0] == old_names[0]
    assert selection.selected_names[2] == old_names[2]
    assert selection.selected_names[3] == "runtime_test_small"
    changed = sampler.sample(2, torch.tensor([3, 1]))
    assert changed[0].selected_assets["pickup"]["registry_name"] == "runtime_test_small"
    assert changed[1].poses == old_layouts[1]
    # env 3 returns to its old geometry: fixed layouts are cached per geometry and absolute env ID.
    for variation in builder.runtime_asset_selections.values():
        variation.select(torch.tensor([1, 3]))
    restored = sampler.sample(2, torch.tensor([1, 3]))
    assert restored[1].poses == old_layouts[3]
    assert len(sampler._samplers) == 2
    return True


def _test_parallel_physics(simulation_app, solver="mjwarp"):
    import torch

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv

    builder = _make_builder()
    cfg, kwargs = builder.build_env_cfg()
    cfg.sim.physics.use_cuda_graph = False
    if solver == "xpbd":
        from isaaclab_newton.physics import XPBDSolverCfg

        cfg.sim.physics.solver_cfg = XPBDSolverCfg()
    else:
        cfg.sim.physics.solver_cfg.njmax = 128
        cfg.sim.physics.solver_cfg.nconmax = 128
    env = IsaacLabArenaManagerBasedRLEnv(cfg=cfg, **kwargs)
    try:
        env.reset()
        asset = env.scene["pickup"]
        torch.testing.assert_close(asset.mesh_variant_ids.torch.cpu(), torch.tensor([0, 1, 0, 1], dtype=torch.int32))
        torch.testing.assert_close(asset.data.body_mass.torch[:, 0].cpu(), torch.tensor([0.2, 0.8, 0.2, 0.8]))
        bounds = env.arena_world.get_aabb_in_local_frame("pickup")
        torch.testing.assert_close(bounds.size[:, 2].cpu(), torch.tensor([0.08, 0.16, 0.08, 0.16]))
        before = env.arena_world.get_pose_e("pickup").clone()
        env._reset_idx(torch.tensor([3, 1], device=env.device))
        torch.testing.assert_close(asset.mesh_variant_ids.torch.cpu(), torch.tensor([0, 0, 0, 1], dtype=torch.int32))
        torch.testing.assert_close(env.arena_world.get_pose_e("pickup")[[0, 2]], before[[0, 2]])
        bounds = env.arena_world.get_aabb_in_local_frame("pickup")
        assert bounds.size[1, 2].item() == pytest.approx(0.08)
        for _ in range(4):
            env.step(torch.zeros((4, 0), device=env.device))
        assert torch.isfinite(asset.data.root_pose_w.torch).all()
    finally:
        env.close()
    return True


def test_selection_and_placement():
    assert run_function_with_persistent_simulation_app(_test_selection_and_placement)


@pytest.mark.with_newton
@pytest.mark.parametrize("solver", ["mjwarp", "xpbd"])
def test_parallel_physics(solver):
    assert run_function_with_persistent_simulation_app(_test_parallel_physics, solver=solver)


def _test_recording_and_replay(simulation_app, tmp_path):
    import json
    import torch
    from unittest.mock import patch

    from isaaclab_arena.relations.runtime.placement_events import PLACEMENT_RESET_EVENT_NAME
    from isaaclab_arena.relations.runtime.placement_sampler import deserialize_placement_samples

    class RecorderOwner:
        def get_episode_index(self, env_id):
            return 0

    builder = _make_builder()
    cfg, kwargs = builder.build_env_cfg()
    owner = RecorderOwner()
    recorder = kwargs["variation_recorder"]
    recorder.bind_env(owner)
    ids = torch.arange(4)
    for selection in builder.runtime_asset_selections.values():
        selection.select(ids)
    sampler = getattr(cfg.events, PLACEMENT_RESET_EVENT_NAME).params["placement_sampler"].sampler
    samples = sampler.sample(4, ids)
    rows = []
    for env_id, sample in enumerate(samples):
        rows.append({
            "variations": {key: record.sample_for_episode(env_id, 0) for key, record in recorder.records.items()},
            "placement": sample.to_record(),
        })
    path = tmp_path / "episodes.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with patch(
        "isaaclab_arena.relations.pooled_object_placer.PooledObjectPlacer.__init__",
        side_effect=AssertionError("must not solve"),
    ):
        replay = _make_builder(replay_path=str(path))
        replay_cfg, replay_kwargs = replay.build_env_cfg()
    scheduler = replay_kwargs["variation_replay_scheduler"]
    replay_kwargs["variation_recorder"].bind_env(owner)
    # Assign source rows 0,1 to different, reordered environment slots.
    scheduler.assign_new_episodes([3, 1])
    for selection in replay.runtime_asset_selections.values():
        selection.select(torch.tensor([3, 1]))
    replay_sampler = getattr(replay_cfg.events, PLACEMENT_RESET_EVENT_NAME).params["placement_sampler"].sampler
    actual = replay_sampler.sample(2, torch.tensor([3, 1]))
    assert actual == samples[:2]
    assert deserialize_placement_samples([sample.to_record() for sample in actual]) == actual
    replay_sampler.recorded_samples[0].selected_assets["pickup"]["fingerprint"] = "different"
    with pytest.raises(AssertionError, match="geometry differs"):
        replay_sampler.sample(1, torch.tensor([3]))
    return True


def test_recording_and_replay(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_recording_and_replay, tmp_path=tmp_path)


def _test_configuration_and_random_sampling(simulation_app):
    import torch

    from isaaclab_arena.variations.rigid_asset_selection_variation import (
        AssetSelectionSamplerCfg,
        RigidAssetSelectionVariationCfg,
    )
    from isaaclab_arena.variations.runtime_rigid_asset_selection_variation import RuntimeRigidAssetSelectionVariation

    with pytest.raises(AssertionError, match="requires Newton"):
        _make_builder(backend="physx").build_env_cfg()
    builder = _make_builder()
    builder.build_env_cfg()
    target = builder.runtime_asset_selections["pickup"].object
    cfg = RigidAssetSelectionVariationCfg(
        enabled=True,
        mode="reset",
        registry_names=[
            "runtime_test_small",
            "runtime_test_large",
            "runtime_test_small",
        ],
        sampler_cfg=AssetSelectionSamplerCfg(random_choice=True),
    )

    def draw(num_envs, rounds):
        selection = RuntimeRigidAssetSelectionVariation(target, cfg, num_envs)
        selection.configure_at_build_time(13, "pickup.asset_selection")
        output = []
        for _ in range(rounds):
            output.extend(selection.select(torch.arange(num_envs)))
        return output

    assert draw(1, 16) == draw(4, 4)
    return True


def test_configuration_and_random_sampling():
    assert run_function_with_persistent_simulation_app(_test_configuration_and_random_sampling)


def _test_variant_geometry_uses_body_frame_and_spawn_scale(simulation_app, tmp_path):
    import numpy as np
    import torch

    from isaaclab.sim import UsdFileCfg
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.assets.spawn_cfg_geometry import SpawnCfgGeometry
    from isaaclab_arena.variations.runtime_rigid_asset_selection_variation import _body_bounds, spawn_fingerprint

    path = tmp_path / "offset_body.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    root.AddScaleOp().Set(Gf.Vec3f(2, 3, 4))
    body = UsdGeom.Xform.Define(stage, "/Asset/Body")
    body.AddTranslateOp().Set(Gf.Vec3d(0.2, 0.1, 0.3))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    variants = root.GetPrim().GetVariantSets().AddVariantSet("size")
    for name, half in (("small", 1), ("large", 2)):
        variants.AddVariant(name)
        variants.SetVariantSelection(name)
        with variants.GetVariantEditContext():
            mesh = UsdGeom.Mesh.Define(stage, "/Asset/Body/Mesh")
            points = [(x * half, y * half, z * half) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr([4] * 6)
            mesh.CreateFaceVertexIndicesAttr([0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3])
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()
    fingerprints = []
    for name, half in (("small", 1), ("large", 2)):
        cfg = UsdFileCfg(usd_path=str(path), scale=(0.5, 1.0, 1.5), variants={"size": name})
        geometry = SpawnCfgGeometry(cfg)
        bounds = _body_bounds(cfg)
        expected = torch.tensor([[0.5, 1.0, 1.5]]) * half
        torch.testing.assert_close(bounds.min_point, -expected)
        torch.testing.assert_close(bounds.max_point, expected)
        mesh = geometry.get_collision_mesh(body_frame=True)
        np.testing.assert_allclose(mesh.bounds, np.concatenate((-expected.numpy(), expected.numpy())), atol=1e-6)
        fingerprints.append(spawn_fingerprint(cfg))
    assert fingerprints[0] != fingerprints[1]
    return True


def test_variant_geometry_uses_body_frame_and_spawn_scale(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_variant_geometry_uses_body_frame_and_spawn_scale, tmp_path=tmp_path
    )


def _test_unsupported_rendering_is_rejected(simulation_app):
    from types import SimpleNamespace

    from isaaclab_arena.variations.runtime_rigid_asset_selection_variation import _validate_rendering

    env = SimpleNamespace(
        scene=SimpleNamespace(sensors={}), sim=SimpleNamespace(cfg=SimpleNamespace(visualizer_cfgs=[]))
    )
    _validate_rendering(env)
    env.scene.sensors["camera"] = SimpleNamespace(
        cfg=SimpleNamespace(renderer_cfg=SimpleNamespace(renderer_type="isaac_rtx"))
    )
    with pytest.raises(AssertionError, match="newton_warp"):
        _validate_rendering(env)
    env.scene.sensors.clear()
    env.sim.cfg.visualizer_cfgs = [SimpleNamespace(backend=[SimpleNamespace(visualizer_type="kit")])]
    with pytest.raises(AssertionError, match="Kit does not refresh"):
        _validate_rendering(env)
    return True


def test_unsupported_rendering_is_rejected():
    assert run_function_with_persistent_simulation_app(_test_unsupported_rendering_is_rejected)


def _test_lazy_pool_initialization_during_rollout(simulation_app):
    import torch
    from unittest.mock import patch

    from isaaclab_arena.environments import relation_solver_interface
    from isaaclab_arena.relations.runtime.placement_events import PLACEMENT_RESET_EVENT_NAME

    builder = _make_builder(num_envs=1)
    cfg, _ = builder.build_env_cfg()
    sampler = getattr(cfg.events, PLACEMENT_RESET_EVENT_NAME).params["placement_sampler"].sampler
    for selection in builder.runtime_asset_selections.values():
        selection.sampler._listeners.clear()
        selection.select(torch.tensor([0]))
    sampler._listeners.clear()
    sampler.sample(1, torch.tensor([0]))
    assert len(sampler._samplers) == 1
    prepare = relation_solver_interface.prepare_live_relation_placement
    modes = []

    def checked_prepare(*args, **kwargs):
        modes.append(torch.is_inference_mode_enabled())
        return prepare(*args, **kwargs)

    # A previously unseen combination needs a new IK solver during an inference-mode rollout.
    with (
        torch.inference_mode(),
        patch.object(relation_solver_interface, "prepare_live_relation_placement", checked_prepare),
    ):
        for selection in builder.runtime_asset_selections.values():
            selection.select(torch.tensor([0]))
        sampler.sample(1, torch.tensor([0]))
    assert modes == [False]
    assert len(sampler._samplers) == 2
    return True


def test_lazy_pool_initialization_during_rollout():
    assert run_function_with_persistent_simulation_app(_test_lazy_pool_initialization_during_rollout)


def _test_camera_follows_selected_geometry(simulation_app):
    import torch

    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import PinholeCameraCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv

    builder = _make_builder()
    cfg, kwargs = builder.build_env_cfg()
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.sim.physics.use_cuda_graph = False
    cfg.sim.physics.solver_cfg.njmax = 128
    cfg.sim.physics.solver_cfg.nconmax = 128
    cfg.scene.selection_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/SelectionCamera",
        width=64,
        height=64,
        data_types=["rgb", "distance_to_image_plane"],
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 1.0), convention="opengl"),
        spawn=PinholeCameraCfg(clipping_range=(0.01, 5.0)),
        renderer_cfg=NewtonWarpRendererCfg(),
    )
    env = IsaacLabArenaManagerBasedRLEnv(cfg=cfg, **kwargs)
    try:
        env.reset()
        camera = env.scene["selection_camera"]

        def capture():
            # Hold poses constant so image changes measure geometry selection, not new placement samples.
            for name, x in (("pickup", 0.0), ("destination", 1.0)):
                T_W_O = torch.zeros((4, 7), device=env.device)
                T_W_O[:, :3] = env.scene.env_origins + torch.tensor([x, 0.0, 0.4], device=env.device)
                T_W_O[:, 6] = 1.0
                env.scene[name].write_root_pose_to_sim(T_W_O)
                env.scene[name].write_root_velocity_to_sim(torch.zeros((4, 6), device=env.device))
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
            camera.update(env.physics_dt, force_recompute=True)
            return (
                camera.data.output["distance_to_image_plane"].torch.clone(),
                camera.data.output["rgb"].torch.clone(),
            )

        before_depth, before_rgb = capture()
        torch.testing.assert_close(
            before_depth[:, 32, 32, 0].cpu(), torch.tensor([0.56, 0.52, 0.56, 0.52]), atol=1e-3, rtol=0.0
        )
        # The environment reset sorts [3, 1]: env 1 changes to small, env 3 retains large.
        env._reset_idx(torch.tensor([3, 1], device=env.device))
        after_depth, after_rgb = capture()
        torch.testing.assert_close(
            after_depth[:, 32, 32, 0].cpu(), torch.tensor([0.56, 0.56, 0.56, 0.52]), atol=1e-3, rtol=0.0
        )
        torch.testing.assert_close(after_rgb[[0, 2, 3]], before_rgb[[0, 2, 3]])
        assert torch.any(after_rgb[1] != before_rgb[1]), "RGB still shows the previous variant"
        pixels = ((after_depth > 0.0) & (after_depth < 0.7)).sum(dim=(1, 2, 3))
        assert pixels[0] > 0
        assert pixels[3] > 2 * pixels[1], "Large variant must have a larger rendered footprint"
    finally:
        env.close()
    return True


@pytest.mark.with_cameras
def test_camera_follows_selected_geometry():
    assert run_function_with_persistent_simulation_app(_test_camera_follows_selected_geometry, enable_cameras=True)
