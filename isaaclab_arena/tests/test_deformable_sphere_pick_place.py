# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for deformable object support and the shared pick-and-place config."""

import types

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def test_deformable_assets_registered() -> None:
    from isaaclab_arena.assets.registries import AssetRegistry

    reg = AssetRegistry()
    for asset_name in (
        "procedural_deformable_sphere",
        "procedural_deformable_cube",
        "procedural_deformable_volume_block",
        "procedural_deformable_cloth",
        "procedural_deformable_cable",
    ):
        assert reg.is_registered(asset_name)
        assert "deformable" in reg.get_asset_by_name(asset_name)().tags


def test_object_hierarchy_reparented() -> None:
    """DeformableObject is a peer of the spawnable objects, not an Object; rigid objects are Spawnable."""
    from isaaclab_arena.assets.deformable_object import DeformableObject
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectBase, SpawnableObjectBase
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_set import RigidObjectSet

    assert issubclass(DeformableObject, ObjectBase)
    assert not issubclass(DeformableObject, SpawnableObjectBase)
    assert not issubclass(DeformableObject, Object)
    assert issubclass(Object, SpawnableObjectBase)
    assert issubclass(ObjectReference, SpawnableObjectBase)
    assert issubclass(RigidObjectSet, Object)


def test_deformable_sphere_cfg_type() -> None:
    """The sphere's object cfg is a soft-body PresetCfg that resolves to a DeformableObjectCfg."""
    from isaaclab.assets import DeformableObjectCfg
    from isaaclab_tasks.utils import PresetCfg
    from isaaclab_tasks.utils.hydra import resolve_presets

    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.utils.pose import Pose

    sphere = AssetRegistry().get_asset_by_name("procedural_deformable_sphere")()
    sphere.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1)))

    assert sphere.object_type == ObjectType.DEFORMABLE
    assert isinstance(sphere.object_cfg, PresetCfg)

    # Soft-body objects only enumerate soft-body presets; ``default`` maps to the soft default (not
    # PhysX). Both resolve to a DeformableObjectCfg with the initial pose stamped on.
    for backend in ("newton_mjwarp_vbd", "default"):
        cfg = resolve_presets(sphere.object_cfg, selected=(backend,))
        assert isinstance(cfg, DeformableObjectCfg)
        assert cfg.init_state.pos == (0.4, 0.0, 0.1)
    # A nodal reset event is generated for the deformable.
    assert sphere.get_event_cfg()[1] is not None


def test_deformable_spawn_uses_pretet_usd() -> None:
    """The soft-body spawn comes from the committed pre-tetrahedralized TetMesh USD (no pytetwild)."""
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab_tasks.utils.hydra import resolve_presets

    from isaaclab_arena.assets.registries import AssetRegistry

    for asset_name, tet_file in (
        ("procedural_deformable_sphere", "procedural_deformable_sphere_tet.usda"),
        ("procedural_deformable_cube", "procedural_deformable_cube_tet.usda"),
        ("procedural_deformable_volume_block", "procedural_deformable_volume_block_tet.usda"),
        ("procedural_deformable_cable", "procedural_deformable_cable_tet.usda"),
    ):
        asset = AssetRegistry().get_asset_by_name(asset_name)()
        for backend in ("newton_mjwarp_vbd", "default"):
            cfg = resolve_presets(asset.object_cfg, selected=(backend,))
            assert isinstance(cfg.spawn, UsdFileCfg), f"{asset_name}/{backend} is not a UsdFileCfg"
            assert cfg.spawn.usd_path.endswith(tet_file), f"{asset_name}/{backend} not pointing at {tet_file}"


def test_pick_and_place_task_uses_geometry_for_deformable() -> None:
    """PickAndPlaceTask uses generic geometry success when the pickup object has no contact sensor.

    The background is stubbed to its ``object_min_z`` (the only field the task reads): constructing a
    real ``Background`` eagerly opens its remote USD, which requires a running SimulationApp.
    """
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.metrics.success_rate import SuccessRateMetric
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask, TerminationsCfg
    from isaaclab_arena.tasks.predicates.spatial import object_is_below_height, object_on_destination_by_geometry

    reg = AssetRegistry()
    sphere = reg.get_asset_by_name("procedural_deformable_sphere")()
    bowl = reg.get_asset_by_name("bowl_ycb_robolab")()
    background = types.SimpleNamespace(name="maple_table_robolab", object_min_z=-0.2)

    task = PickAndPlaceTask(
        pick_up_object=sphere,
        destination_location=bowl,
        background_scene=background,
    )

    termination_cfg = task.get_termination_cfg()
    success_predicates = termination_cfg.success.params["predicates"]

    assert task.success_strategy == "geometry"
    assert task.contact_sensor_name is None
    assert task.get_scene_cfg() is None
    assert task.get_events_cfg() is None
    assert isinstance(termination_cfg, TerminationsCfg)
    assert len(success_predicates) == 1
    assert success_predicates[0].func is object_on_destination_by_geometry
    assert success_predicates[0].params["velocity_threshold"] == task.velocity_threshold
    assert termination_cfg.object_dropped.func is object_is_below_height
    assert isinstance(task.get_metrics()[0], SuccessRateMetric)


def test_pick_and_place_task_keeps_contact_success_for_contact_objects() -> None:
    from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg

    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.tasks.predicates.spatial import object_is_below_height, object_on_destination

    class ContactObject:
        name = "rigid_object"
        prim_path = "{ENV_REGEX_NS}/rigid_object"

        def supports_contact_sensor(self):
            return True

        def get_contact_sensor_cfg(self, contact_against_object=None):
            return ContactSensorCfg(prim_path=self.prim_path)

    pick_up_object = ContactObject()
    destination = types.SimpleNamespace(name="bowl", prim_path="{ENV_REGEX_NS}/bowl")
    background = types.SimpleNamespace(name="table", object_min_z=-0.2)

    task = PickAndPlaceTask(
        pick_up_object=pick_up_object,
        destination_location=destination,
        background_scene=background,
    )
    success_predicates = task.get_termination_cfg().success.params["predicates"]

    assert task.success_strategy == "contact"
    assert task.contact_sensor_name == "contact_sensor_rigid_object"
    assert task.get_scene_cfg() is not None
    assert len(success_predicates) == 1
    assert success_predicates[0].func is object_on_destination
    assert task.get_termination_cfg().object_dropped.func is object_is_below_height


def test_deformable_environment_in_cli_registry() -> None:
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena_environments.cli import ensure_environments_registered

    ensure_environments_registered()
    env_registry = EnvironmentRegistry()
    assert env_registry.is_registered("deformable_sphere_pick_place")
    assert env_registry.get_component_by_name("deformable_sphere_pick_place").name == "deformable_sphere_pick_place"


def test_deformable_physics_backend_selection() -> None:
    """ArenaEnvBuilder defaults generic deformable scenes to Newton VBD and validates explicit presets."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    sphere = AssetRegistry().get_asset_by_name("procedural_deformable_sphere")()
    builder = object.__new__(ArenaEnvBuilder)
    builder.arena_env = types.SimpleNamespace(scene=types.SimpleNamespace(assets={"object": sphere}))

    assert builder._scene_needs_soft_body() is True

    # Non-soft-body presets fail before launch with a clear error.
    for preset in ("default", "newton"):
        with pytest.raises(NotImplementedError, match="soft-body"):
            builder._select_backend_preset(preset, needs_soft_body=True)
    with pytest.raises(NotImplementedError, match="does not support"):
        builder._select_backend_preset("newton_mjwarp_vbd_surface", needs_soft_body=True)

    # No preset on a generic soft-body scene -> the global Newton VBD default.
    assert builder._select_backend_preset(None, needs_soft_body=True) == "newton_mjwarp_vbd"

    # Validated volume soft-body presets are passed through unchanged.
    assert builder._select_backend_preset("physx", needs_soft_body=True) == "physx"
    assert builder._select_backend_preset("newton_mjwarp_vbd", needs_soft_body=True) == "newton_mjwarp_vbd"
    assert builder._select_backend_preset("newton_mjwarp_vbd_proxy", needs_soft_body=True) == "newton_mjwarp_vbd_proxy"

    # Rigid-only scenes with no preset stay on the stock PhysX spawn.
    assert builder._select_backend_preset(None, needs_soft_body=False) is None


def test_surface_deformables_can_use_surface_preset() -> None:
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    cloth = AssetRegistry().get_asset_by_name("procedural_deformable_cloth")()
    builder = object.__new__(ArenaEnvBuilder)
    builder.arena_env = types.SimpleNamespace(scene=types.SimpleNamespace(assets={"cloth": cloth}))

    assert builder._scene_soft_body_kinds() == frozenset({"surface"})
    assert (
        builder._select_backend_preset("newton_mjwarp_vbd_surface", needs_soft_body=True) == "newton_mjwarp_vbd_surface"
    )


def test_cable_uses_volume_soft_body_preset() -> None:
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    cable = AssetRegistry().get_asset_by_name("procedural_deformable_cable")()
    builder = object.__new__(ArenaEnvBuilder)
    builder.arena_env = types.SimpleNamespace(scene=types.SimpleNamespace(assets={"cable": cable}))

    assert builder._scene_soft_body_kinds() == frozenset({"volume"})
    with pytest.raises(NotImplementedError, match="does not support"):
        builder._select_backend_preset("newton_mjwarp_vbd_surface", needs_soft_body=True)
    assert builder._select_backend_preset("newton_mjwarp_vbd", needs_soft_body=True) == "newton_mjwarp_vbd"


def _test_deformable_sphere_newton_smoke(simulation_app) -> bool:
    """Boot a minimal deformable scene and check the default Newton VBD path."""
    import torch

    from isaaclab.managers import SceneEntityCfg

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.terms.events import set_deformable_object_pose
    from isaaclab_arena.utils.pose import Pose

    registry = AssetRegistry()
    sphere = registry.get_asset_by_name("procedural_deformable_sphere")()
    sphere.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.2)))
    ground = registry.get_asset_by_name("ground_plane")()
    embodiment = registry.get_asset_by_name("franka_joint_pos")()
    arena_env = IsaacLabArenaEnvironment(
        name="minimal_deformable_sphere",
        scene=Scene([ground, sphere]),
        embodiment=embodiment,
    )
    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False))
    env = builder.make_registered().unwrapped

    def set_sphere_pose(position_xyz: tuple[float, float, float]) -> None:
        set_deformable_object_pose(
            env,
            env_ids=torch.tensor([0], device=env.device),
            asset_cfg=SceneEntityCfg("procedural_deformable_sphere"),
            pose=Pose(position_xyz=position_xyz),
        )

    try:
        # Newton VBD backend was actually selected.
        assert env.cfg.scene.replicate_physics is True

        env.reset()
        asset = env.scene["procedural_deformable_sphere"]
        nodal_before = asset.data.nodal_pos_w.torch.clone()
        assert nodal_before.shape[1] > 0, "deformable has no simulation nodes"
        assert torch.isfinite(nodal_before).all(), "nodal positions not finite after reset"

        hold_action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        for _ in range(15):
            env.step(hold_action)

        nodal_after = asset.data.nodal_pos_w.torch
        assert torch.isfinite(nodal_after).all(), "nodal positions diverged (non-finite) after stepping"
        # The VBD solver must actually advance the soft body under gravity/contact.
        max_delta = (nodal_after - nodal_before).abs().max().item()
        assert max_delta > 1e-5, f"deformable did not move under Newton stepping (max delta {max_delta})"

        set_sphere_pose((0.1, 0.0, 0.2))
        for _ in range(2):
            env.step(hold_action)
        moved_centroid = asset.data.root_pos_w.torch[0]
        assert torch.isfinite(moved_centroid).all(), "deformable centroid not finite after explicit reset"
        assert torch.allclose(moved_centroid[:2], torch.tensor([0.1, 0.0], device=env.device), atol=0.02)
    finally:
        env.close()
    return True


def _test_surface_cable_and_volume_newton_smoke(simulation_app) -> bool:
    import torch

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    def assert_deformables_step(arena_env, preset: str, asset_names: tuple[str, ...]) -> None:
        builder = ArenaEnvBuilder(
            arena_env,
            ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, presets=preset),
        )
        env = builder.make_registered().unwrapped
        try:
            env.reset()
            assert env.cfg.scene.replicate_physics is True
            hold_action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
            before = {}
            for name in asset_names:
                asset = env.scene[name]
                before[name] = asset.data.nodal_pos_w.torch.clone()
                assert before[name].shape[1] > 0
                assert torch.isfinite(before[name]).all()

            for _ in range(10):
                env.step(hold_action)

            for name in asset_names:
                after = env.scene[name].data.nodal_pos_w.torch
                assert torch.isfinite(after).all()
                assert (after - before[name]).abs().max().item() > 1.0e-6
        finally:
            env.close()

    registry = AssetRegistry()
    ground = registry.get_asset_by_name("ground_plane")()
    cloth = registry.get_asset_by_name("procedural_deformable_cloth")()
    volume_block = registry.get_asset_by_name("procedural_deformable_volume_block")()
    cable = registry.get_asset_by_name("procedural_deformable_cable")()
    embodiment = registry.get_asset_by_name("franka_joint_pos")()
    cloth.set_initial_pose(Pose(position_xyz=(-0.2, 0.0, 0.5)))
    volume_block.set_initial_pose(Pose(position_xyz=(0.0, 0.25, 0.5)))
    cable.set_initial_pose(Pose(position_xyz=(0.25, 0.0, 0.5)))

    arena_env = IsaacLabArenaEnvironment(
        name="minimal_surface_deformables",
        scene=Scene([ground, cloth, volume_block, cable]),
        embodiment=embodiment,
    )
    assert_deformables_step(arena_env, "newton_mjwarp_vbd", (cloth.name, volume_block.name, cable.name))

    cloth_surface = registry.get_asset_by_name("procedural_deformable_cloth")(instance_name="cloth_surface_only")
    cloth_surface.set_initial_pose(Pose(position_xyz=(-0.2, 0.0, 0.5)))
    surface_env = IsaacLabArenaEnvironment(
        name="minimal_surface_preset_deformable",
        scene=Scene([registry.get_asset_by_name("ground_plane")(), cloth_surface]),
        embodiment=registry.get_asset_by_name("franka_joint_pos")(),
    )
    assert_deformables_step(surface_env, "newton_mjwarp_vbd_surface", (cloth_surface.name,))
    return True


def _test_physx_deformable_visual_proxy_tracks_sim(simulation_app) -> bool:
    import torch

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    registry = AssetRegistry()
    ground = registry.get_asset_by_name("ground_plane")()
    block = registry.get_asset_by_name("procedural_deformable_volume_block")()
    block.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.35)))
    embodiment = registry.get_asset_by_name("franka_joint_pos")()

    def enable_physx_visual_sync(env_cfg):
        env_cfg.sync_deformable_visual_meshes_from_sim = True
        return env_cfg

    arena_env = IsaacLabArenaEnvironment(
        name="minimal_physx_deformable_visual_proxy",
        scene=Scene([ground, block]),
        embodiment=embodiment,
        env_cfg_callback=enable_physx_visual_sync,
    )
    env = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, presets="physx"),
    ).make_registered(render_mode="rgb_array")
    base = env.unwrapped
    try:
        env.reset()
        syncs = base._deformable_visual_mesh_syncs  # noqa: SLF001
        assert len(syncs) == 1
        assert len(syncs[0].proxy_prims) == 1
        assert len(syncs[0].proxy_translate_ops) == 1

        start_translate = syncs[0].proxy_translate_ops[0].Get()
        start_z = float(start_translate[2])
        action = torch.zeros(env.action_space.shape, device=base.device)
        for _ in range(45):
            env.step(action)

        after_translate = syncs[0].proxy_translate_ops[0].Get()
        after_z = float(after_translate[2])
        assert after_z < start_z - 0.01, f"PhysX visual proxy did not follow falling deformable: {start_z} -> {after_z}"
    finally:
        env.close()
    return True


@pytest.mark.with_subprocess
def test_deformable_sphere_newton_smoke() -> None:
    assert run_simulation_app_function(
        _test_deformable_sphere_newton_smoke,
        headless=HEADLESS,
    )


@pytest.mark.with_subprocess
def test_surface_cable_and_volume_newton_smoke() -> None:
    assert run_simulation_app_function(
        _test_surface_cable_and_volume_newton_smoke,
        headless=HEADLESS,
    )


@pytest.mark.with_subprocess
def test_physx_deformable_visual_proxy_tracks_sim() -> None:
    assert run_simulation_app_function(
        _test_physx_deformable_visual_proxy_tracks_sim,
        headless=HEADLESS,
        enable_cameras=True,
    )
